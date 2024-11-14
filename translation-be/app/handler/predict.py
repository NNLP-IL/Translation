import os
from typing import List, Union
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .loader.module_loader import *
from utils.general_utils import convert_data_to_list
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel
from consts import LOGGER_CONFIG_PATH, BATCH_SIZE 
from .objects.config import TranslatorConfig
from .objects.results import TranslatorInfo 
from .objects.map import TranslitMap
    
class Translator:
    def __init__(self, config: TranslatorConfig):
        """
        Initializes the Translator class.

        Args:
            config (TranslatorConfig): Configuration for the Translator.
            logger (NRMLogger): Logger instance for logging.
        """
        self.logger = NRMLogger(logger_name="Translator", config_path=LOGGER_CONFIG_PATH)
        self.tokenizer = self._load_component(TokenizerLoader(config.tokenizer_path), "Tokenizer")
        self.module_info = TranslatorInfo(device=config.device, translate_direction=config.translate_direction, translation_model=config.trans_model_path, alignment_method=config.alignment_method, batch_size=BATCH_SIZE)
        [setattr(self, attr.lower().replace(' ', '_'), self._load_component(loader, attr)) for attr, loader in self._get_compontents_map(config=config)]
        
        
    def _get_compontents_map(self, config: TranslatorConfig):
        translit_map: TranslitMap = TranslitMap(translate_direction=config.translate_direction)
        return [
            ("MyAligner", AlignerLoader(entity_extract=config.entity_extract, alignment_method=config.alignment_method, token_classify_model=config.token_classify_model, token_classify_target_model=config.token_classify_target_model, device=config.device, translit_map=translit_map)),
            ("Model", ModelLoader(model_path=config.trans_model_path, device=config.device)),
            ("Chunker", ChunkerLoader(tokenizer=self.tokenizer, chunk_size=config.chunk_size))  
        ]
        
    def _load_component(self, loader, component_name: str):
        try:
            component = loader.load()
            self.logger.log(f"{component_name} loaded")
            return component
        except LoaderError as e: 
            self.logger.log(message=f"Failed module loader: {e}", level=LogLevel.ERROR)
    
    def translate_one_sentence(self, text: str):
        """
        A simple translation - encode arab words, generate output using the language model, and then decode it to hebrew sentence
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate(self, text: str | List[str]):
        """
        Our translation function.
        This function supports as input string or list of strings, which we send to the translation model as a batch
        Returns: Tuple[str, str] -> translated_text, origin_text.
        """
        if isinstance(text, str):
            yield self.translate_one_sentence(text=text, model=self.model), text
        else:
            for i in tqdm(range(0, len(text), self.module_info.batch_size)): 
                batch_texts = text[i:i + self.module_info.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()} # Move inputs to the appropriate device (GPU if available)
                outputs = self.model.generate(**inputs)
                for i, output in enumerate(outputs):
                    yield self.tokenizer.decode(output, skip_special_tokens=True), batch_texts[i]

    @staticmethod
    def remove_chunks_end_dots(chunks: List[str]):
        end_dots_map: List[bool] = []
        for i, chunk in enumerate(chunks):
            end_dots_map.append(chunk.strip().endswith("."))
            chunks[i] = chunk.strip()[:-1] if end_dots_map[i] else chunk
        return chunks, end_dots_map
    
    def chunker_preprocess(self, input: List[str]):
        """ Pre Process - Chunking """
        self.logger.log("Starting chunk preprocess")
        chunks, chunks_mapping = [], []
        for sentence_idx, src_sentence in enumerate(input):
            sentence_chunks, paragraphs_splits = self.chunker.split_text2chunks(src_sentence)
            sentence_chunks, end_dots_map = self.remove_chunks_end_dots(chunks=sentence_chunks)
            chunks.extend([(sentence_chunks, paragraphs_splits, end_dots_map)])
            chunks_mapping.extend([(sentence_idx, j) for j in range(len(sentence_chunks))])
        return chunks, chunks_mapping
    
    @staticmethod
    def _parse_translate_chunk(translated_chunk: str, current_text: str, chunk_idx: int, paragraphs_idx: int, total_paragraphs_splits: List[int]):
            if current_text:
                separator = '.'
                if paragraphs_idx > total_paragraphs_splits[chunk_idx - 1]:
                    separator = '\n'
                current_text += separator + translated_chunk
            else:
                current_text = translated_chunk
            return current_text
        
    def _combine_translated_chunk(self, translated_chunks: List[str], total_paragraphs_splits: List[str], chunks_mapping: List[int], batch_length: int):
        """ Reassemble the translated chunks into the original sentences """
        translated_texts = [''] * batch_length
        for (original_idx, chunk_idx), translated_chunk, paragraphs_idx\
                    in zip(chunks_mapping, translated_chunks, total_paragraphs_splits):
            translated_texts[original_idx] = self._parse_translate_chunk(translated_chunk=translated_chunk, current_text=translated_texts[original_idx], chunk_idx=chunk_idx, 
                                        paragraphs_idx=paragraphs_idx, total_paragraphs_splits=total_paragraphs_splits)
        return translated_texts
        
    def _translate_chunks(self, chunks: List[tuple[str, int]]):
        """ Sends chunks to translation """
        total_chunks: List[str]  = sum([chunk[0] for chunk in chunks], [])
        total_paragraphs_splits: List[int] = sum([chunk[1] for chunk in chunks], []) 
        total_end_periods: List[bool] = sum([chunk[2] for chunk in chunks], [])
        translated_chunks_iter: List[str] = self._translate(text=total_chunks)
        return translated_chunks_iter, total_paragraphs_splits, total_end_periods
    
    def translate_chunks_texts(self, chunks: List[tuple[str, int]]):
        """ Translates chunks and combines translation results into chunks origin sentences. 
            Returns: __List[str]__ - translated origin sentences."""
        self.logger.log(f"Translating {len(chunks)} Chunks...")
        return self._translate_chunks(chunks=chunks)
   
    def aligned_translated_texts(self, batch: Union[str, List[str]], translated_texts: Union[str, list[str]], entity_align: bool):
        """ Post Process - Handling mixed-text with words alignment """
        self.logger.log(f"Aligns {len(translated_texts)} translated texts")
        if isinstance(translated_texts, list):
            return [self.myaligner.sentences_align(src_sentence=src_sentence, tgt_sentence=trans, entity_align=entity_align)
                                            for src_sentence, trans in zip(batch, translated_texts)]
        return self.myaligner.sentences_align(src_sentence=batch, tgt_sentence=translated_texts, entity_align=entity_align)
        
    @staticmethod
    def split_into_batch(input: str | List[str], batch_size: int):
        """ Splits inputs list[str]/str into multiple batch by max batch size."""
        input_batches = []
        input: List[str] = [input] if isinstance(input, str) else input
        for i in range(0, len(input), batch_size):
            input_batches.append(convert_data_to_list(input[i:i + batch_size]))
        return input_batches
    
    async def translate(self, src_sentences: str | List[str], entity_align: bool = True):
        self.logger.log(f"Start Translate: {src_sentences}, with batch size: {self.module_info.batch_size}")
        input_batches: List[str] = self.split_into_batch(input=src_sentences, batch_size=self.module_info.batch_size)
        for batch in tqdm(input_batches):
            chunks, chunks_mapping = self.chunker_preprocess(input=batch)
            translated_texts_iter, total_paragraphs_splits, total_end_periods = self.translate_chunks_texts(chunks=chunks)
            for i, (translated_text, origin_text) in enumerate(translated_texts_iter):
                try:
                    translated_text = translated_text.strip() + "." if total_end_periods[i] else translated_text
                    yield self.aligned_translated_texts(batch=origin_text, translated_texts=translated_text, entity_align=entity_align)
                except:
                    self.logger.log(message=f"Failed align translated texts", level=LogLevel.ERROR) 