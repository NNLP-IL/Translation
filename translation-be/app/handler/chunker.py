from langchain.text_splitter import RecursiveCharacterTextSplitter
from .sentence.sentence_combiner import SentenceCombiner
from typing import List, Optional
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel
from consts import LOGGER_CONFIG_PATH
from utils.print_colors import *
import re
import os

class Chunker:
    SENTENCE_SEPARATORS: str = os.getenv("SENTENCE_SEPARATORS", "?<=\.")

    def __init__(self, tokenizer: str = None, chunk_size: int = 60, embedder_name: str = 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli'):
        self.logger = NRMLogger(logger_name="Chunker", config_path=LOGGER_CONFIG_PATH)
        # Load external tokenizer
        self.tokenizer = tokenizer
        # Create a RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer,
                                                                                       chunk_size=chunk_size,
                                                                                       chunk_overlap=0,
                                                                                       separators=['\n', '. ', '،'],
                                                                                       keep_separator=False)
        self.chunk_size = chunk_size
        self.sentence_combiner = SentenceCombiner(embedder_name=embedder_name)
    
    def split_sentences_by_words(self, words: List[str]):
        total_chunks, current_chunk = [], []
        current_size = 0
        for word in words:
            word_tokens = self.tokenizer.tokenize(word) 
            if current_size + len(word_tokens) <= self.chunk_size:
                # If adding this word doesn't exceed the limit, add it to the current chunk
                current_chunk.append(word)
                current_size += len(word_tokens)
            else:
                # If the current word exceeds the limit, save the current chunk and start a new one
                total_chunks.append(' '.join(current_chunk))  
                current_chunk = [word]  # Start a new chunk with the current word
                current_size = len(word_tokens)  # Reset the size counter to the current word's size
        if current_chunk:
            total_chunks.append(' '.join(current_chunk))
        return total_chunks
    
    def split_sentences_by_space(self, sentences: List[str]):
        """
        Splits sentences into multiple sentences based on the limit of tokens.
        Parameters:
        - sentences (List[str]): The original sentences to be split.
        Returns:
        List[str]: A list containing the original or split sentences.
        """
        total_chunks = []
        for sen in sentences:
            tokens = self.tokenizer.tokenize(sen)
            if len(tokens) <= self.chunk_size:
                total_chunks.append(sen)
            else:
                words = sen.split() 
                total_chunks.extend(self.split_sentences_by_words(words=words))
        return total_chunks
    
    def split_paragraph2chunks(self, paragraph: str):
        # Remove empty and add '.'
        senetnces: List[str] = re.split(f'({self.SENTENCE_SEPARATORS})', paragraph)
        cmb_splitted_src: List[str] = self.sentence_combiner.combine(array_text=senetnces, threshold=0.35)
        paragraph_chunks = []
        for chunk in cmb_splitted_src:
            new_chunks = self.text_splitter.split_text(chunk)
            # in case there are no punctuations split by space
            new_chunks = self.split_sentences_by_space(new_chunks)
            paragraph_chunks.extend(new_chunks)
        return paragraph_chunks
    
    @staticmethod
    def split_text2paragraphs(text: str):
        return [item for item in text.split('\n') if item.strip()]
    
    def split_text2chunks(self, text: str):
        # Split the text to paragraphs by '\n'
        paragraphs: List[str] = self.split_text2paragraphs(text=text)
        total_chunks, paragraphs_counter = [], []
        self.logger.log(f"Chunking {len(paragraphs)} paragraphs")
        for idx, paragraph in enumerate(paragraphs):
            paragraph_chunks: List[str] = self.split_paragraph2chunks(paragraph=paragraph)
            total_chunks.extend(paragraph_chunks)
            paragraphs_counter.extend([idx] * len(paragraph_chunks))
        return total_chunks, paragraphs_counter
