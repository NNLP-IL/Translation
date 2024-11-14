import argparse
import os
from typing import List
from tqdm import tqdm
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sentence_alignment import CustomSentenceAligner
from chunker import Chunker
from scripts.utils.general_utils import load_data, convert_data_to_list, list_to_file, SuppressPrint

import json
import time

class Translator():
    def __init__(self, tokenizer_path='Helsinki-NLP/opus-mt-ar-he', trans_model_path='../MODELS/ft_opus_pre_0.0',
                 chunk_size=60, alignment_method='itermax',
                 device='cuda'):
        self.device = device
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Tokenizer loaded")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_path, output_attentions=True)
        self.model.to(self.device)
        print(f"Translation model is loaded on the {self.device}")
        # Create a Chunker
        self.chunker = Chunker(tokenizer=self.tokenizer, chunk_size=chunk_size)
        print("Chunker loaded")
        # Load the Sentence Aligner
        self.myaligner = CustomSentenceAligner(matching_method=alignment_method, device=self.device)
        print(f"Alignment model is loaded on the {self.device} [{alignment_method} alignment]")

    def translate_one_sentence(self, text: str):
        """
        A simple translation - encode arab words, generate output using the language model, and then decode it to hebrew sentence
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(self, text: str | List[str], batch_size: int = 8):
        """
        Our translation function.
        This function supports as input string or list of strings, which we send to the translation model as a batch
        """
        if isinstance(text, str) or len(text) == 1:
            if isinstance(text, list):
                text = text[0]
            return [self.translate_one_sentence(text)]
        else:
            translated_texts = []
            print(f"batch size = {batch_size}")
            for i in tqdm(range(0, len(text), batch_size)):
                # Get the batch of texts
                batch_texts = text[i:i + batch_size]

                # Tokenize the batch
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)

                # Move inputs to the appropriate device (GPU if available)
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

                # Generate translations
                outputs = self.model.generate(**inputs)

                # Decode the outputs
                batch_translations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # Append the batch translations to the final list
                translated_texts.extend(batch_translations)

            return translated_texts

    def run(self, src_sentences: str | List[str], batch_size: int = 8):

        timing_info = {
            'total': [],
            'chunking': [],
            'translation': [],
            'alignment': []
        }

        if isinstance(src_sentences, str):
            src_sentences = [src_sentences]

        input_batches = []
        for i in range(0, len(src_sentences), batch_size):
            input_batches.append(convert_data_to_list(src_sentences[i:i + batch_size]))

        translations = []
        for batch in tqdm(input_batches):

            # --------------------------
            # Pre Process - Chunking
            # --------------------------
            start_time = time.time()

            chunks, chunks_mapping = [], []
            for sentence_idx, src_sentence in enumerate(batch):
                with SuppressPrint():
                    sentence_chunks, paragraphs_splits = self.chunker.split_text2chunks(src_sentence)
                chunks.extend([(sentence_chunks, paragraphs_splits)])
                chunks_mapping.extend([(sentence_idx, j) for j in range(len(sentence_chunks))])

            chunking_time = time.time() - start_time
            timing_info['chunking'].append(chunking_time)
            # --------------------------
            # Send to translation
            # --------------------------
            start_time = time.time()

            with SuppressPrint():
                total_chunks = sum([chunk[0] for chunk in chunks], [])
                total_paragraphs_splits = sum([chunk[1] for chunk in chunks], [])
                translated_chunks = self.translate(total_chunks, batch_size=batch_size)

            # Reassemble the translated chunks into the original sentences
            translated_texts = [''] * len(batch)
            for (original_idx, chunk_idx), translated_chunk, paragraphs_idx\
                    in zip(chunks_mapping, translated_chunks, total_paragraphs_splits):
                if translated_texts[original_idx]:
                    separator = '.'
                    if paragraphs_idx > total_paragraphs_splits[chunk_idx - 1]:
                        separator = '\n'
                    translated_texts[original_idx] += separator + translated_chunk
                else:
                    translated_texts[original_idx] = translated_chunk

            translation_time = time.time() - start_time
            timing_info['translation'].append(translation_time)
            # --------------------------
            # Post Process - Handling mixed-text with words alignment
            # --------------------------
            start_time = time.time()

            try:
                with SuppressPrint():
                    aligned_translated_texts = [self.myaligner.sentences_align(src_sentence, trans)
                                                for src_sentence, trans in zip(batch, translated_texts)]
            except:
                pass

            alignment_time = time.time() - start_time
            timing_info['alignment'].append(translation_time)
            timing_info['total'].append(chunking_time + translation_time + alignment_time)

            # Save to translations list
            translations.extend(aligned_translated_texts)

        if len(translations) == 0:
            return '', timing_info
        elif len(translations) == 1:
            return translations[0], timing_info
        return translations, timing_info

def main():
    parser = argparse.ArgumentParser(description="Send sentences to translation")
    parser.add_argument('src_file', type=str, help="File containing the source data.")
    parser.add_argument('output_dir', type=str, help="The output dir to save the translations")
    parser.add_argument('--model', type=str, default='../MODELS/ft_opus_pre_0.0', help="The model we use for translations")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for translation")
    parser.add_argument('--alignment_method', type=str, default='itermax', choices=["inter", "itermax", "mwmf"],
                        help="The alignment method we use, based on this paper: https://arxiv.org/pdf/2004.08728")
    parser.add_argument('--timing', action='store_true', help='whether to time the code (for batch)')

    args = parser.parse_args()
    batch_size = args.batch_size
    trans_model_path = args.model
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device to CUDA if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    src_sentences = convert_data_to_list(load_data(args.src_file), desired_field='source')

    # Create a translator
    translator = Translator(device=device, trans_model_path=trans_model_path, alignment_method=args.alignment_method)
    translations, timing_info = translator.run(src_sentences=src_sentences, batch_size=batch_size)

    if args.timing:
        # Calculate averages and prepare JSON structure
        timing_results = {
            "info": {
                "device": device,
                "translation model": trans_model_path,
                "batch size": batch_size,
                "alignment_method": args.alignment_method},
            "results": {}
        }
        for module, times in timing_info.items():
            avg_time = sum(times) / len(times)
            timing_results["results"][module] = {
                'raw': times,
                'average': avg_time
            }

        # Print average times (optional, you can keep or remove this)
        print(f"\nAverage time per batch (size = {batch_size}):")
        for module, data in timing_results["results"].items():
            print(f"{module.capitalize()}: {data['average']:.4f} seconds")

        # Save timing results to JSON file
        json_path = os.path.join(args.output_dir, 'timing_results.json')
        with open(json_path, 'w') as f:
            json.dump(timing_results, f, indent=2)
        print(f"\nTiming results saved to {json_path}")

    # Save translations
    list_to_file(input_list=translations, file_path=os.path.join(args.output_dir, 'translations.txt'))


if __name__ == "__main__":
    main()
