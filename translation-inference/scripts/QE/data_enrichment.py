import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from model import TranslationQualityClassifier
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scripts.utils.print_colors import *
from scripts.utils.general_utils import load_config, load_data, list_to_file, split_into_batches


def translate(tokenizer, translation_model, inputs, tgt_lang='heb_Hebr', device='cpu'):
    # Tokenize the inputs as a batch
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate translations in batch
    translated_tokens = translation_model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                                                   max_length=1000)

    # Decode the translations
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


def save_mid_file(args, num_samples: int, new_src_sentences: list, new_translations: list):
    new_src_sentences = [sentence for sentence in new_src_sentences if sentence != '<DELETE>']
    new_translations = [sentence for sentence in new_translations if sentence != '<DELETE>']

    print("")
    print(f"Original Data size: {len(src_sentences)}")
    print(f"Data size after QE: {len(src_sentences) - len(bad_trans_indexes)}")
    print(f"Data size after Re-Translation (after {num_samples} bad samples): {len(new_src_sentences)}")

    # Save the new data
    src_file_name = f"{args.src_file.split('/')[-1].split('.')[0]}-{num_samples}_filtered-enriched.txt"
    trans_file_name = f"{args.trans_file.split('/')[-1].split('.')[0]}-{num_samples}_filtered-enriched.txt"
    list_to_file(new_src_sentences, os.path.join(args.output_dir, src_file_name))
    list_to_file(new_translations, os.path.join(args.output_dir, trans_file_name))
    print(f"Saved new data to: {args.output_dir}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filter and enrich our data")
    parser.add_argument('config_file', type=str, help="Config file for the QE and alternative translation model")
    parser.add_argument('src_file', type=str, help="File containing the source data.")
    parser.add_argument('trans_file', type=str, help="File containing google translations, aligned with the source sentences.")
    parser.add_argument('output_dir', type=str, help="The output dir to save the filtered data (two files"
                                                     "for source and translations)")
    parser.add_argument('--mask_file', type=str, default=None,
                        help="File containing 0s and 1s. The original preds from our QE.")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for QE")
    parser.add_argument('--src_lang', type=str, default='arb_Arab', help="source language")
    parser.add_argument('--tgt_lang', type=str, default='heb_Hebr', help="target language")

    # Parse the arguments
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{PRINT_START}{BLUE} Using {device} device{PRINT_STOP}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Config initialization
    config_path = args.config_file if os.path.isfile(args.config_file) else 'Configs/qe_config.yaml'
    config = load_config(config_path)
    saving_interval = config.get("save_interval", 10)
    translation_model_name = config.get("translation_model", "facebook/nllb-200-distilled-600M")

    # data
    src_sentences = load_data(args.src_file)
    translations = load_data(args.trans_file)
    batch_size = args.batch_size

    # setup the QE model
    qe_model = TranslationQualityClassifier(config=config_path)

    # Setup alternative translation model
    try:
        tokenizer = AutoTokenizer.from_pretrained(translation_model_name,
                                                  use_auth_token=config["huggingface_access_token"], src_lang=args.src_lang)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name,
                                                                  use_auth_token=config["huggingface_access_token"]).to(device)
        print(f"{PRINT_START}{BLUE}Translation model {translation_model_name} is loaded{PRINT_STOP}")
    except:
        print(f"{PRINT_START}{RED}Translation model {translation_model_name} failed to load, please check: \n"
              f"1. {translation_model_name} is a valid model with Huggingface page.\n"
              f"2. Your config file contains a valid huggingface access token.\n"
              f"3. The chosen model ({translation_model_name}) support {args.src_lang}{PRINT_STOP}"
              )
        print(f"{PRINT_START}{BLUE}Keep going without an alternative translation model\n"
              f"We'll just filter the original translations{PRINT_STOP}")
        translation_model = None

    # Load original QE prediction on all the data
    if args.mask_file is None:
        # We don't have QE prediction. that means we have to calc them for ourselves
        print(f"{PRINT_START}{RED}Going through the Dataset and estimate translations (batch_size = {args.batch_size}) ..{PRINT_STOP}")

        src_batches = split_into_batches(src_sentences, batch_size)
        trans_batches = split_into_batches(translations, batch_size)

        batch_preds = []
        for src_batch, trans_batch in tqdm(zip(src_batches, trans_batches)):
            outputs = qe_model.get_quality_scores(src_batch, trans_batch)
            batch_preds.append(outputs)
            all_qe_predictions = np.concatenate(batch_preds, axis=0)
        # save
        np.savetxt('qe_preds.txt', all_qe_predictions, fmt='%d')
    else:
        all_qe_predictions = load_data(args.mask_file)
        all_qe_predictions = np.array(all_qe_predictions, dtype=int)

    # Fetch only the bad translations and try to translate again
    new_src_sentences = src_sentences.copy()
    new_translations = translations.copy()
    print(f"{PRINT_START}{RED}Fetching all the bad translations ..{PRINT_STOP}")
    bad_trans_indexes = np.where(all_qe_predictions == 0)[0].tolist()
    if translation_model is not None:
        print(f"{PRINT_START}{RED}Trying to correct them (batch_size = {args.batch_size}) ..{PRINT_STOP}")
    else:
        print(f"{PRINT_START}{RED}Deleting them (batch_size = {args.batch_size}) ..{PRINT_STOP}")
    bad_trans_indexes_batches = split_into_batches(bad_trans_indexes, args.batch_size)
    saving_counter = 0
    for batch_idx, indexes_batch in tqdm(enumerate(bad_trans_indexes_batches)):
        src_batch = []
        for index in indexes_batch:
            src_batch.append(src_sentences[index])
        if translation_model is not None:
            trans_batch_alternatives = translate(tokenizer, translation_model, src_batch, tgt_lang=args.tgt_lang,
                                                 device=device)
            qe_scores_batch = qe_model.get_quality_scores(src_batch, trans_batch_alternatives)
            for i, (index, score) in enumerate(zip(indexes_batch, qe_scores_batch)):
                if score == 1:
                    # input the new translation
                    new_translations[index] = trans_batch_alternatives[i]
                else:
                    # "delete" the sentences from source and target lists
                    new_src_sentences[index] = '<DELETE>'
                    new_translations[index] = '<DELETE>'
        else:
            for index in indexes_batch:
                # "delete" the sentences from source and target lists
                new_src_sentences[index] = '<DELETE>'
                new_translations[index] = '<DELETE>'

        if ((saving_counter + 1) * batch_size >= saving_interval) or (batch_idx + 1 == len(bad_trans_indexes_batches)):
            save_mid_file(args, (batch_idx + 1)*batch_size, new_src_sentences, new_translations)
            saving_counter = 0
        else:
            saving_counter += 1
