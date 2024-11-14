import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import load_data


class TranslationDataset(Dataset):
    """
    A class that represent the dataset. Suitable for huggingface training process (torch).
    """
    def __init__(self, target_file, source_file, tokenizer, max_length):
        self.source_sentences = load_data(source_file)
        self.target_sentences = load_data(target_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        """
        Get data (batch), performing tokenizing before returning.
        :param idx:
        :return:
        """
        source_sentences = self.source_sentences[idx]
        target_sentences = self.target_sentences[idx]
        # Tokenize the source sentence
        tokenized_input = self.tokenizer(source_sentences, padding="max_length", truncation=True,
                                         max_length=self.max_length, return_tensors="pt")
        # Tokenize the target sentence without padding applied
        tokenized_target = self.tokenizer(text_target=target_sentences, truncation=True, max_length=self.max_length,
                                          return_tensors="pt")
        input_ids_target = tokenized_target["input_ids"].squeeze().tolist()
        if not isinstance(input_ids_target, list):
            dummy_input_ids = [self.tokenizer.pad_token_id] * self.max_length
            tokenized_output = {
                "input_ids": torch.tensor(dummy_input_ids, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.tensor([self.tokenizer.pad_token_id] * self.max_length, dtype=torch.long),
                "source_text": source_sentences  # Include the original source sentence text
            }
            print(f"idx {idx} input_ids_target is not a list.")
            return tokenized_output

        # Ensure everything is converted back to tensors
        tokenized_input["input_ids"] = tokenized_input["input_ids"].squeeze()  # Squeeze to remove the batch dimension
        tokenized_input["attention_mask"] = tokenized_input["attention_mask"].squeeze()
        tokenized_output = {
            "input_ids": tokenized_input["input_ids"],  # Source input ids
            "attention_mask": tokenized_input["attention_mask"],  # Source attention mask
            "labels": torch.tensor(input_ids_target, dtype=torch.long),  # Target input ids with -100 padding
            "source_text": source_sentences  # Include the original source sentence text
        }
        return tokenized_output

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        return [sentence.strip() for sentence in sentences]

    def find_max_and_avg_length(self):
        """
        :return: The max and average tokens number per sentence in the dataset (for both source and target sentences).
        """
        max_length_source = 0
        total_length_source = 0

        max_length_target = 0
        total_length_target = 0

        # Iterate over both source and target sentences simultaneously
        for source_sentence, target_sentence in zip(self.source_sentences, self.target_sentences):
            # Process source sentence
            tokenized_source = self.tokenizer(source_sentence, truncation=False, return_tensors="pt")
            length_source = tokenized_source["input_ids"].size(1)
            total_length_source += length_source
            if length_source > max_length_source:
                max_length_source = length_source

            # Process target sentence
            tokenized_target = self.tokenizer(text_target=target_sentence, truncation=False, return_tensors="pt")
            length_target = tokenized_target["input_ids"].size(1)
            total_length_target += length_target
            if length_target > max_length_target:
                max_length_target = length_target

        avg_length_source = total_length_source / len(self.source_sentences) if self.source_sentences else 0
        avg_length_target = total_length_target / len(self.target_sentences) if self.target_sentences else 0

        return {
            "max_length_source": max_length_source,
            "avg_length_source": avg_length_source,
            "max_length_target": max_length_target,
            "avg_length_target": avg_length_target
        }


def compute_metrics_with_args(tokenizer, bleu_metric, meteor_metric, comet_metric, val_dataset):
    """
    Evaluation of translation quality using BLEU, METEOR, and COMET metrics is performed during the training process.
    """
    sources = [example['source_text'] for example in val_dataset]

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute BLEU score
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Compute METEOR score
        # Flatten the decoded_labels for METEOR and COMET
        decoded_labels_flat = [label[0] for label in decoded_labels]
        meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels_flat)

        # Compute COMET score
        comet_result = comet_metric.compute(predictions=decoded_preds,
                                            references=decoded_labels_flat, sources=sources)
        return {
            "bleu": bleu_result["score"],
            "meteor": meteor_result["meteor"],
            "comet": comet_result["mean_score"]
        }

    return compute_metrics
