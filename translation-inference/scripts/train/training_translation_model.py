from transformers import AutoTokenizer, AutoConfig
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
import torch
from torch.utils.data import random_split
from training_dataset import TranslationDataset, compute_metrics_with_args
import evaluate
import random
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import load_data

SEED = 42
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
comet_metric = evaluate.load('comet', config_name='Unbabel/wmt22-comet-da')


def set_seed(seed_value):
    """
    Function to set seed in order to split the validation to the same set in different experiments
    """
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If you are using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_input_constraints(parser, args, config):
    """ Validating the arguments of the """
    if config.get("src_txt_path") is None or config.get("target_txt_path") is None:
        parser.error("please specify both of data paths: target_txt_path and src_txt_path.")

    if not args.from_scratch and config.get("pretrained_model") is None:
        parser.error("pretrained_model is required when training from a pretrained.")

    if args.from_scratch and config.get("from_config") is None:
        parser.error("from_config is required when training from scratch.")


def main():
    parser = argparse.ArgumentParser(description="Training an LM model using huggingface framework.")
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path of the config training process.')
    parser.add_argument('--from_scratch', action='store_true',
                        help='if training the model from scratch or training from a checkpoint.')
    args = parser.parse_args()

    # Example usage:
    # python3 training_translation_model.py --config_file training_config_example.json --from_scratch

    config = load_data(args.config_file)
    check_input_constraints(parser, args, config)

    MODEL_NAME = config.get("model_name", "trained_model")
    MODEL_VERSION = config.get("model_version", "0.00")
    MAX_LENGTH = config.get("max_length", 512)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for computations.")


    set_seed(config.get("validation_seed", SEED))

    # Step 1: Collect dataset
    source_file = config.get("src_txt_path")
    target_file = config.get("target_txt_path")

    # Step 2: Preprocess data
    tokenizer_dir = config.get("tokenizer_dir")
    if tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, return_tensors="pt")
    else:
        model_checkpoint_for_tokenizer = config.get("pretrained_model", config.get("from_config"))
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_for_tokenizer, return_tensors="pt")
    translation_dataset = TranslationDataset(target_file, source_file, tokenizer, MAX_LENGTH)

    print("len of source sentences: ", len(translation_dataset))
    validation_ratio = config.get("validation_ratio")  # X% of the data will be used for validation
    num_validation_samples = int(validation_ratio * len(translation_dataset))
    num_training_samples = len(translation_dataset) - num_validation_samples
    train_dataset, val_dataset = random_split(translation_dataset, [num_training_samples, num_validation_samples])

    # Step 3: Load a model to train
    if args.from_scratch:
        # Train model from scratch
        config_model = AutoConfig.from_pretrained(config.get("from_config"))
        model = AutoModelForSeq2SeqLM.from_config(config_model)
    else:
        # Train model from pretrained
        model = AutoModelForSeq2SeqLM.from_pretrained(config.get("pretrained_model"))

    model.to(device)  # Move the model to the device
    print("model is loaded to GPU!")

    # Step 4: Configure training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get("output_dir", ".") + "/",
        save_strategy=config.get("save_strategy", "epoch"),
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 32),
        weight_decay=config.get("weight_decay", 0.001),
        save_total_limit=config.get("save_total_limit", 3),
        num_train_epochs=config.get("num_train_epochs", 15),
        per_device_eval_batch_size=config.get("per_device_train_batch_size", 32),
        predict_with_generate=config.get("predict_with_generate", True),
        fp16=config.get("fp16", True),
        evaluation_strategy="steps",
        eval_steps=config.get("eval_steps", 5000)  # Perform evaluation every X steps,
    )

    compute_metrics = compute_metrics_with_args(tokenizer, bleu_metric, meteor_metric, comet_metric, val_dataset)

    # Step 5: Train the model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    torch.cuda.empty_cache()

    print("----------------train-model-" + MODEL_NAME + "---------------")
    trainer.train()

    model_path = config.get("output_dir", ".") + "/" + MODEL_NAME + "_" + MODEL_VERSION
    model.save_pretrained(model_path)


if __name__ == '__main__':
    main()
