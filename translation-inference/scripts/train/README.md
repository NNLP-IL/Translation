# Translation Model Training with Hugging Face Transformers

This project utilizes the Hugging Face Transformers library to train a sequence-to-sequence model for language translation. Designed for flexibility, it supports training from scratch or fine-tuning a pre-trained model with custom datasets.

## Features

- Customizable training parameters through a JSON configuration file.
- Options for training from scratch or from a pre-trained model checkpoint.
- Evaluation of translation quality using BLEU, METEOR, and COMET metrics.

## Installation

Ensure you have the following prerequisites installed:

`Python 3.6 >= `

`PyTorch 1.8.1`

`Transformers 4.0+`

`evaluate`

## Configuration

Before running the training script, create a configuration file in JSON format with your training parameters and paths to your data. Here's an **example** of what this file should look like:

```
{
  "learning_rate": 2e-5,
  "num_train_epochs": 15,
  "per_device_train_batch_size": 32,
  "weight_decay": 0.01,
  "save_total_limit": 3,
  "predict_with_generate": true,
  "fp16": false,
  "save_strategy": "epoch",
  "eval_steps": 5500,
  "pretrained_model": "Helsinki-NLP/opus-mt-ar-he",
  "from_config": "Helsinki-NLP/opus-mt-ar-he",
  "tokenizer_dir": "",
  "target_txt_path": "../dataset/train_he.txt",
  "src_txt_path": "../dataset/train_ar.txt",
  "validation_ratio": 0.005,
  "validation_seed": 53,
  "output_dir": "../models",
  "model_name": "OPUS-FT",
  "model_version": "0.0",
  "max_length": 512
}
```

## Training Configuration Parameters

`learning_rate`: Steps size for updating model weights. Lower values slow down training but can improve results.

`num_train_epochs`: Total passes through the dataset.

`per_device_train_batch_size`: Number of examples processed in parallel on each device.

`weight_decay`: Regularization to limit overfitting by penalizing large weights.

`save_total_limit`: Maximum number of saved model checkpoints.

`predict_with_generate`: Whether to use the generate method for predictions.

`fp16`: Enables mixed precision training to reduce memory usage.

`save_strategy`: Checkpoint saving strategy, e.g., at every epoch.

`eval_steps`: Number of training steps between evaluations.

`pretrained_model`: Pretrained model identifier for initialization.

`from_config`: enables training a new model using the architecture from a checkpoint or HuggingFace Hub (when training from scratch).

`tokenizer_dir`: Directory for tokenizer files; If tokenizer_dir is not specified the tokenizer will be sourced from the pretrained model.

`target_txt_path and src_txt_path`: Paths to target and source training data.

`validation_ratio`: Fraction of training data used for validation.

`validation_seed`: Set the random seed to ensure repeatability.

`output_dir`: Where to save outputs.

`model_name`: Name for the trained model.

`model realized_version`: Model version.

`max_length`: Max sequence length for inputs to the model.

## Training

To train your model, run the script with the required --config_file argument specifying the path to your configuration file. Optionally, use --from_scratch to train from scratch instead of fine-tuning a pre-trained model.

If the --from_scratch option is not used, the base model will be sourced from the pretrained_model parameter in the config file. This could either be a path to a local model or a model name from the Huggingface hub.

```bash
python3 training_translation_model.py --config_file training_config_example.json --from_scratch
```