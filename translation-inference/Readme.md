# Arabrew Translator

## Overview

**Arabrew Translator** is a comprehensive project for translating text from Arabic to Hebrew. This project includes all the necessary code for preparing the data, training the model, performing inference, evaluating the model, and more. Aim to provide a robust and accurate translation model that can be easily integrated into various applications.

## Project Structure

```plaintext
Arabrew Translator/
│
├── assets/
└── scripts/
    ├── prepare_data/
    ├── train/
    ├── eval/
    ├── inference/
    └── QE/
```

## Features

- **Data Preparation**: Scripts to preprocess and clean the data.
- **Model Training**: Code to define and train the translation model.
- **Evaluation**: Scripts to assess translation quality.
- **Inference**: Fast and efficient translation pipeline for real-world applications.
- **Quality Estimation (QE)**: Advanced techniques to estimate the quality of translations without reference texts.


### Installation

1. Create virtual environment.

   Ensure you have the following prerequisites installed:

   `Python 3.10 >= `

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Inference

```
python3 /scripts/inference/predict.py <src_file> <output_dir> [options]
```

Required Arguments:
- `src_file`: Path to the file containing the source data.
- `output_dir`: Directory to save the translations and timing results.

Optional Arguments:
- `--model`: Path to the translation model (default: '../MODELS/ft_opus_pre_0.0')
- `--batch_size`: Batch size for translation (default: 8)
- `--alignment_method`: Alignment method to use. Options: "inter", "itermax", "mwmf" (default: "itermax")
- `--timing`: Enable timing of the code (for batch processing)

Example 

```bash
python3 /scripts/inference/predict.py input.txt output_folder --model my_model --batch_size 16 --alignment_method mwmf --timing
```

This command will:
1. Read source sentences from `input.txt`
2. Use the model `my_model` for translation
3. Process in batches of 16
4. Use the "mwmf" alignment method
5. Save translations to `output_folder/translations.txt`
6. Save timing results (if `--timing` is used) to `output_folder/timing_results.json`

___

### Data Preparation

```bash
python3 scripts/prepare_data/clean_training_data.py --src_file_path source.txt --clean_file_path cleaned_text.txt
```

Navigate to the [scripts/prepare_data/](scripts/prepare_data/README.md) directory for detailed instructions.

### Training

```bash
python3 scripts/train/training_translation_model.py --config_file scripts/train/training_config_example.json --from_scratch
```

Refer to the [scripts/train/](scripts/train/README.md) directory for information on model architecture, hyperparameter tuning, and training procedures optimized for Arabic-Hebrew translation.

### Evaluation

```bash
python3 scripts/eval/eval.py
```

Navigate to the [scripts/eval/](scripts/eval/Readme.md) directory for detailed instructions.

### Quality Estimation

A pipeline that gets pairs of sentences (source-translation) and determine the quality of the translation.

This enables us to filter pairs from the data, and even to replace those
bad translation with alternative options from another non-google models.

Explore [scripts/QE/](scripts/QE/Readme.md) directory for more details.

___

## Contributing
We welcome contributions! Please read our Contributing Guidelines for more information on how to get started.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
* Thanks to all contributors who helped in developing this project.
* Special thanks to the open-source community for providing valuable resources.
