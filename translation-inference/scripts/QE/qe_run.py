import os
import argparse
import numpy as np
from tqdm import tqdm

from model import TranslationQualityClassifier
from scripts.utils.general_utils import load_data, split_into_batches


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Get QE predictions")
    parser.add_argument('config_file', type=str, help="Config file for the QE and alternative translation model")
    parser.add_argument('src_file', type=str, help="File containing the source data.")
    parser.add_argument('trans_file', type=str,
                        help="File containing google translations, aligned with the source sentences.")
    parser.add_argument('output_dir', type=str, default='.', help="The output dir to save the preds file")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size for QE")

    # Parse the arguments
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract data and split to batches
    src_sentences = load_data(args.src_file)
    translations = load_data(args.trans_file)
    batch_size = args.batch_size
    src_batches = split_into_batches(src_sentences, batch_size)
    trans_batches = split_into_batches(translations, batch_size)

    # Load the QE model
    config_path = args.config_file if os.path.isfile(args.config_file) else 'Configs/qe_config.yaml'
    qe_model = TranslationQualityClassifier(config=config_path)

    # Calculating the quality predictions for each batch and save them all together to one file
    batch_preds = []
    for src_batch, trans_batch in tqdm(zip(src_batches, trans_batches)):
        outputs = qe_model.get_quality_scores(src_batch, trans_batch)
        batch_preds.append(outputs)
        all_predictions = np.concatenate(batch_preds, axis=0)
        np.savetxt(os.path.join(args.output_dir, 'qe_preds.txt'), all_predictions, fmt='%d')


if __name__ == "__main__":
    main()

