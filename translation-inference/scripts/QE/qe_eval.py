import argparse
import torch
import os

from model import TranslationQualityClassifier
from scripts.utils.print_colors import *
from scripts.utils.plots_utils import cm, pr_optimal, roc_optimal, plot_corrs
from scripts.utils.general_utils import load_data, convert_data_to_list


def main():

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate a Quality Estimator based on specific ranker")
    parser.add_argument('config_file', type=str, help="Config file for the QE and alternative translation model")
    parser.add_argument('xsts_data_file', type=str, help="csv file containing all the necessary data -"
                                                    "src sentences, translations (Google), ranks from human annotator")
    parser.add_argument('output_dir', type=str, default='.', help="The output dir to save the plots")
    parser.add_argument('--gt', type=str, default='google_rank2',
                        help="The column in the imported data file we use as our GT for evaluation")
    parser.add_argument('--mask_file', type=str, default=None,
                        help="File containing 0s and 1s. The original preds from our QE.")

    # Parse the arguments
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{PRINT_START}{BLUE} Using {device} device{PRINT_STOP}")

    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    # Config initialization
    config_path = args.config_file if os.path.isfile(args.config_file) else 'Configs/qe_config.yaml'
    config = load_data(config_path)

    # Data
    xsts_df = load_data(args.xsts_data_file)
    try:
        src_sentences = convert_data_to_list(xsts_df["src"])
        translations = convert_data_to_list(xsts_df["google"])
        gts = xsts_df.get(args.gt, "google_rank2")
    except:
        raise KeyError(f"At least one of this columns not exist in {args.xsts_data_file}:"
                       f" {set(['src', 'google', args.gt, 'google_rank2'])}")

    # Getting the QE scores
    results = {}
    THRESHOLD = None
    if args.mask_file is not None and os.path.isfile(args.mask):
        qe_scores = load_data(args.mask_file)
        print(f"{PRINT_START}{GREEN}QE scores are loaded from {args.mask_file}{PRINT_STOP}")
    else:
        # Setup the QE model
        qe_model = TranslationQualityClassifier(config=config_path)
        THRESHOLD = qe_model.get("threshold", None)
        # Embeddings to FCNN Classifier
        model_outputs, qe_scores = qe_model.get_quality_scores(src_sentences, translations, return_raw_output=True)
        print(f"{PRINT_START}{GREEN}QE score have been calculated {PRINT_STOP}")

    results['embedding_google_FCNN'] = qe_scores

    # ---------
    # Plots

    for plot_name, preds in results.items():
        plot_title = f'{plot_name}@thr={THRESHOLD}' if THRESHOLD else plot_name

        # Confusion Matrix
        cm(preds, gts, plot_title, save_path)

        # Correlations
        ranks_df = xsts_df[['google_rank1', 'google_rank2', 'google_rank3',
                            'google_GT_google',
                            'google_gpt-4-turbo-preview']]
        ranks_df = ranks_df.iloc[len(ranks_df)-len(preds):]
        # add our predictions
        ranks_df['QE_ranks'] = preds
        plot_corrs(ranks_df, save_path, plot_title, method="pearson")

        # These graphs are only plotted if we calculate the QE scores in this run and do not get them from an external source
        if 'model_outputs' in locals():
            # ROC curve and find optimal treshold
            roc_optimal(model_outputs, gts, plot_name, save_path)

            # Precision-Recall curve and find optimal treshold
            pr_optimal(model_outputs, gts, plot_name, save_path)

        print(f"Plots for {plot_name} have been generated and saved to {save_path}")


if __name__ == "__main__":
    main()