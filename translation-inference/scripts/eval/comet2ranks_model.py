import argparse
import os.path
import pandas as pd
import mord as m
import pickle

from scripts.utils.general_utils import load_data
from scripts.utils.plots_utils import cm, plot_corrs


def fit_ord_logistic_model(X, y):

    # Initialize and fit the Ordinal Logistic Regression Model
    model = m.LogisticAT(alpha=0)  # LogisticAT is an ordinal logistic regression model
    model.fit(X, y)

    # Predict rankings for the same dataset (or replace X with new data for predictions)
    predicted_rankings = model.predict(X)

    return model, predicted_rankings


def dump_model(model, save_path):
    """
    dump the model that predicts comet ranking
    """
    # Saving the model to a file
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Training a simple classifier to convert COMET scores to XSTS rankings")
    parser.add_argument('xsts_data_file', type=str, help="csv file containing all the necessary data -"
                                                    "src sentences, translations (Google), ranks from human annotator")
    parser.add_argument('output_dir', type=str, default='.', help="The output dir to save the model and plots")
    parser.add_argument('--gt', type=str, default='google_rank2',
                        help="The column in the imported data file we use as our GT for evaluation")

    # Parse the arguments
    args = parser.parse_args()

    # Create output dir
    save_path = args.output_dir
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'vis'), exist_ok=True)

    # Load data
    xsts_df = load_data(args.xsts_data_file)
    try:
        google_scores = xsts_df["comet_google"]
        gts = xsts_df.get(args.gt, "google_rank2").astype(int)
    except:
        raise KeyError(f"At least one of this columns not exist in {args.xsts_data_file}:"
                       f" { {'comet_google', args.gt, 'google_rank2'} }")

    # Fitting the model
    X = google_scores.values.reshape(-1, 1)
    ord_log_model, ord_log_preds = fit_ord_logistic_model(X, gts)

    # Saving the model
    dump_model(ord_log_model, save_path=os.path.join(save_path, 'model/comet2ranks.pkl'))

    # ---------
    # Plots

    # Confusion Matrix
    cm(ord_log_preds, gts, model_name='Ordinal Logistic Regression', vis_res=os.path.join(save_path, 'vis'))

    # Correlations

    # List of keywords to check in column names alongside "google"
    keywords = ["comet", "cohere", "gpt", "rank"]
    # Select columns where names contain 'google' and any of the keywords, and data types are numeric
    filtered_columns = [
        col for col in xsts_df.columns
        if 'google' in col
           and any(keyword in col for keyword in keywords)
           and pd.api.types.is_numeric_dtype(xsts_df[col])
    ]
    google_df = xsts_df[filtered_columns]
    plot_corrs(google_df, save_path, "google", method="pearson")


if __name__ == '__main__':
    main()












