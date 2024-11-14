import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from scripts.utils.general_utils import load_data, convert_data_to_list
from scripts.utils.plots_utils import plot_histogram


operators = {
    "model >= google": lambda row: row['model_AIXSTS'] >= row['google_AIXSTS'],
    "model < google": lambda row: row['model_AIXSTS'] < row['google_AIXSTS'],
    "low scores": lambda row: is_low_score(row['model_AIXSTS']) and is_low_score(row['google_AIXSTS']),
    "high scores": lambda row: (not is_low_score(row['model_AIXSTS'])) and (not is_low_score(row['google_AIXSTS'])),
    "model >= google (low scores)":
        lambda row: is_low_score(row['model_AIXSTS']) and is_low_score(row['google_AIXSTS']) and row[
        'model_AIXSTS'] >= row['google_AIXSTS'],
    "model < google (high scores)":
        lambda row: (not is_low_score(row['model_AIXSTS'])) and (
        not is_low_score(row['google_AIXSTS'])) and row['model_AIXSTS'] < row['google_AIXSTS'],
    "model low, google high":
        lambda row: (is_low_score(row['model_AIXSTS'])) and (not is_low_score(row['google_AIXSTS'])),
    "model high, google low":
        lambda row: (not is_low_score(row['model_AIXSTS'])) and (is_low_score(row['google_AIXSTS'])),
    "large diff": lambda row: abs(row['model_AIXSTS'] - row['google_AIXSTS']) >= 3
}


def is_low_score(score: int):
    return score <= 3


def calc_xsts_data_distribution(xsts_metadata: pd.DataFrame, calc_percentage: bool = False):

    # Calculate percentages of each use-case (operator)
    results, usecases_idexes = {}, {}
    for usecase, func in operators.items():
        is_op_true = xsts_metadata.apply(func, axis=1)
        results[usecase] = is_op_true.sum()
        if calc_percentage:
            results[usecase] = 100 * is_op_true.sum() / len(xsts_metadata)
        usecases_idexes[usecase] = list(is_op_true[is_op_true].index)
        print(f"{usecase}: {results[usecase]:.2f}")

    return results, usecases_idexes


def main():
    parser = argparse.ArgumentParser(description="sample data and create XSTS form for human annotators")
    parser.add_argument('--config_file', type=str, required=True, help='Path to a json/yaml file.')
    parser.add_argument('--n_samples', type=int, default=700, help='number of random samples that are taken')
    parser.add_argument('--include_aixsts_results', action='store_true',
                        help='whether to save also AI-XSTS eval results or not')

    args = parser.parse_args()
    config = load_data(args.config_file)
    # Extract parameters from the configuration
    src_path = config.get('src_path', None)
    google_trans_path = config.get('google_res_path', None)
    trans_res_path = config.get('trans_res_path', None)
    aixsts_outputh_path = config.get('ai_xsts_outputh_path', None)

    paths = [src_path, google_trans_path, trans_res_path]
    if any([(path is None) or (not os.path.isfile(path)) for path in paths]):
        raise IOError("At least one of the paths in config is not set properly, check them")

    # create the output directories
    output_path = 'XSTS_forms'
    os.makedirs(output_path, exist_ok=True)

    # load the data
    data_keys = ["source", "google translate", "generated"]
    if args.include_aixsts_results and os.path.isdir(aixsts_outputh_path):
        for model_type in ["google", "model"]:
            paths.append(os.path.join(aixsts_outputh_path, f"{model_type}_eval.csv"))
            data_keys.append(f"{model_type}_AIXSTS")
    data = {key: convert_data_to_list(load_data(path, supported_formats=['txt', 'csv', 'xls', 'xlsx']), desired_field=key)
            for key, path in zip(data_keys, paths)}

    # Get the length of the shortest list
    min_length = min(len(v) for v in data.values())
    # Create a DataFrame with truncated lists
    data_df = pd.DataFrame({k: list(v)[:min_length] for k, v in data.items()})

    # Sample rows from this dataframe
    data_df_random_sampled = data_df.sample(n=args.n_samples, random_state=42)

    # Add samples from the data that is classified as 'large diff'
    is_large_diff = data_df.apply(operators['large diff'], axis=1)
    large_diff_indxs = list(is_large_diff[is_large_diff].index)
    rows_to_add = data_df.loc[large_diff_indxs]
    data_df_random_sampled = pd.concat([data_df_random_sampled, rows_to_add]).drop_duplicates()
    # shuffle
    data_df_random_sampled = data_df_random_sampled.sample(frac=1)

    # Plot data distribution
    xsts_dist, xsts_indxs = calc_xsts_data_distribution(data_df_random_sampled, calc_percentage=False)
    data_dist, _ = calc_xsts_data_distribution(data_df, calc_percentage=False)
    plot_histogram([data_dist, xsts_dist], os.path.join(output_path, "xsts_dist-counts.png"),
                   x_label='Use Cases', y_label='Counts', title='Amount for Different Use Cases',
                   legend=['Full Data', 'Sampled XSTS'])

    # Randomly choose which column would be candidate1, the not chosen one would be candidate2
    trans_options = ["generated", "google translate"]
    data_df_random_sampled["candidate1"] = data_df_random_sampled.apply(lambda row: np.random.choice(trans_options), axis=1)
    data_df_random_sampled["candidate2"] = data_df_random_sampled.apply(lambda row: [v for v in trans_options if v != row['candidate1']][0], axis=1)

    # Add new columns for use-cases to the XSTS metadata
    for usecase, indxs in xsts_indxs.items():
        data_df_random_sampled[usecase] = data_df_random_sampled.index.isin(indxs).astype(int)

    # Now that we have all we need - let's create the XSTS form
    xsts_form = pd.DataFrame(columns=['sentence', 'candidate1', 'rank1', 'candidate2', 'rank2'])
    for i, row in tqdm(data_df_random_sampled.iterrows(), total=len(data_df_random_sampled)):
        cand1 = row['candidate1']
        cand2 = row['candidate2']
        xsts_form.loc[len(xsts_form)] = [row['source'], row[cand1], '', row[cand2], '']

    # Save
    xsts_form.to_csv(os.path.join(output_path, "xsts_form.csv"), index=False)
    data_df_random_sampled.to_csv(os.path.join(output_path, "xsts_metadata.csv"), index_label='sentence_ind')



if __name__ == "__main__":
    main()