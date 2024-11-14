import sys
import argparse
import os
from pathlib import Path
current_path = Path(os.getcwd())
parent_path = str(current_path.parent)
sys.path.append(parent_path)

from Configs.llms2use import models
from scripts.utils.plots_utils import plot_diff_hist, plot_accumulated_percentage
from scripts.utils.print_colors import *
from scripts.utils.eval_utils import *
from scripts.utils.general_utils import load_data, load_pkl_model, convert_data_to_list, list_to_file
from scripts.utils.llm_utils import send_df_to_llm


def sub_ranks_merge_4_5(ranks_cand2, ranks_cand1):
    delta_ranks = []
    for r2, r1 in zip(ranks_cand2, ranks_cand1):
        if (r2 == 4 and r1 == 5) or (r1 == 4 and r2 == 5):
            if isinstance(r2-r1, np.ndarray):
                delta_ranks.append(np.array([0]))
            else:
                delta_ranks.append(0)
        else:
            delta_ranks.append(r2 - r1)
    return np.array(delta_ranks)


def main():
    parser = argparse.ArgumentParser(description="Send data to Comet and/or AI-XSTS evaluation.")
    parser.add_argument('--config_file', type=str, default='Configs/config.json', help='Path to a config file (json).')
    parser.add_argument('--run_google', action='store_true', help='if run AI-XSTS also on google file or not')
    parser.add_argument('--run_aixsts', action='store_true', help='if run also AI-XSTS eval or not')
    parser.add_argument('--no_comet_score', action='store_true',
                        help='if calc COMET scores or not, By default we calculate COMET scores and ranks')
    parser.add_argument('--start_index', type=int, default=0,
                        help='The index of the row we want to start from (can be useful in cases were the process is stopped in the middle).')
    parser.add_argument('--stop_index', type=int, default=None,
                        help='The index of the row we want to stop in (can be useful in cases were we want to stop in the middle).')

    args = parser.parse_args()
    config = load_data(args.config_file)
    # Extract parameters from the configuration
    llm_output_path = config.get('llm_output_path', 'runs/llms_run/output')
    ai_xsts_outputh_path = config.get('ai_xsts_outputh_path', 'runs/AI_XSTS/output')
    comet_outputh_path = config.get('comet_outputh_path', 'runs/comet/output')
    plots_output_path = config.get('vis_result_eval', 'runs/vis_results')
    src_path = config.get('src_path', None)
    ref_path = config.get('ref_path', None)
    google_trans_path = config.get('google_res_path', None)
    trans_res_path = config.get('trans_res_path', None)
    comet2ranks_model_path = config.get('comet2ranks_model_path', 'models/model-comet2ranks.pkl')

    paths = [src_path, ref_path, google_trans_path, trans_res_path]
    if any([(path is None) or (not os.path.isfile(path)) for path in paths]):
        raise IOError("At least one of the paths in config is not set properly, check them")

    # Create the output directories
    outputs_paths = [llm_output_path, ai_xsts_outputh_path, comet_outputh_path, plots_output_path]
    if not args.run_aixsts:
        os.makedirs(outputs_paths[-1], exist_ok=True)
    else:
        for output_path in outputs_paths:
            os.makedirs(output_path, exist_ok=True)

    # Load the data
    data_keys = ["source", "target", "google translate", "generated"]
    data = {key: convert_data_to_list(load_data(path, supported_formats=['txt', 'csv', 'xls', 'xlsx']),
                                      desired_field=key)[args.start_index: args.stop_index]
            for key, path in zip(data_keys, paths)}
    data_df = pd.DataFrame(data)

    # 'google' refer to google translation and 'model' to our trained model
    translation_results = {"google": data_df["google translate"], "model": data_df["generated"]}

    comet_scores = {"google": None, "model": None}
    comet_ranks = {"google": None, "model": None}
    comet2ranks_model = load_pkl_model(comet2ranks_model_path)
    if not args.no_comet_score:
        # Get also the COMET scores for each translation type
        comet_trans_res = get_comet_scores(data["source"], data["target"], data["generated"])
        comet_google_res = get_comet_scores(data["source"], data["target"], data["google translate"])

        # Convert COMET scores into XSTS Ranks
        comet_trans_ranks = get_comet_rank(comet2ranks_model, comet_trans_res)
        comet_google_ranks = get_comet_rank(comet2ranks_model, comet_google_res)

        # Set to dictionary
        comet_scores.update({"google": comet_google_res, "model": comet_trans_res})
        comet_ranks.update({"google": comet_google_ranks, "model": comet_trans_ranks})

        # Save into txt files
        for model_type in comet_ranks.keys():
            list_to_file(input_list=convert_data_to_list(comet_scores[model_type]),
                         file_path=os.path.join(comet_outputh_path, f"{model_type}_comet_scores.txt"))
            list_to_file(input_list=convert_data_to_list(comet_ranks[model_type]),
                         file_path=os.path.join(comet_outputh_path, f"{model_type}_comet_ranks.txt"))

    if args.run_aixsts:
        for model_type, trans_res in translation_results.items():
            print(f"{PRINT_START}{RED}{model_type} translations are ranked{PRINT_STOP}")

            # STEP 1: check if ranking is already exist
            if os.path.isfile(os.path.join(outputs_paths[1], f"{model_type}_eval.csv")):
                print(f"{PRINT_START}{BLUE}{model_type} has been ranked before, we'll use previous results{PRINT_STOP}")
                continue

            google_prompt = False
            if model_type == "google":
                google_prompt = True
                if not args.run_google:
                    if os.path.isfile(os.path.join(ai_xsts_outputh_path, "google_eval.csv")):
                        # we don't need to run AI-XSTS on Google translation, we already have their ranks.
                        print(f"{PRINT_START}{BLUE}{model_type} has been ranked before, we'll use previous results{PRINT_STOP}")
                        continue
                    else:
                        print(f"{PRINT_START}{BLUE}You chose not to rank {model_type}{PRINT_STOP}")
                        continue

            # STEP 2: Send an adjusted prompt, based on the data, to LLM models (defined in models.py) and save the results
            print("Sending to LLMs for ranking .. ")
            base_prompt = base_aixsts_prompt(google_prompt)
            llms_outputs = send_df_to_llm(models, base_prompt, data_df, os.path.join(llm_output_path, f"{model_type}.csv"),
                                          validation_function=validate_aixstsx_response)

            # STEP 3: Calculate and convert COMET score into ranks
            print("Calculating COMET ranks .. ")
            if comet_ranks[model_type] is None:
                comet_res = get_comet_scores(data["source"], data["target"], convert_data_to_list(trans_res))
                comet_ranks[model_type] = get_comet_rank(comet2ranks_model, np.array(comet_res).reshape(-1, 1))

            # STEP 4: Average ranks from multiple sources (COMET ranks and the ones we got from the LLMs),and save the results
            # This step return the final ranks of the translation
            print("Calculating final ranks .. ")
            ai_xsts(comet_ranks=comet_ranks[model_type], llms_outputs=llms_outputs,
                    output_file=os.path.join(outputs_paths[1], f"{model_type}_eval.csv"))

        # Work with the results from AI-XSTS.
        # We should get two files - ranks for our model and for Google.
        ai_xsts_res = pd.read_csv(os.path.join(ai_xsts_outputh_path, "model_eval.csv"), header=None, index_col=False)
        ai_xsts_google_res = pd.read_csv(os.path.join(ai_xsts_outputh_path, "google_eval.csv"), header=None, index_col=False)

    # Plot and save histograms that show the subtraction between the two models
    # cand2: model, cand1: google
    # We plot 3 diff graphs:
    # 1) Diff between comet scores
    # 2) Diff between AI-XSTS rankings (if exists)
    # 3) Diff between comet scores, which were converted to rankings using designated trained model.
    print("Plotting and saving diff histograms .. ")
    pairs, hist_titles = [], []
    if args.run_aixsts:
        pairs.append((ai_xsts_google_res, ai_xsts_res))
        hist_titles.append("AI-XSTS")
        # concat the results to one df
        aixsts_results = pd.concat([ai_xsts_res, ai_xsts_google_res], axis=1)
        aixsts_results.columns = ['model', 'google']
        # plot accumulated percentage graph
        plot_accumulated_percentage(aixsts_results, save_path=os.path.join(plots_output_path, 'Accumulated_Percentage.png'))
    if not args.no_comet_score:
        pairs.extend([(comet_google_res, comet_trans_res), (comet_google_ranks, comet_trans_ranks)])
        hist_titles.extend(["COMET", "COMET_RANKS"])

    if len(pairs) > 0:
        plot_diff_hist(compare_pairs=pairs,
                       labels=hist_titles,
                       save_path=plots_output_path,
                       subtraction_restrictions=sub_ranks_merge_4_5)
    else:
        print(f"{PRINT_START}{RED}There are no figures to plot, we are done here :]{PRINT_STOP}")
        sys.exit(0)


if __name__ == "__main__":
    main()