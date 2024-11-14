import pandas as pd
import numpy as np
from scripts.utils.general_utils import load_data
from typing import List

import evaluate
comet_metric = evaluate.load('comet', config_name='Unbabel/wmt22-comet-da')


def get_comet_scores(src_sentences: List[str], ref_sentences: List[str], trans_sentences: List[str]):
    comet_result = comet_metric.compute(predictions=trans_sentences,
                                        references=[label for label in ref_sentences], sources=src_sentences)
    print(f"mean COMET score: {comet_result['mean_score']}")
    return comet_result['scores']


def get_comet_qe_scores(comet_qe_model, src_sentences: List[str], trans_sentences: List[str]):

    # Order data in the right format
    data = [{"src": src, "mt": mt} for src, mt in zip(src_sentences, trans_sentences)]

    # Call predict method:
    model_output = comet_qe_model.predict(data, batch_size=8, gpus=0)
    return model_output.scores


def get_comet_rank(comet2rank_model, comet_scores: List | np.array):
    if isinstance(comet_scores, list):
        comet_scores = np.array(comet_scores).reshape(-1, 1)
    return comet2rank_model.predict(comet_scores)


def base_aixsts_prompt(run_google: bool):
    prompt = """
    Given the task of analyzing linguistic similarity between two sentences in different languages, perform the following steps:

    Compare the translated version of Text 1 with Text 2.
    Evaluate their similarity based on criteria including meaning equivalence, expression formality, style, emphasis, and usage of idioms or metaphors. 

    Score their similarity on a scale of 1 to 5, where:
    5 = Exactly and completely equivalent in meaning and expression.
    4 = Near-equivalent meanings with only minor differences in expression.
    3 = Mostly equivalent, with some unimportant details differing.
    2 = Some shared details but not equivalent in important aspects.
    1 = Not equivalent, sharing very little details or about different topics.

    Examples:

    Example 1:
    Text 1: لا لا لا ما بدنا انتعب الحجة
    Text 2: לא, לא, לא, אנחנו לא רוצים למצות את הטיעון
    Answer: 1


    Example 2:
    Text 1: يعني اللّي بخشّ فيها الأزمة بطلع بكرة العصر.
    Text 2: כלומר, מי שחושש מהמשבר ייצא מוקדם אחר הצהריים.
    Answer: 2

    Example 3:
    Text 1: يزن حبيبي مش معقول شو كبران 
    Text 2: יאזן, יקירתי, לא ייאמן כמה הוא גדל
    Answer: 3

    Example 4:
    Text 1: ولازم كمان تزور المطاعم القديمة زي هاشم.
    Text 2: צריך לבקר גם במסעדות ותיקות כמו האשם.
    Answer: 4

    Example 5:
    Text 1: وشو بالنسبة للطيارة والمطار يا أم إنصاف؟
    Text 2: מה עם המטוס ושדה התעופה, אום אנסף?
    Answer: 5


    Based on this analysis, provide only the similarity score (1 - 5) as the output, without any explantaion.

    Your Task:

    Text 1: {source}
    Text 2: {generated}
    Answer: Your score here
    """
    if run_google:
        prompt = prompt.replace("{generated}", "{google translate}")
    return prompt


def validate_aixstsx_response(answer: List | str):
    if len(answer) > 1:
        answer = answer[0]
    if answer.isdigit():
        return answer
    else:
        return "0"


def ai_xsts(comet_ranks: List, llms_outputs: dict, output_file: str):
    """
    Calculate the final AI-XSTS rankings based on COMET and LLMs rankings.
    Save those results to CSV file.

    Parameters:
        comet_ranks (list): List of COMET scores converted to rankings in range 1-5.
        llms_outputs (dict): Each key is the name of the LLM we used, and the corresponding value is the path were his
        LLM rankings are stored.
        output_file (str): Path to save the output CSV file.

    Returns:
        None
    """

    # Creating an empty Dataframe to put in the results and save later as CSV file
    df_tmp = pd.DataFrame()

    # Adding the COMET ranks and the LLMs rankings
    df_tmp["comet_ranks"] = comet_ranks
    for model_name, results_path in llms_outputs.items():
        llm_res = load_data(results_path)
        if isinstance(llm_res, pd.DataFrame):
            df_tmp[model_name] = llm_res.get('response', llm_res.columns[0])
        else:
            df_tmp[model_name] = llm_res

    # Calculating the mean value between all the rankings - this will be our final ranking
    numeric_df = df_tmp.applymap(float)
    row_means = numeric_df.mean(axis=1)
    rounded_means = row_means.round()
    ai_xsts_scores = rounded_means.astype(int)

    # Save as csv file
    ai_xsts_scores.to_csv(output_file, header=False, index=False)
