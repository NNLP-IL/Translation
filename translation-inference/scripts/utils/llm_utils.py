import requests
import os
import json
import pandas as pd
import csv
from tqdm import tqdm
from typing import List, Dict, TextIO
from dotenv import load_dotenv
load_dotenv()


def format_prompt_with_data(prompt: str, row: pd.Series):
    """
    Formats a prompt string with data from a DataFrame row.

    Parameters:
        prompt (str): The template prompt containing placeholders.
        row (Series): A pandas Series representing a row of data.

    Returns:
        str: The prompt filled with actual data from the row.
    """
    cleaned_dict = {k: v.strip() if isinstance(v, str) else v for k, v in row.to_dict().items()}
    return prompt.format(**cleaned_dict)


def send_prompt_to_llm(model: str, prompt: str, temp: float = 0, max_tokens: int = 1000):
    """
    Sends a formatted prompt to an LLM for processing using the Chat Completion API.

    Parameters:
        model (str): The model ID to use for processing.
        prompt (str): The fully formatted prompt to send.
        temp: 0
        max_tokens: 1000

    Returns:
        str: The LLM's response.
    """

    # adjust the prompt to be in openai format
    prompt = [{"role": "user", "content": prompt}]
    response = requests.post(
        url=f"{os.getenv('OpenRouter_API_Base_URL')}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OpenRouter_API_KEY')}",
        },
        data=json.dumps({
            "model": model,
            "messages": prompt,
            "temperature": temp,
            "max_tokens": max_tokens,
        })
    )

    try:
        response = response.json()
        reply = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("An error occurred:", e)
        # temporary solution
        reply = '0'
        return reply
    return reply


def send_df_to_llm(models: str | List[str], base_prompt: str, data: pd.DataFrame, output_file: str, validation_function=None):
    """
    Processes each row of the data DataFrame through the LLM and collects responses.

    Parameters:
        models (str or List[str]): List of model IDs to use for processing, can be also a single string.
        base_prompt (str): The base prompt for the LLM.
        data (DataFrame): DataFrame containing multiple rows of data.
        output_file (str): Path to save the output CSV files.
        validation_function (callable, optional): A custom function to validate the llm output.

    Returns:
        files (dict): The paths were each model results are saved.
    """

    if isinstance(models, str):
        models = [models]
    else:
        # We assume 'models' is a list - if the list is empty, we raise a flag
        if len(models) == 0:
            raise ValueError("'models' list is empty,"
                             " make sure the 'models' variable in Configs/llms2use.py is set correctly (str or list)")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory for 'output_file' does not exist: {output_dir}")


    # Create .csv writers
    writers: Dict[str, csv.writer] = {}
    files: Dict[str, TextIO] = {}
    output_file = f"{'.'.join(output_file.split('.')[:-1])}@{{model_name}}.csv"

    try:
        # Create writers objects to write for files
        for model in models:
            model_name = model.replace("/", "-")
            csv_path = output_file.format(model_name=model_name)
            file_exist = os.path.exists(csv_path)
            existing_indices = set()
            if file_exist:
                # Read existing indices
                try:
                    existing_data = pd.read_csv(csv_path)
                    existing_indices = set(existing_data['index'].values)
                except (IOError, pd.errors.EmptyDataError) as e:
                    print(f"Warning: Error reading existing file {csv_path}: {e}")
            try:
                file = open(csv_path, 'a+', newline='')
                files[model_name] = file
                writers[model_name] = csv.writer(file)
                if not file_exist:
                    writers[model_name].writerow(['index', 'response'])
            except IOError as e:
                raise IOError(f"Error opening file {csv_path} for writing: {e}")

        for i, row in tqdm(data.iterrows(), total=len(data)):
            if i not in existing_indices:
                try:
                    # Insert data to prompt
                    formatted_prompt = format_prompt_with_data(base_prompt, row)
                    # Send the same prompt to all the defined models
                    models_responses = {}
                    for model in models:
                        model_name = model.replace("/", "-")
                        # Send prompt to LLM
                        response = send_prompt_to_llm(model, formatted_prompt)
                        # Validate the response
                        if validation_function:
                            response = validation_function(response)
                        models_responses[model_name] = response
                    # Add to file (Note: here Just to make sure that writing to all files happens at the same time)
                    for model_name, response in models_responses.items():
                        writers[model_name].writerow([i, response])
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    for model in models:
                        model_name = model.replace("/", "-")
                        writers[model_name].writerow([i, "0"])
            else:
                print(f"index {i} already exist (has been processed), we can skip him")

    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

    finally:
        # Ensure all files are closed properly
        for file in files.values():
            file.close()

    return {model_name: file.name for model_name, file in files.items()}
