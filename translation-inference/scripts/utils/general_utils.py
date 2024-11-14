import sys
import os
import pandas as pd
import csv
import yaml
import json
import pickle
from typing import Literal, List, Union
import re


def load_txt(txt_file: str):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    return data


def load_config(file_path: str):
    # Function to load YAML configuration
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_json(json_file: str):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def load_pkl_model(model_path: str):
    with open(model_path, 'rb') as file:
        # Load the model from the file
        model = pickle.load(file)
    return model


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "txt": load_txt,
    "yaml": load_config,
    "json": load_json,
    # "pkl": load_pkl_model
}
SupportedFormats = Union[Literal['all'], List[Literal['csv', 'xls', 'xlsx', 'txt', 'yaml', 'json']]]


def load_data(uploaded_file: str, supported_formats: SupportedFormats = 'all'):
    if os.path.isfile(uploaded_file):
        ext = uploaded_file.split(".")[-1].lower()
        if (ext in file_formats and (
                supported_formats == 'all' or (supported_formats != 'all' and ext in supported_formats))):
            return file_formats[ext](uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        raise ValueError(f"{uploaded_file} doesn't exist")


def convert_data_to_list(input_data, desired_field: str = None):
    if isinstance(input_data, list):
        return input_data
    try:
        if isinstance(input_data, int) or isinstance(input_data, float) or isinstance(input_data, bool):
            return [input_data]
        elif isinstance(input_data, pd.DataFrame):
            if len(input_data.columns) == 1:
                return list(input_data.values[:, 0])
            else:
                return input_data[desired_field].tolist()
        elif isinstance(input_data, dict):
            if len(input_data.keys()) == 1:
                return list(input_data.values())[0]
            else:
                return input_data[desired_field].tolist()
        else:
            return list(input_data)
    except:
        raise NameError("'desired_field' has not been chosen, or you chose incorrect one")


def split_into_batches(data: list, batch_size: int):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def list_to_file(input_list: list, file_path: str):
    with open(file_path, 'w') as file:
        for item in input_list:
            file.write(f"{item}\n")


def save_df_column_to_txt(df, file_name, column_name="text"):
    """
    This function saves a specified column from a pandas DataFrame to a text file,
    writing each row from the column as a separate line in the file.

    Parameters:
    - df: DataFrame containing the data.
    - file_name: Name of the text file to save the data to.
    - column_name: Name of the column to save from the DataFrame. Defaults to "text".
    """
    # Check if the column exists in the dataframe
    if column_name in df.columns:
        # Open the file in write mode
        with open(file_name, 'w', encoding='utf-8') as f:
            # Iterate through each row in the dataframe
            i = 0
            for value in df[column_name]:
                # Write the value of the selected column to the file, followed by a newline
                f.write(str(value) + '\n')
                i += 1
        print(f"Data was successfully saved to {file_name}.")
    else:
        print(f"Error: Column '{column_name}' does not exist in the dataframe.")


def filter_df_by_indexes(df: pd.DataFrame, indxs_list: list = []):
    not_found_indexes = [idx for idx in indxs_list if idx not in df.index]
    valid_indexes = [idx for idx in indxs_list if idx in df.index]
    print(f"{100 * len(not_found_indexes) / len(indxs_list):.2f}% of the indexes weren't fount")
    return df.drop(valid_indexes, axis=0)


def subtract_lists(list1: list, list2: list):
    result = []
    for item in list1:
        if item not in list2:
            result.append(item)
    return result


def append_to_csv(filename: str, row: list):
    """
    Appends a single row to a CSV file.
    """
    # Open the file in append mode
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)


def replace_multiple_patterns(text: str, patterns_replacements: dict):
    """
    Function to perform multiple replacements in a string
    """

    # Create a combined pattern from all patterns to be replaced
    combined_pattern = re.compile("|".join(re.escape(pattern) for pattern in patterns_replacements.keys()))

    # Function to return the replacement for a match
    def replace_match(match):
        return patterns_replacements[match.group(0)]

    # Substitute patterns with their corresponding replacements
    return combined_pattern.sub(replace_match, text)


def remove_lines_from_file(file_to_modify, lines_to_remove_file):
    """
    Removing specific lines from a text file by indexes.
    :param file_to_modify:
    :param lines_to_remove_file:
    :return:
    """
    # Read the line numbers to remove
    with open(lines_to_remove_file, 'r', encoding='utf-8') as file:
        lines_to_remove = {int(line.strip()) for line in file}
    # Read from your target file and keep lines that are not in lines_to_remove
    with open(file_to_modify, 'r', encoding='utf-8') as file:
        remaining_lines = [line for line_number, line in enumerate(file, start=1)
                           if line_number not in lines_to_remove]
    # Optional: Write to a new file or overwrite original
    # Here we overwrite the original, you can change the filename to create a new file instead
    with open(file_to_modify, 'w', encoding='utf-8') as file:
        file.writelines(remaining_lines)
    print(f"Lines removed. Updated file saved as: {file_to_modify}")


def split_train_test_txt(num_lines, percentage, in_path, out_train_path, out_test_path):
    """
    Splits a text file into a training set and a test set based on a specified percentage.
    :param num_lines: The number of lines from the end of the file to consider for splitting.
    :param percentage: The fraction of these lines that should go into the test set (0-1).
    :param in_path: Path to the input text file.
    :param out_train_path: Path to the output training set file.
    :param out_test_path: Path to the output test set file.
    """
    # Read the last num_lines from the file
    with open(in_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[-num_lines:]
    # Calculate the number of lines for the test set
    test_size = int(len(lines) * percentage)
    # Split the lines into training and test sets
    train_lines = lines[:-test_size]
    test_lines = lines[-test_size:]
    # Write the training lines to the training file
    with open(out_train_path, 'w', encoding='utf-8') as train_file:
        for line in train_lines:
            train_file.write(line)
    # Write the test lines to the test file
    with open(out_test_path, 'w', encoding='utf-8') as test_file:
        for line in test_lines:
            test_file.write(line)


def remove_duplicate_sentences(input_file_path, output_file_path):
    """
    Removing duplicate lines from a text file.
    :param input_file_path:
    :param output_file_path:
    :return:
    """
    # Read all lines from the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    # Remove leading and trailing whitespaces from each line
    cleaned_lines = [line.strip() for line in lines]
    # Remove duplicate lines by converting the list to a set, then back to a list
    unique_lines = list(set(cleaned_lines))
    # Sort the unique lines back to their original order
    unique_lines_sorted = sorted(unique_lines, key=cleaned_lines.index)
    # Write the unique lines to the output file
    with open(output_file_path, 'w') as file:
        for line in unique_lines_sorted:
            file.write("%s\n" % line)


def count_newlines(file_path):
    """
    Counting lines in  a text file
    :param file_path: line count (int)
    :return:
    """
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()
            # Count the number of newline characters
            newline_count = content.count('\n')
        return newline_count
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return -1  # Return -1 to indicate file not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1  # Return -1 for any other error


def remove_empty_rows(input_file, output_file):
    """
    Removing empty rows from a text file.
    :param input_file:
    :param output_file:
    :return:
    """
    with open(input_file, 'r') as file_in, open(output_file, 'w') as file_out:
        for line in file_in:
            if line.strip():  # Check if the line is not empty
                file_out.write(line)


def merge_text_files(file_paths, output_path):
    """
    Merging text files
    :param file_paths: List of paths (str)
    :param output_path:  path
    :return:
    """
    try:
        with open(output_path, 'w') as outfile:
            for file_path in file_paths:
                with open(file_path, 'r') as infile:
                    # Copy content of each file to the output file
                    outfile.write(infile.read())
                    # Optionally, write a newline after each file's content if you want separation
                    # outfile.write('\n')  # Remove or comment out this line if you don't want extra newlines
    except IOError as e:
        print(f"An error occurred while processing the files: {e}")


class Write2Streams:
    """
    To write in parallel to 2 streams (for example: the stdout in terminal, and into a log file)
    """

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()


class SuppressPrint:
    """
    Suppress print statements.
    A custom class to achieve suppression of print statements from functions not within the current file,
    that redirects sys.stdout and sys.stderr to os.devnull globally and ensure this suppression.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr