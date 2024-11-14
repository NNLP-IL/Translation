import argparse
import pandas as pd
import re
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import save_df_column_to_txt, merge_text_files

ARABIC_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0660-\u0669\u061B\u061F\u0621-\u063A\u0640-\u064A\u066A-\u066C0-9:,/\-\.\?!–\u200e\u200f]+|\.{2,}'
)


def create_bins(mixed_df: pd.DataFrame):
    """
    This function takes a DataFrame with an 'arabic_ratio' column, bins the values in this column into
    predefined ranges, adds these bin labels as a new column to the DataFrame, and calculates the ratio of
    values in each bin.

    Args:
    mixed_df (pd.DataFrame): A pandas DataFrame that contains a column named 'arabic this function creates bins for values in this column.

    Returns:
    tuple: The first element is the input DataFrame with an additional column 'bin' indicating the bin label
           for each row. The second element is a pandas Series containing the ratio of counts in each bin relative
           to the total number of rows in the DataFrame.
    """
    # Define bin edges
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    # Define bin labels
    labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1']
    # Add a new column 'bin' to the DataFrame with the bin labels
    mixed_df['bin'] = pd.cut(mixed_df['arabic_ratio'], bins=bins, labels=labels, right=False)
    # Count the values in each bin
    bin_counts = mixed_df['bin'].value_counts()
    # Calculate the ratio of values in each bin
    total_count = len(mixed_df)
    bin_ratios = bin_counts / total_count
    return mixed_df, bin_ratios


def remove_non_arab_pattern_by_regex(df: pd.DataFrame):
    """
    This function takes a DataFrame with an 'arabic_ratio' column, bins the values in this column into
    predefined ranges, adds these bin labels as a new column to the DataFrame, and calculates the ratio of
    values in each bin.
    Args:
    mixed_df (pd.DataFrame): A pandas DataFrame that contains a column named 'arabic this function creates bins for values in this column.
    Returns:
    tuple: The first element is the input DataFrame with an additional column 'bin' indicating the bin label
           for each row. The second element is a pandas Series containing the ratio of counts in each bin relative
           to the total number of rows in the DataFrame.
    """
    # Define a regular expression pattern to match hashtags
    hashtag_pattern = re.compile(r'#\w+')
    # Define a regular expression pattern to match locations
    location_pattern = re.compile(r'@\w+')
    # Define a regular expression pattern to match URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # Define a regular expression pattern to match Arabic characters
    non_arabic_indexes = ~df['text'].apply(lambda x: bool(ARABIC_PATTERN.search(x)))
    not_arabic_df = df[non_arabic_indexes]  # Create a new DataFrame to hold the rows that didn't meet the conditions
    df = df[~non_arabic_indexes]  # Keep only rows with Arabic text
    # Remove \n (new line characters)
    df['text'] = df['text'].str.replace('\n', '', regex=False)
    # Remove hashtags, locations, and URLs from the 'text' column
    df['text'] = df['text'].apply(lambda x: hashtag_pattern.sub(r'', x))  # Remove hashtags
    df['text'] = df['text'].apply(lambda x: location_pattern.sub(r'', x))  # Remove locations
    df['text'] = df['text'].apply(lambda x: url_pattern.sub(r'', x))  # Remove URLs

    return df, not_arabic_df


def find_arab_text_mixed_with_non_arab_text(df: pd.DataFrame):
    """
    Identifies texts within a DataFrame containing a mix of Arabic and non-Arabic words.
    It calculates the ratio of Arabic words to the total word count for each mixed text entry,
    returning a DataFrame with these mixed texts, their Arabic word ratios, and original indices.

    Parameters:
    - df (pd.DataFrame): DataFrame with a 'text' column to analyze.

    Returns:
    - pd.DataFrame: A new DataFrame containing texts with Arabic and non-Arabic mix,
      their Arabic-to-total word ratios, and their original DataFrame indices.
    """
    mixed_texts = []
    ratios = []
    indexes = []
    # Iterate over the texts in the DataFrame
    for index, text in df['text'].items():
        # Split the text into words
        words = text.split()
        # Search for Arabic words in the text
        arabic_words = ARABIC_PATTERN.findall(text)
        # Calculate the ratio of Arabic to non-Arabic words
        if len(words) > 0:
            arabic_ratio = len(arabic_words) / len(words)
        else:
            arabic_ratio = 0
        # Check if there are Arabic words and non-Arabic words in the text
        if arabic_words and len(arabic_words) < len(words):
            mixed_texts.append(text)  # Add the text to the list
            ratios.append(arabic_ratio)  # Add the ratio to the list
            indexes.append(index)  # Add the original index to the list
    # Create a new DataFrame containing the texts, along with their ratios and original indexes
    mixed_df = pd.DataFrame({'text': mixed_texts, 'arabic_ratio': ratios}, index=indexes)

    return mixed_df


def create_characters_dict(df: pd.DataFrame):
    # Initialize an empty dictionary to store character counts
    char_count = {}

    def count_char_occurrences(text):
        """
        Updates the 'char_count' dictionary with counts of each character in 'text'.
        - Uses the 'nonlocal' keyword to access and modify 'char_count' defined in the outer scope.
        - Iterates through each character in 'text': if the character exists in 'char_count', increment its value; if not, set it to 1.
        """
        nonlocal char_count  # Access the outer scope variable char_count
        # Iterate through each character in the text
        for char in text:
            # Update the count for each character
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1

    # Apply the count_char_occurrences function to each text entry in the DataFrame
    df['text'].apply(count_char_occurrences)
    return char_count


def remove_mixed_text_rows_from_df(df: pd.DataFrame, mixed_df: pd.DataFrame):
    """
    Removes rows from 'df' that are identified as mixed text in 'mixed_df'.
    Parameters:
    - df: The original pandas DataFrame from which rows are to be removed.
    - mixed_df: A pandas DataFrame containing the rows identified as having mixed text,
      which are to be removed from 'df'.
    Returns:
    - cleaned_df: The 'df' DataFrame after removing the rows that are present in 'mixed_df'.
    """
    # Get the indices of rows to remove from the original DataFrame
    indices_to_remove = mixed_df.index
    # Remove the rows from the original DataFrame
    cleaned_df = df.drop(indices_to_remove)
    return cleaned_df


def count_words_in_each_text(df):
    """
    A function that counts words in Arabic text.
    """

    def count_words_arabic(text):
        if isinstance(text, str):  # Check if the text is a string
            return len(text.split())
        else:
            return 0  # Return 0 for missing or non-string values

    word_count = df['text'].apply(lambda text: count_words_arabic(text))
    word_count = word_count.sort_values()
    return word_count


def text_file_to_dataframe(file_path: str):
    """Convert a text file into a pandas DataFrame.
    Each row in the DataFrame represents one line from the text file.
    Args:
    file_path (str): The path to the text file.
    Returns:
    pandas.DataFrame: A DataFrame with a single column 'text' containing the lines of the file.
    """
    # Initialize an empty list to hold lines from the file
    lines = []
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Append each line to the list, removing any leading/trailing whitespace
            lines.append(line.strip())
    # Convert the list of lines to a DataFrame
    df = pd.DataFrame(lines, columns=['text'])
    return df


def analyze_arab_data(chunk_len, df_text, not_arabic_df, mixed_text, word_count, bin_ratios, char_count, total_char_sum,
                      analyze_output_file):
    """
    Saves the analyzed data into a csv file.
    """
    result_df = pd.DataFrame({
        'Chunk len': [chunk_len],
        'Arab text len': [len(df_text)],
        'Non arab text len': [len(not_arabic_df)],
        'Mixed arab text len': [len(mixed_text)],
        'Min word count': [word_count.min()],
        'Max word count': [word_count.max()],
        'Mean word count': [word_count.mean()],
        'Arab ratio from mixed 0.75-1': [bin_ratios[0]],
        'Arab ratio from mixed 0.5-0.75': [bin_ratios[1]],
        'Arab ratio from mixed 0.25-0.5': [bin_ratios[2]],
        'Arab ratio from mixed 0-0.25': [bin_ratios[3]],
        'Char count': [len(char_count)],
        'Total char sum': [total_char_sum]
    })
    # Check if the file exists
    if os.path.isfile(analyze_output_file):
        # Append to the existing file without writing headers
        result_df.to_csv(analyze_output_file, mode='a', index=False, header=False)
    else:
        # Create a new file and write the DataFrame to it
        result_df.to_csv(analyze_output_file, index=False)


def clean_and_analyze_data(df: pd.DataFrame, output_clean_path: str, mixed_text_file_path: str,
                           analyze_output_file: str = None):
    """
    Cleans and analyzes Arabic text data from a DataFrame by removing empty and non-Arabic texts, identifying mixed Arabic texts,
    and calculating word count statistics. The cleaned text and identified mixed texts are saved to specified file paths.
     Optionally, generates an analysis report summarizing the cleaning and analysis process.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a 'text' column.
    - output_clean_path (str): Path to save the cleaned Arabic texts.
    - mixed_text_file_path (str): Path to save mixed Arabic and non-Arabic texts.
    - analyze_output_file (str, optional): Path to save the analysis report.

    Returns:
    - A tuple with the cleaned DataFrame, its index, and the DataFrame of mixed texts.
    """
    df = df[df['text'] != '']
    chunk_len = len(df)
    word_count = count_words_in_each_text(df)
    print("Min word count in text (before cleaning) :", word_count.min())
    print("Max word count in text (before cleaning) :", word_count.max())
    print("Mean of word count (before cleaning) :", word_count.mean())
    print("Filtering to arab and non arab text...")
    clean_df, not_arabic_df = remove_non_arab_pattern_by_regex(df)
    mixed_text = find_arab_text_mixed_with_non_arab_text(clean_df)
    mixed_text, bin_ratios = create_bins(mixed_text)
    print("Finding mixed Arabic texts...")
    clean_df = remove_mixed_text_rows_from_df(clean_df, mixed_text)
    word_count_clean = count_words_in_each_text(clean_df)
    print("Min word count in text (after cleaning) :", word_count_clean.min())
    print("Max word count in text (after cleaning) :", word_count_clean.max())
    print("Mean of word count (after cleaning) :", word_count_clean.mean())
    print("Creating characters dictionary...")
    char_count = create_characters_dict(clean_df)
    total_char_sum = sum(char_count.values())
    save_df_column_to_txt(clean_df, output_clean_path)
    if mixed_text_file_path:
        save_df_column_to_txt(mixed_text, mixed_text_file_path)
    if analyze_output_file:
        analyze_arab_data(chunk_len, clean_df, not_arabic_df, mixed_text, word_count, bin_ratios, char_count,
                          total_char_sum, analyze_output_file)
    return clean_df, clean_df.index, mixed_text


def clean_and_analyze_data_by_chunks(source_txt_file_path, clean_file_path, mixed_text_file_path, analyzed_file_path,
                                     num_chunks,
                                     chunk_size=1000):
    """
    Handle large text files by reading, cleaning, and analyzing the data in chunks, making the process memory-efficient.
    """

    clean_file_base, clean_file_ext = '.'.join(clean_file_path.split('.')[:-1]), clean_file_path.split('.')[-1]
    if mixed_text_file_path:
        mixed_text_file_base, mixed_text_file_ext = '.'.join(mixed_text_file_path.split('.')[:-1]), \
        mixed_text_file_path.split('.')[-1]

    # Read the file in chunks
    i = 0
    for chunk in pd.read_csv(source_txt_file_path, header=None, names=['text'], sep='\t', chunksize=chunk_size):
        i += 1

        # Apply process to the chunk
        clean_and_analyze_data(chunk, clean_file_base + '_' + str(i) + '.' + clean_file_ext,
                               f'{mixed_text_file_base}_{i}.{mixed_text_file_ext}' if mixed_text_file_path else None,
                               analyzed_file_path)

        # Break the loop after processing num_chunks chunks
        if i >= num_chunks:
            break

    # Merge all the outputs from different chunks together
    merge_text_files([f'{clean_file_base}_{i}.{clean_file_ext}' for i in range(1, i + 1)], clean_file_path)
    if mixed_text_file_path:
        merge_text_files([f'{mixed_text_file_base}_{i}.{mixed_text_file_ext}' for i in range(1, i + 1)],
                         mixed_text_file_path)


def main():
    parser = argparse.ArgumentParser(description="Cleans and analyzes source text files for Arabic text content.")
    parser.add_argument('--src_file_path', type=str, required=True,
                        help='Path to the source text file that needs to be cleaned and analyzed.')
    parser.add_argument('--clean_file_path', type=str, required=True,
                        help='Output filename where cleaned text will be saved.')
    parser.add_argument('--analyzed_file_path', type=str, required=False,
                        help='Optional: Output filename where analysis of the input text will be saved (ends with .csv).')
    parser.add_argument('--mixed_text_file_path', type=str, required=False,
                        help='Optional: Output filename where the mixed data from the text (sentences that include arab and non arab words) will be saved.')
    parser.add_argument('--num_chunks', type=int, default=1000,
                        help='Number of chunks to process from the source text file. Default is 1000.')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of chunks to divide the source text file into for processing. Default is 1000 sentences per chunk.')
    parser.add_argument('--clean_by_chunks', action='store_true',
                        help='Option to clean and analyze the source text file by dividing it into manageable chunks.')

    args = parser.parse_args()

    if args.clean_by_chunks:
        clean_and_analyze_data_by_chunks(args.src_file_path, args.clean_file_path, args.mixed_text_file_path,
                                         args.analyzed_file_path,
                                         args.num_chunks, args.chunk_size)
    else:
        df = text_file_to_dataframe(args.src_file_path)
        clean_and_analyze_data(df, args.clean_file_path, args.mixed_text_file_path, args.analyzed_file_path)


if __name__ == '__main__':
    main()
