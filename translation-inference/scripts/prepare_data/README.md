# Arabic Text Cleaning, Analyze, and Translate Toolkit

This toolkit is crafted to streamline the process of cleaning, analyzing, and translating Arabic text data. It includes several utilities: removing non-Arabic text, identifying mixed Arabic and non-Arabic text elements, calculating word counts, and conducting basic textual analysis, and translating the source text (in order to create target set) using Google Translate. This document outlines how to set up and use the toolkit.

## Features

- Text Cleaning: Removes non-Arabic text from your dataset, retaining only valid Arabic textual content.
- Mixed Text Identification: Identifies and isolates texts containing a mix of Arabic and non-Arabic words.
- Arabic Word Count: Calculates the total and average word counts for the Arabic texts.
- Text Analysis: Performs a basic analysis of the Arabic content, including character count and the distribution of Arabic content across predefined bins.
- Data Chunk Processing: Efficiently processes large text files by dividing them into manageable chunks.
- Text Translation: Equipped with the capability to translate text from Arabic to Hebrew.

## Usage for Clean and Analyze the Source Text
The toolkit can be operated via the command line. Parameters allow you to specify input files, output directories, and behavior.

### Command Line Arguments

`--src_file_path`: Path to the source text file that needs to be cleaned and analyzed. (Required)

`--clean_file_path`: Directory and filename prefix where cleaned text will be saved. (Required)

`--analyzed_file_path`: Optional. Directory and filename prefix where analyzed data from the text will be saved (ends with .csv).

`--mixed_text_file_path`: Optional. Directory and filename prefix where the mixed data from the text (sentences that include Arabic and non-Arabic words) will be saved.

`--num_chunks`: Number of chunks to process from the source text file. Default is 5.

`--chunk_size`: Size of chunks to divide the source text file into for processing. Default is 1000 records per chunk.

`--clean_by_chunks`: Option to clean and analyze the source text file by dividing it into manageable chunks (Bool).

### Basic Example

To start processing a text file named `source.txt`, cleaning the data, and saving the cleaned data to a file with a prefix `cleaned_text`, run:

`python clean_training_data.py --src_file_path source.txt --clean_file_path cleaned_text.txt`

### Advanced Use

For processing large files in chunks and saving both cleaned text and analytical data:

`python clean_training_data.py --src_file_path large_source.txt --clean_file_path cleaned.txt --analyzed_file_path analysis.csv --mixed_text_file_path mixed_texts.txt --num_chunks 10 --chunk_size 2000 --clean_by_chunks`

This command processes `large_source.txt`, divides it into chunks of 2000 records, cleans the data, identifies mixed Arabic text, and saves the cleaned and analyzed data with the specified prefixes. It processes a maximum of 10 chunks.

## Usage for Translating

Before you can use File Translator, make sure you have the following:

- Python 3 installed on your machine.
- A Google Cloud account with the Translation API enabled.
- A Google Cloud project with billing enabled.
- Generated Google Cloud credentials in the form of a JSON file, see [google_env_example.json](google_env_example.json).


`python google_translate.py --input_path "./example_ar.txt" --output_path "./translated_he.txt"`

## Google Translate Cleaning 🧹
Google translations have sometimes particular unique characters that we want to clean.
To do that you can us `clean_google_translate.py` script.

`python clean_google_translate.py --input_path ./google_translate_results.txt --output_path ./google_translate-cleaned.txt`


