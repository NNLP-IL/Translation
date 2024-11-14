import argparse
from typing import List

from scripts.utils.general_utils import replace_multiple_patterns, load_data


def clean_translated_texts(translated_texts: List[str]):
    """Processes each translated text to replace specific patterns,
    and maintaining original line count."""
    cleaned_texts = []
    patterns_replacements = {
        '&quot;': '"',
        '&#39;': "'"
    }
    for text in translated_texts:
        clean_text = replace_multiple_patterns(text, patterns_replacements)
        cleaned_texts.append(clean_text)
    return cleaned_texts


def clean_file_line_by_line(input_filename: str, output_filename: str):
    """Reads a file line by line, processes each line to remove specific patterns and words,
    and maintains the original line count by preserving empty lines."""
    try:
        lines = load_data(input_filename)
        cleaned_lines = clean_translated_texts(lines)
        # Write the cleaned lines to the output file, preserving the original number of lines
        with open(output_filename, 'w', encoding="utf-8") as file:
            for cleaned_line in cleaned_lines:
                file.write(f"{cleaned_line}\n")
        print(f"Processing done. Output saved to {output_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description='Clean Google Translated files.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()
    clean_file_line_by_line(input_filename=args.input_path, output_filename=args.output_path)


if __name__ == '__main__':
    main()