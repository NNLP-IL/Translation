import argparse
import os
import time
from google.cloud import translate_v2
from tqdm import tqdm

AR_CODE = "ar"
HE_CODE = "he"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"google_env_example.json"
print(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))


def translate_text(target: str, text: str) -> dict:
    client = translate_v2.Client()

    response = client.translate(text, target_language=target)

    return response["translatedText"]


def translate_text_in_file_(f_path, output_path, src_lang, tgt_lang):
    input_char_count = 0
    client = translate_v2.Client()
    iter_count = 0

    with open(f_path, "r", encoding='utf-8') as input_f:
        lines = input_f.readlines()

    with tqdm(total=len(lines), desc="Translating") as pbar, open(output_path, "w", encoding='utf-8') as output_f:
        sentences_batch = []
        for sentence in lines:
            # Count the characters in the input sentence
            input_char_count += len(sentence.strip())
            sentences_batch.append(sentence.strip())

            # Batch processing
            if iter_count % 50 == 0 or sentence.strip() == "":
                if sentences_batch:  # Ensure the batch is not empty
                    translations = client.translate(sentences_batch, source_language=src_lang, target_language=tgt_lang)
                    for translation in translations:
                        output_f.write(
                            translation['translatedText'] + "\n")  # Make sure to add a newline for each translated text
                    sentences_batch = []  # Clear the batch for the next set of sentences

            iter_count += 1
            pbar.update(1)  # Update progress bar after each line is processed

    # Ensure any remaining sentences in the batch are processed
    if sentences_batch:
        translations = client.translate(sentences_batch, source_language=src_lang, target_language=tgt_lang)
        with open(output_path, "a",
                  encoding='utf-8') as output_f:  # Open the file again to append remaining translations
            for translation in translations:
                output_f.write(translation['translatedText'] + "\n")

    return input_char_count


def main():
    parser = argparse.ArgumentParser(description='Translate text files.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    src = AR_CODE
    tgt = HE_CODE
    start_time = time.time()
    input_char_count = translate_text_in_file_(args.input_path, args.output_path, src, tgt)
    end_time = time.time()
    print("char count : ", input_char_count)

    execution_time = end_time - start_time
    print("Execution Time: {:.2f} seconds".format(execution_time))


if __name__ == '__main__':
    main()
