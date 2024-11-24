import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from app.utils.file_utils import FileLoader
from app.utils.general_utils import convert_data_to_list, list_to_file
from app.handler.predict import Translator, TranslatorConfig
from typing import Optional, List


def test_translator():
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    translator = Translator(TranslatorConfig(device=device))
    src_sentences: List[str] = ["عندما تُفتتح الرواية الأولى من السلسلة، هاري بوتر وحجر الفيلسوف، من الواضح أن أمرًا مهمًّا قد حَدَثَ في عالم السحرة"]
    translations = translator.translate(src_sentences=src_sentences)
    assert translations == ["כשפתחו את הספר הראשון בסדרה, הארי פוטר ואבן הפילוסוף, ברור שדבר חשוב קרה בעולם הקוסמים"]
    
def translator_by_file(src_file: str, output_dir: str, model_path: Optional[str] = None, alignment_method: Optional[str] = 'alignment_method'):
    # Set device to CUDA if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    src_sentences = convert_data_to_list(FileLoader.load_data(src_file), desired_field='source')

    # Create a translator
    translator = Translator(TranslatorConfig(device=device, trans_model_path=model_path, alignment_method=alignment_method))
    translations = translator.translate(src_sentences=src_sentences)

    # Save
    list_to_file(input_list=translations, file_path=os.path.join(output_dir, 'translations.txt'))

def main():
    parser = argparse.ArgumentParser(description="Send sentences to translation")
    parser.add_argument('src_file', type=str, help="File containing the source data.")
    parser.add_argument('output_dir', type=str, help="The output dir to save the translations")
    parser.add_argument('--model', type=str, default='HebArabNlpProject/mt-ar-he', help="The model we use for translations")
    parser.add_argument('--alignment_method', type=str, default='itermax', choices=["inter", "itermax", "mwmf"],
                        help="The alignment method we use, based on this paper: https://arxiv.org/pdf/2004.08728")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    translator_by_file(src_file=args.src_file,output_dir=args.output_dir,model_path=args.model_path, alignment_method=args.alignment_method)

if __name__ == '__main__':
    test_translator()