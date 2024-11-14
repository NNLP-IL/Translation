from typing import List, Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ...utils.sentence_utils import split_ar_sentences
from ...objects.entities import Sentence, Entity, EntityTagSource


class NER:
    """ 
    Named Entity Recognition (NER) model is a type of Natural Language Processing (NLP) algorithm 
    designed to identify and classify key elements (named entities) in text into predefined categories. 
    """
    def __init__(self, token_classify_model: Optional[str] = "hatmimoha/arabic-ner"):       
        self.tokenizer = AutoTokenizer.from_pretrained(token_classify_model)
        self.model = AutoModelForTokenClassification.from_pretrained(token_classify_model)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_nlp_annotation(self, sentences: List[str]):
        """ Annotate the given text using the NLP model. """
        return self.nlp(sentences)

    def get_entities(self, text: str, split_text: bool = True, tag_map: dict = {}):
        """
        Extract named entities from the given text.
        Returns: List[Entity]: A list of extracted entities with their corresponding tags.
        """
        sentneces_entities: List[Sentence] = []
        sentences = split_ar_sentences(text=text) if split_text else [text]
        for i, sentence in enumerate(self.get_nlp_annotation(sentences=sentences)):
            sentence_entities: Sentence = Sentence(sentence=sentences[i])
            for item in sentence:
                if item["word"].startswith("##") and len(sentence_entities.entities) > 0:
                    sentence_entities.entities[-1].word = sentence_entities.entities[-1].word + item["word"].replace("##", "")
                    sentence_entities.entities[-1].offset = (sentence_entities.entities[-1].offset[0], item["end"])
                else:
                    if tag_hex:= tag_map.get(item["entity"].rsplit("-")[-1], {}).get("hex"):
                        sentence_entities.entities.append(Entity(word=item["word"], tag=item["entity"], tag_hex=tag_hex, 
                                                                offset=(item["start"], item["end"]), source=EntityTagSource.NER.value))
            sentneces_entities.append(sentence_entities)
        return sentneces_entities
 