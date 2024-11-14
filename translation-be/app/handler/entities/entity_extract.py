import os
from .ner import NER
from typing import List
from ...utils.regex_extract import RegexUtil
from ...utils.file_utils import FileLoader
from ...objects.entities import Sentence

class EntityExtract:
    TAGGING_MAP: str = os.getenv("TAGGING_MAP", "app/handler/entities/config/tag_map.yaml")
    def __init__(self, token_classify_model: str = None):
        self.tag_map: dict = self._load_tagging_map()
        self.ner = NER(token_classify_model=token_classify_model)
    
    def _load_tagging_map(self):
        try:
            return FileLoader.load_config(file_path=self.TAGGING_MAP)["tags"]
        except:
            return {}
        
    def get_entities(self, text: str, split_text: bool = True):
        """ Extracts named entities from the given text using NER model, Regex. """
        sentences_entities: List[Sentence] = self.ner.get_entities(text=text, split_text=split_text, tag_map=self.tag_map)
        any(sentence_entities.entities.extend(RegexUtil.extract_entities(text=sentence_entities.sentence, tag_map=self.tag_map)) 
            for sentence_entities in sentences_entities if sentence_entities)
        return sentences_entities