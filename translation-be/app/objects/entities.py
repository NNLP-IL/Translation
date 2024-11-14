from pydantic import BaseModel
from app.enums.tagger_src import EntityTagSource
from typing import List, Union, Optional
import re

class Entity(BaseModel):
    word: str
    tag: str
    tag_hex: Optional[str] = None
    offset: Union[tuple[int, int], List[tuple[int, int]]] = None 
    source: EntityTagSource = None
    
class Sentence(BaseModel):
    sentence: str
    entities: List[Entity] = []
    
    def update_entities_locations(cls):
        for i, entity in enumerate(cls.entities):
            if not entity.offset:
                cls.entities[i].offset = [match.span() for match in re.finditer(rf"{re.escape(entity.word)}(?=\s|$)", cls.sentence)]