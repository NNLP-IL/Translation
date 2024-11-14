from enum import Enum

class EntityTagSource(Enum):
    NER: str = "NER"
    REGEX: str = "Regex"
    