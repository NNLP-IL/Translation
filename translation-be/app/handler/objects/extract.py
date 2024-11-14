from ..entities.entity_extract import EntityExtract
from ...consts import ArabicConfig, HebrewConfig
from typing import Optional
from pydantic import BaseModel

class EntityExtractModels(BaseModel, arbitrary_types_allowed=True):
    arabic_extract: Optional[EntityExtract] = None
    hebrew_extract: Optional[EntityExtract] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.arabic_extract is None:
            self.arabic_extract = EntityExtract(token_classify_model=ArabicConfig.TOEKN_CLASSIFY_MODEL)
        if self.hebrew_extract is None:
            self.hebrew_extract = EntityExtract(token_classify_model=HebrewConfig.TOEKN_CLASSIFY_MODEL)
            