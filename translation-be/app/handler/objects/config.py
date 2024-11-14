from pydantic import BaseModel, Field, computed_field
from ...shared.shared import get_arabic_entity_extract, get_hebrew_entity_extract
from ..entities.entity_extract import EntityExtract
from typing import Optional
from enums.language import TranslateDirection

class TranslatorConfig(BaseModel, arbitrary_types_allowed=True):
    translate_direction: TranslateDirection = Field(default=TranslateDirection.AR2HE)
    tokenizer_path: str = Field(default='Helsinki-NLP/opus-mt-ar-he')
    trans_model_path: str = Field(default='../MODELS/ft_opus_pre_0.0')
    token_classify_model: str = Field(default='hatmimoha/arabic-ner')
    token_classify_target_model: str = Field(default='hatmimoha/arabic-ner')
    chunk_size: int = Field(default=60, gt=0)
    alignment_method: str = Field(default='itermax')
    device: str = Field(default='cuda')

    @computed_field(return_type=Optional[EntityExtract])
    @property
    def entity_extract(self):
        if self.translate_direction.name == TranslateDirection.AR2HE.name:
            return get_arabic_entity_extract()
        elif self.translate_direction.name == TranslateDirection.HE2AR.name:
            return get_hebrew_entity_extract()