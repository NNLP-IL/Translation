from pydantic import BaseModel, model_validator, Field
from typing import Optional, List, Union
from ..enums.language import TranslateDirection, EntityExtractLanguage
from ..handler.detection.consts import SIMILAR_LANGS
from ..routers.detect import detection_handler

class TranslateContent(BaseModel):
    sentences: Union[str, List[str]]
    entity_alignment: Optional[bool] = True
    translate_direction: Optional[TranslateDirection] = TranslateDirection.AR2HE
    
    @model_validator(mode="after")
    def validate_sentences_language(self):
        lang_origin = self.translate_direction.value.split("2")[0]
        if isinstance(self.sentences, str):
            detected = detection_handler.detect_language(words=self.sentences).language
            if not detected or not (detected == lang_origin or detected in SIMILAR_LANGS.get(lang_origin, [lang_origin])):
                raise ValueError(f"Sentences not in selected source language ({lang_origin}): {self.sentences}")
        else:
            excepted = [w for w in self.sentences if detection_handler.detect_language(words=w).language not in SIMILAR_LANGS.get(lang_origin, [lang_origin])]
            if len(excepted) > 0:
                raise ValueError(f"Sentences not in selected source language ({lang_origin}): {excepted}")
        return self
    
class EntityExtractContent(BaseModel):
    language: EntityExtractLanguage = Field(description="Entity content source language, supports: HE/AR (Hebrew or Arabic).", default=EntityExtractLanguage.AR)
    text: str 
    split_sentences: Optional[bool] = True