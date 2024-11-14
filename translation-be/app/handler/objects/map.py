from pydantic import BaseModel
from typing import Optional
from objects.language import Language
from enums.language import TranslateDirection

class TranslitMap(BaseModel):
    src: Optional[Language] = "ar"
    tgt: Optional[Language] = "he"
    
    def __init__(self, src: Optional[Language] = "ar", tgt: Optional[Language] = "he", translate_direction: Optional[TranslateDirection] = TranslateDirection.AR2HE):
        super().__init__(src=src, tgt=tgt)
        if translate_direction.name == TranslateDirection.AR2HE.name:
            self.src = "ar"
            self.tgt = "he"
        elif translate_direction.name == TranslateDirection.HE2AR.name:
            self.src = "he"
            self.tgt = "ar"