from pydantic import BaseModel
from enums.language import TranslateDirection
from typing import List, Union, Optional

class TranslatorInfo(BaseModel):
    device: Optional[str] = None
    translate_direction: Optional[TranslateDirection] = TranslateDirection.AR2HE
    translation_model: Optional[str] = None
    batch_size: Optional[int] = None
    alignment_method: Optional[str] = None
    
class TranslatorResults(BaseModel):
    translations: Optional[Union[List[str], str]] = ''
    info: TranslatorInfo