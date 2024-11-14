from pydantic import BaseModel
from ..objects.language import LanguageForm

class LanguageDetection(BaseModel):
    language: LanguageForm
    words: str