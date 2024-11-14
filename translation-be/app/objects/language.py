from pydantic import BaseModel
from typing import Optional

Language = Optional[str]

class LanguageForm(BaseModel):
    language: str
    display_text: str
    country: Optional[str] = None