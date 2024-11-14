from dataclasses import dataclass
import os 

DEVICE = os.getenv("DEVICE")
LOGGER_CONFIG_PATH = os.getenv("LOGGER_CONFIG_PATH")
try:
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 8))
except:
    BATCH_SIZE: int = 8
    
@dataclass(frozen=True)
class ArabicConfig:
    TRANSLATE_MODEL_PATH: str = os.getenv("ARABIC_TRANSLATE_MODEL_PATH", "HebArabNlpProject/mt-ar-he") 
    TOKENIZER_NAME: str = os.getenv("ARABIC_TOKENIZER_NAME", "Helsinki-NLP/opus-mt-ar-he")
    TOEKN_CLASSIFY_MODEL: str = os.getenv("ARABIC_TOEKN_CLASSIFY_MODEL", "hatmimoha/arabic-ner")  
    TOEKN_CLASSIFY_TARGET_MODEL: str = os.getenv("HEBREW_TOEKN_CLASSIFY_MODEL", "dicta-il/dictabert-ner")

@dataclass(frozen=True)
class HebrewConfig:
    TRANSLATE_MODEL_PATH: str = os.getenv("HEBREW_TRANSLATE_MODEL_PATH", "HebArabNlpProject/mt-he-ar")
    TOKENIZER_NAME: str = os.getenv("HEBREW_TOKENIZER_NAME", "Helsinki-NLP/opus-mt-he-ar")
    TOEKN_CLASSIFY_MODEL: str = os.getenv("HEBREW_TOEKN_CLASSIFY_MODEL", "dicta-il/dictabert-ner")
    TOEKN_CLASSIFY_TARGET_MODEL: str = os.getenv("ARABIC_TOEKN_CLASSIFY_MODEL", "hatmimoha/arabic-ner")
