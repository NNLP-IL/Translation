import os
from ..enums.translit_engine import TransliterateEngine
from ..handler.objects.extract import EntityExtractModels, EntityExtract

# shared public variables:
transliteration_engine = getattr(TransliterateEngine, os.getenv("TRANSLITERATE_ENGINE", "GIMELTRA")) or ModuleNotFoundError(f"Not Found TRANSLITERATE_ENGINE module named: {os.getenv('TRANSLITERATE_ENGINE')}")
transliterate_service = transliteration_engine()
entity_extract_services = EntityExtractModels()

def get_transliterate_service() -> TransliterateEngine:
    return transliterate_service

def get_hebrew_entity_extract() -> EntityExtract:
    return entity_extract_services.hebrew_extract

def get_arabic_entity_extract() -> EntityExtract:
    return entity_extract_services.arabic_extract