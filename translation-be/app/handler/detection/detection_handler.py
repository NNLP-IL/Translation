from typing import Optional
from ...objects.language import LanguageForm
from .consts import SUPPORTED_LANGUAGES
from ...consts import LOGGER_CONFIG_PATH
from langdetect import detect
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel

class DetectionHandler:
    """
    Language Detection by `content` words.
    """
    def __init__(self):
        self.supported_languages = SUPPORTED_LANGUAGES
        self.logger = NRMLogger(logger_name="DetectionHandler", config_path=LOGGER_CONFIG_PATH)
        
    def detect_language(self, words: str) -> Optional[LanguageForm]:
        self.logger.log(f"Start language detection - {words}")
        result: str = ''
        if words:
            try:
                result = detect(words)
            except:
                return LanguageForm(language='', display_text='unknown')
        if result in self.supported_languages.keys():
            return LanguageForm(language=result, **self.supported_languages[result])
        self.logger.log("Couldnt detect language, not found in service `supported languages`", level=LogLevel.ERROR)