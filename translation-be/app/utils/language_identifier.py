from typing import Optional
from langcodes import Language as langcode
from ..objects.language import Language, LanguageForm
from ..handler.detection.detection_handler import DetectionHandler
from ..consts import LOGGER_CONFIG_PATH 
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel

class LanguageIdentify:
    detector_handler = DetectionHandler()
    logger = NRMLogger(logger_name="LanguageIdentify", config_path=LOGGER_CONFIG_PATH)
    @classmethod
    def identify(cls, content: str) -> Optional[str]:
        """
        Identify LangCode by `content` words.
        Returns LanguageForm object.
        """
        lang: LanguageForm = cls.detector_handler.detect_language(words=content)
        if lang:
            return lang.language
        return 'he'

    def default_target_by_source(src: str) -> Optional[str]:
        """
        Target language by source language factory.
        """
        return {
            'he': 'ar',
            'ar': 'he'
        }.get(src) or 'en'

    @classmethod
    def search_language_code(cls, lang: str) -> LanguageForm:
        """
        Searching for language argument an LanguageCode.
        Returns `LanguageForm` object. contains (language(`ISO-2`), display_text(`display name`))
        """
        cls.logger.log(message=f"Searching '{lang}' language ISO-2 code")
        try:
            match = langcode.find(lang)
            if match:
                display_name = match.display_name()
                cls.logger.log(message=f"Found ISO-2 code: '{match.language}'")
                return LanguageForm(language=match.language, display_text=display_name)
        except LookupError as e:
            cls.logger.log(message=f"Not found matching language: {lang}", level=LogLevel.ERROR)
            return LanguageForm(language="en", display_text="English")
        return LanguageForm(language=lang, display_text=lang)
    
    # TODO: standard code for language (e.g we might get "Hebr" or "hebrew" - there should be a function to convert it to standard code)