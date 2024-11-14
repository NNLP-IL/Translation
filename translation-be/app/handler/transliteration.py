import requests
from ..utils.language_identifier import LanguageIdentify
from typing import Union, Optional
from abc import ABC, abstractmethod
from ..enums.language import GimeltraLangCode, TranslitLangCode
from ..objects.language import Language
from ..consts import LOGGER_CONFIG_PATH
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel
LangCodes = Union[TranslitLangCode, GimeltraLangCode]

class TransliterationService(ABC):
    """
    Abstract Base Class for Transliteration collections modules, collections has some concrete classes that derive from TransliterationService(ABC); which further derived.
    """
    @classmethod
    @staticmethod
    def validate_language_code(language_code: LangCodes, lang: Language) -> LangCodes:
        """
        Validates language arg, by `language_code` enum object. 
        If language inside enum object -> returns language as a `language_code` enum instance,
        else, searcing for language `ISO-2` code, if succeed `ISO-2` search, searcing inside enum instance, returns enum instance -> if founded else defaults("EN"/"HE"). 
        """
        if lang_code:= getattr(language_code, lang.upper(), None):
            return lang_code
        lang_code = LanguageIdentify.search_language_code(lang=lang).language
        return getattr(language_code, lang_code.upper(), getattr(language_code, "EN", getattr(language_code, "HE")))
    
    @abstractmethod
    def _transliterate(self, content: str, from_lang: Language, to_lang: Language):
        raise NotImplementedError("Undefined Transliteration Engine")
    
    @abstractmethod
    def __language_code__(self) -> LangCodes:
        raise NotImplementedError("Undefined Transliteration Engine - Not Found Language Code")
    
    def transliterate(cls, content: str, src: Language = None, tgt: Language = None, validate_src: bool = True) -> str:
        """
        Identify content language if not implemented src arg and sets matching tgt if not implemented.
        Validates src, tgt languages into target `LanguageCode` enum object (defined in services).
        Runs transliterate operation.
        """
        ident_lang = LanguageIdentify.identify(content)
        if src and src != ident_lang and validate_src:
            raise ValueError(f"Source language: {src}, Not match content language: {ident_lang}")
        elif not src:
            src = ident_lang
        src = cls.validate_language_code(language_code=cls.__language_code__(), lang=src)
        tgt = tgt or LanguageIdentify.default_target_by_source(src.value)
        tgt = cls.validate_language_code(language_code=cls.__language_code__(), lang=tgt)
        return cls._transliterate(content=content, from_lang=src, to_lang=tgt)

class GimeltraTransliterationService(TransliterationService):
    """
    Gimeltra performs simplified abjad-only transliteration, and is primarily intended for translating simple texts from modern to ancient scripts. It uses a non-standard romanization scheme. Arabic, Greek or Hebrew letters outside the basic consonant set will not transliterate.
    """
    def __init__(self):
        super().__init__()
        from gimeltra.gimeltra import Transliterator
        self.engine = Transliterator()
        self.logger = NRMLogger(logger_name="GimeltraTranslit", config_path=LOGGER_CONFIG_PATH)
    
    def __language_code__(self) -> GimeltraLangCode:
        return GimeltraLangCode
    
    def _transliterate(self, content: str, from_lang: GimeltraLangCode, to_lang: GimeltraLangCode) -> str:
        self.logger.log(message=f"Transliteration from {from_lang.value} to {to_lang.value} - texts: {content}")
        return self.engine.tr(content, sc=from_lang.value, to_sc=to_lang.value)

class TranslitMeRestTransliterationService(TransliterationService):
    "MEHDIE Transliteration Service - RESTful API that can be used to transliterate names between Hebrew, Arabic and Latin characters."

    API_URL = 'https://hebrew-transliteration-service-snlwejaxvq-ez.a.run.app/'

    def __init__(self):
        super().__init__()
        self.logger = NRMLogger(logger_name="CloudTranslitMe", config_path=LOGGER_CONFIG_PATH) 
        
    def __language_code__(self) -> TranslitLangCode:
        return TranslitLangCode
    
    def _transliterate(self, content: str, from_lang: TranslitLangCode, to_lang: TranslitLangCode) -> str:
        """
        This method invokes a cloud run service to transliterate a list of strings
        (e.g., ['نوعم', 'مانض', 'پيشينة'])
        from the from_lang (e.g., 'ar') to the to_lang (e.g., 'en').
        Supported languages: ('he','ar','en'). Anything non 'he'/'ar' will be treated
        as 'en'
        """
        self.logger.log(message=f"Transliteration from {from_lang.value} to {to_lang.value} - texts: {content}")
        args = {'from_lang': from_lang.value, 'to_lang': to_lang.value, 'data': content.split()}
        self.logger.log(message=f"Posting: {self.API_URL}, args: {args}")
        x = requests.post(self.API_URL, json=args)
        if x.status_code == 200:
            res_list = x.json()['transliterations']
            return ' '.join(res_list)
        self.logger.log(message=f"Failed Translit - {content}, status code: {x.status_code}", level=LogLevel.ERROR)
        
class TranslitMeLocalTransliterationService(TransliterationService):
    "MEHDIE Transliteration Python package"

    def __init__(self):
        super().__init__()
        from translit_me.transliterator import transliterate as tr
        from translit_me import lang_tables
        self.tr = tr
        self.lang_tables = lang_tables
        self.logger = NRMLogger(logger_name="LocalTranslitMe", config_path=LOGGER_CONFIG_PATH) 

    def __language_code__(self) -> TranslitLangCode:
        return TranslitLangCode
    
    def _transliterate(self, content: str, from_lang: TranslitLangCode, to_lang: TranslitLangCode) -> str:
        self.logger.log(message=f"Transliteration from {from_lang.value} to {to_lang.value} - texts: {content}")
        table_name = f"{from_lang.value}_{to_lang.value}".upper()
        if table_name in dir(self.lang_tables):
            table = getattr(self.lang_tables, table_name)
            res = self.tr(content.split(), table)
            return " ".join(res)
        self.logger.log(f"{table_name} transliteration is not supported with translit-me local engine", level=LogLevel.ERROR)