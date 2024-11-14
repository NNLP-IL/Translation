from ..handler.transliteration import GimeltraTransliterationService, TranslitMeLocalTransliterationService, TranslitMeRestTransliterationService  

class TransliterateEngine:
    GIMELTRA = GimeltraTransliterationService
    TRANSLIT_ME_CLOUD = TranslitMeRestTransliterationService
    TRANSLIT_ME_LOCAL = TranslitMeLocalTransliterationService
