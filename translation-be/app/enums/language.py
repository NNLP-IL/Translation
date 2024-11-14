from enum import Enum

class TranslateDirection(Enum):
    HE2AR = "he2ar"
    AR2HE = "ar2he"
    
class EntityExtractLanguage(Enum):
    HE = 'he'
    AR = 'ar'
    
class TranslitLangCode(Enum):
    HE = 'he'
    AR = 'ar'
    EN = 'en'

class GimeltraLangCode(Enum):
    AR = "Arab"      # Arabic
    HE = "Hebr"      # Hebrew
    AMH = "Ethi"     # Ge'ez (Ethiopic)
    ARC = "Armi"     # Imperial Aramaic
    BRA = "Brah"     # Brahmi
    XCO = "Chrs"     # Chorasmian
    EGY = "Egyp"     # Egyptian (Hieroglyphs)
    XLY = "Elym"     # Elymaic
    ELL = "Grek"     # Greek
    XHT = "Hatr"     # Hatran
    XMN = "Mani"     # Manichaean
    XNA = "Narb"     # Old North Arabian
    XNB = "Nbat"     # Nabataean
    ARC_2 = "Palm"   # Palmyrene (used ARC_2 to avoid conflict with ARC)
    # NONE = "Phli"   # Inscriptional Pahlavi - Not found in Gimeltra library (used NONE_ to avoid conflict with None)
    PAL = "Phlp"     # Psalter Pahlavi
    PHN = "Phnx"     # Phoenician
    XPR = "Prti"     # Inscriptional Parthian
    SMP = "Samr"     # Samaritan
    XSA = "Sarb"     # Old South Arabian
    SOG = "Sogd"     # Sogdian
    SOG_2 = "Sogo"   # Old Sogdian (used SOG_2 to avoid conflict with SOG)
    SYC = "Syrc"     # Syriac
    UGA = "Ugar"     # Ugaritic