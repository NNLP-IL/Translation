from pydantic import BaseModel 
from typing import Optional, Union 
from ...consts import ArabicConfig, HebrewConfig
from ...enums.language import TranslateDirection
from utils.resource_check import ResourceChecker
from ..predict import Translator, TranslatorConfig

class TranslateModels(BaseModel, arbitrary_types_allowed=True):
    device: Optional[str] = "cpu"
    ar2he: Optional[Translator] = None
    he2ar: Optional[Translator] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.ar2he is None:
            self.ar2he = self.get_translator_model(config=ArabicConfig, translate_direction=TranslateDirection.AR2HE)
        if self.he2ar is None:
            self.he2ar = self.get_translator_model(config=HebrewConfig, translate_direction=TranslateDirection.HE2AR)
    
    def resource_checker(self):
        return ResourceChecker.check_cpu_memory_usage if self.device == "cpu" else ResourceChecker.check_gpu_memory_usage
    
    def get_translator_model(self, config: Union[HebrewConfig, ArabicConfig], translate_direction: TranslateDirection):
        """ Initialize model with CPU/GPU usage checks."""
        if self.resource_checker()(threshold = 50):
            translator: Translator = Translator(config=TranslatorConfig(
                                            device=self.device,
                                            translate_direction=translate_direction.value,
                                            trans_model_path=config.TRANSLATE_MODEL_PATH,
                                            tokenizer_path=config.TOKENIZER_NAME,
                                            token_classify_model=config.TOEKN_CLASSIFY_MODEL,
                                            token_classify_target_model=config.TOEKN_CLASSIFY_TARGET_MODEL))
            if self.resource_checker()(threshold = 80):
                return translator
        return None
    
    def swap_translator_usage(self, translator_direction: TranslateDirection):
        if translator_direction.name == TranslateDirection.AR2HE.name:
            self.he2ar = None
            self.ar2he = self.get_translator_model(config=ArabicConfig, translate_direction=TranslateDirection.AR2HE)
        else:
            self.ar2he = None
            self.he2ar = self.get_translator_model(config=HebrewConfig, translate_direction=TranslateDirection.HE2AR)
 
    def translator(self, translate_direction: TranslateDirection):
        """ 
        Retuns Translator model translate direction [TranslateDirection] (Arabic/Heberw),
        Swaps models if CPU/GPU usage checks failed in initialize both Hebrew/Arabic models.
        """
        if translate_direction.name == TranslateDirection.AR2HE.name:
            if not self.ar2he:
                self.swap_translator_usage(translator_direction=translate_direction)
            return self.ar2he
        else:
            if not self.he2ar:
                self.swap_translator_usage(translator_direction=translate_direction)
            return self.he2ar