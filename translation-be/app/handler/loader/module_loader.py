from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..chunker import Chunker
from ..sentence.sentence_alignment import CustomSentenceAligner, TranslitMap
from ..entities.entity_extract import EntityExtract
from abc import ABC, abstractmethod
from ...utils.resource_check import ResourceChecker
from typing import Optional

class LoaderError(Exception):
    """
    Custom exception raised for errors in loaders.
    """
    def __init__(self, message):
        super().__init__(message)
    
class Loader(ABC):
    """
    Abstract base class for loaders. Each loader is responsible for loading a specific component.

    Methods:
        load: Abstract method to load a component. Must be implemented by subclasses.
    """
    @abstractmethod
    def _load(self):
        raise NotImplementedError("Load method must be implemented by subclasses")

    def load(self):
        try:
            return self._load()
        except Exception as e:
            raise LoaderError(f"Failed to load tokenizer: {str(e)}")
        
class EntityExtractLoader(Loader):
    def __init__(self, tokenizer: str, token_classify_model: str):
        self.tokenizer = tokenizer
        self.token_classify_model = token_classify_model
        
    def _load(self):
        return EntityExtract(tokenizer=self.tokenizer, token_classify_model=self.token_classify_model)

class TokenizerLoader(Loader):
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path

    def _load(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

class ModelLoader(Loader):
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path 
        self.device = device
        self.resource_checker_callback = ResourceChecker.check_cpu_memory_usage if device == "cpu" else ResourceChecker.check_gpu_memory_usage
        
    def _load(self):
        """ Initialize models with CPU usage checks."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, output_attentions=True)
        model.to(self.device)
        return model
    
class ChunkerLoader(Loader):
    def __init__(self, tokenizer, chunk_size: int):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def _load(self):
        return Chunker(tokenizer=self.tokenizer, chunk_size=self.chunk_size)

class AlignerLoader(Loader):
    def __init__(self, entity_extract: Optional[EntityExtract], alignment_method: str, token_classify_model: str, token_classify_target_model: str,device: str, translit_map: TranslitMap):
        self.entity_extract = entity_extract
        self.alignment_method = alignment_method
        self.token_classify_model =token_classify_model
        self.token_classify_target_model =token_classify_target_model
        self.device = device
        self.translit_map = translit_map
        
    def _load(self):
        return CustomSentenceAligner(entity_extract=self.entity_extract, matching_method=self.alignment_method, token_classify_model=self.token_classify_model, token_classify_target_model=self.token_classify_target_model, device=self.device, translit_map=self.translit_map)
