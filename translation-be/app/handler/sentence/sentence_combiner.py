from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_distance
from typing import Optional, List
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel
from consts import LOGGER_CONFIG_PATH

class SentenceCombiner:
    """
    Combines sentences by relative content if exists.
    """
    def __init__(self, embedder_name: str):
        self.embedder = SentenceTransformer(model_name_or_path=embedder_name)
        self.logger = NRMLogger(logger_name="SentenceCombiner", config_path=LOGGER_CONFIG_PATH)
        
    @staticmethod
    def compute_cosine_similarity(embedding1: List[float], embedding2: List[float]):
        """ Using Scipy to calculate cosine similarity """
        similarity = 1 - cosine_distance(embedding1, embedding2)
        return similarity

    def combine(self, array_text: List[str], threshold: float):
        i = 0
        while i < len(array_text) - 1:
            try:
                embeddings_1 = self.embedder.encode(array_text[i])
                embeddings_2 = self.embedder.encode(array_text[i + 1])
                if self.compute_cosine_similarity(embeddings_1, embeddings_2) >= threshold:
                    # Combine sentences by merging the next sentence into the current one
                    array_text[i] = array_text[i] + " " + array_text[i + 1]
                    del array_text[i + 1]
                    continue
            except IndexError:
                self.logger.log(message="Reached the end of the array_text. Exiting.", level=LogLevel.ERROR)
                break
            except Exception as e:
                self.logger.log(message=f"An error occurred: {e}", level=LogLevel.ERROR)
                break
            # Only increment i if sentences were not combined, to move on to the next pair
            i += 1
        return array_text
    