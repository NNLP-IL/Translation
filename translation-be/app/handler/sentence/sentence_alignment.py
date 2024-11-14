from unittest.mock import inplace

from simalign import SentenceAligner
from ..objects.map import TranslitMap
from ..entities.entity_extract import EntityExtract
from ...shared.shared import get_transliterate_service
import re
from utils.print_colors import *
from ...objects.entities import Entity, Sentence
from typing import List
from nrm_logger.src.logger.nrm_logger import NRMLogger, LogLevel
from consts import LOGGER_CONFIG_PATH
from ...utils.print_colors import GREEN
from transformers import AutoModel, AutoTokenizer


def is_english_word(word: str):
    """
    Pattern to match sequences of English words (and spaces between them) as a single match,
    as well as email addresses.
    """
    pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(([A-Za-z]+(?:\s+[A-Za-z]+)*))')
    matches = pattern.findall(word)
    if len(matches) > 0:
        return True
    else:
        return False


class CustomSentenceAligner(SentenceAligner):
    # List of common Hebrew prepositions
    HEBREW_PREPOSITIONS = ["ב", "כ", "ל", "מ", "שה", "ש", "ו", "ה", "מה", "וכש", "כש", "כשה"]
    VALID_MATCH_METHODS = ["inter", "itermax", "mwmf"]
    SEGMENT_SEPERATORS = ['[CLS]', '[SEP]']

    def __init__(self, entity_extract: EntityExtract = None, model="intfloat/multilingual-e5-large", token_type="bpe",
                 token_classify_model: str = None, token_classify_target_model: str = None,
                 matching_method: str = "mwmf", device: str = None, translit_map: TranslitMap = TranslitMap()):
        super().__init__(model=model, token_type=token_type, device=device)
        self.logger = NRMLogger(logger_name="CustomSentenceAligner", config_path=LOGGER_CONFIG_PATH)
        self.matching_method = self._validate_match_method(method=matching_method)
        self.base_entity_extract = entity_extract if entity_extract else EntityExtract(
            token_classify_model=token_classify_model)
        self.target_entity_extract = EntityExtract(token_classify_model=token_classify_target_model)
        self.transliterate = get_transliterate_service()
        self.translit_map: TranslitMap = translit_map
        self.segment_tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
        self.segment_model = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)
        self.alignments: list[str] = [""]

    def _validate_match_method(self, method: str):
        if method in self.VALID_MATCH_METHODS:
            return method
        self.logger.log(f"Warning: Invalid matching method '{method}'. Defaulting to 'itermax'.",
                        level=LogLevel.WARNING)
        return "itermax"

    @staticmethod
    def sentence2words(text: str):
        return re.split('( |, |، |\.)', text)

    @staticmethod
    def get_english_words_index(words: List[str]):
        return [idx for idx, word in enumerate(words) if is_english_word(word)]

    def validate_hebrew_preposition_prefix(self, word: str):
        """ Check if there is a preposition """
        for preposition in self.HEBREW_PREPOSITIONS:
            if word.startswith(preposition):
                return preposition
        return ''

    def remove_hebrew_prepositions(self, word: str):
        for preposition in self.HEBREW_PREPOSITIONS:
            if word.startswith(preposition + '-'):
                return word[len(preposition) + 1:]
        return word

    def get_words_alignment(self, src_sent: List[str], tgt_sent: List[str]):
        """
        Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
        Returns: __dict__: dictionary with different matching methods.
        """
        alignments = self.get_word_aligns(src_sent, tgt_sent)
        return alignments[self.matching_method]

    def _update_english_matches_map(self, curr_eng_matches: dict, trg_word_idx: int, src_word: str, tgt_word: str):
        prefix = self.validate_hebrew_preposition_prefix(word=tgt_word)
        prefix += '-' if prefix else ''
        if trg_word_idx not in curr_eng_matches.keys():
            curr_eng_matches[trg_word_idx] = [prefix + src_word]
        else:
            curr_eng_matches[trg_word_idx].append(src_word)
        return curr_eng_matches

    def swap_words_target(self, src_english_words_idxs: List[int], entities_map: dict, src_sentence_list: List[int],
                          tgt_sentence_list: List[str]):
        """
        Keep track on the matches for the English words or names from entities with a dictionary:
        ~ keys - the target indexes we want to replace their words
        ~ values - list of the words we want to put in these indexes, if there are multiple values we will join them by ' '
        """

        entities_map = {key: ent for key, ent in entities_map.items() if
                        ent.tag.endswith('PER') or ent.tag.endswith('PERSON')}

        matches = {}
        has_match = []
        for match in self.alignments:
            src_word_idx, trg_word_idx = match
            if src_word_idx in src_english_words_idxs:
                if tgt_english_idx := self.get_align_of_two_english_words(src_word_idx,
                                                                          tgt_sentence_list) and not src_word_idx in has_match:
                    self.logger.log(
                        f"{PRINT_START}{RED}{src_sentence_list[src_word_idx]} <-> {tgt_sentence_list[tgt_english_idx]}{PRINT_STOP}")
                    matches = self._update_english_matches_map(curr_eng_matches=matches, trg_word_idx=tgt_english_idx,
                                                               src_word=src_sentence_list[src_word_idx],
                                                               tgt_word=tgt_sentence_list[tgt_english_idx])
                    has_match.append(src_word_idx)
                elif not src_word_idx in has_match:
                    self.logger.log(
                        f"{PRINT_START}{RED}{src_sentence_list[src_word_idx]} <-> {tgt_sentence_list[trg_word_idx]}{PRINT_STOP}")
                    matches = self._update_english_matches_map(curr_eng_matches=matches, trg_word_idx=trg_word_idx,
                                                               src_word=src_sentence_list[src_word_idx],
                                                               tgt_word=tgt_sentence_list[trg_word_idx])
                    has_match.append(src_word_idx)
                else:
                    if self.get_align_of_two_english_words(src_word_idx, tgt_sentence_list):
                        continue
                    else:
                        matches[trg_word_idx] = ''
            elif not trg_word_idx in matches and src_word_idx in entities_map:
                if not src_word_idx in has_match:
                    self.logger.log(
                        f"{PRINT_START}{RED}{src_sentence_list[src_word_idx]} <-> {tgt_sentence_list[trg_word_idx]}{PRINT_STOP}")
                    matches[trg_word_idx] = [entities_map[src_word_idx].word]
                    has_match.append(src_word_idx)
                else:
                    matches[trg_word_idx] = ''

        # Replace words in the target and remove empty strings
        for idx, aligned_words in matches.items():
            tgt_sentence_list[idx] = ' '.join(aligned_words)
        return ''.join([word for word in tgt_sentence_list if word != ''])

    def align_entities(self, sentence: str, sentence_list: List[str], is_source: bool = True):
        """
        Creates entities map, if is_source equals True -  using `base_entity_extract` and returns the transliterated word as indexed (key) value.
        else, using `target_entity_extract` and returns word as indexed (key) value.
        entity_map - keys > word index, values > Entity object.
        """
        entity_extract = self.base_entity_extract if is_source else self.target_entity_extract
        entities_map = {}
        for _extracted_sentence in entity_extract.get_entities(text=sentence, split_text=True):
            for entity in _extracted_sentence.entities:
                if not entity.word in self.HEBREW_PREPOSITIONS:
                    if is_source:
                        entity_transliterate = self.transliterate.transliterate(content=entity.word,
                                                                                src=self.translit_map.src,
                                                                                tgt=self.translit_map.tgt,
                                                                                validate_src=False)
                        word = entity_transliterate.split()[0] if entity_transliterate.split() else entity_transliterate
                    else:
                        word = entity.word
                    if entity.word in sentence_list:
                        src_word_idx = sentence_list.index(entity.word)
                        entities_map[src_word_idx] = Entity(word=word, tag=entity.tag,
                                                            tag_hex=entity.tag_hex, source=entity.source)
        return entities_map

    def tokenize_sentences_to_words(self, src_sentence: str, tgt_sentence: str):
        """ Tokenize source and target sentences to words """
        src_sentence_list = self.sentence2words(src_sentence)
        tgt_sentence_list = self.sentence2words(tgt_sentence)
        return src_sentence_list, tgt_sentence_list

    def clean_duplicate_alignments(self, src_sentence_list):
        if not self.alignments:
            return []

        latest_word = src_sentence_list[self.alignments[0][0]]
        result = [self.alignments[0]]
        for item in self.alignments[1:]:
            if src_sentence_list[item[0]] != latest_word:
                result.append(item)
            latest_word = src_sentence_list[item[0]]
        return result

    def join_segmented_tokens(self, token_lists: List):
        # Join the tokens into a single string with spaces
        merged_tokens = ' '.join(token for sublist in token_lists for inner_list in sublist for token in inner_list if
                                 token not in self.SEGMENT_SEPERATORS)
        return merged_tokens

    def sentence_segmentation(self, sentence: str):
        """
        Segments the sentence, fixing apostrophe, partial number and time format issues etc.
        Returns the segmented sentence with corrections applied.
        """
        sentence = re.sub(r"(?:(?:['’`])) ", "’_", sentence)
        segmented_sentence = self.segment_model.predict([sentence], self.segment_tokenizer)
        sentence = self.join_segmented_tokens(segmented_sentence)
        sentence = re.sub(r" (?:(?:['’`])) ", "’", sentence) # fix splitted ' character
        sentence = re.sub(r'(\d)\s*([-/\\:;_=\.+*!&|#^~])\s*(\d)', r'\1\2\3', sentence) # fix splitted numbers
        return re.sub("_", " ", sentence) if "_" in sentence else sentence

    def apply_segmentation(self, src_sentence: str, tgt_sentence: str, entity_align: bool = False):
        """
        Segmentation for the hebrew sentence. Eg., אכלתי ביפו   > אכלתי ב יפו
        """
        if entity_align:
            if self.translit_map.src == 'ar':
                tgt_sentence = self.sentence_segmentation(sentence=tgt_sentence)
            else:
                src_sentence = self.sentence_segmentation(sentence=src_sentence)

        return src_sentence, tgt_sentence

    def unsegment_hebrew_sentence(self, translated_sen: str):
        if self.translit_map.src == 'ar':
            # Split the sentence into words
            words = translated_sen.split()
            new_words = []
            # Iterate over each word in the sentence
            i = 0
            while i < len(words):
                word = words[i]
                # Check if word is one of the specified prefixes
                if word in self.HEBREW_PREPOSITIONS and i + 1 < len(words):
                    # Concatenate with the next word
                    new_word = word + words[i + 1]
                    new_words.append(new_word)
                    # Skip the next word as it is already concatenated
                    i += 2
                else:
                    new_words.append(word)
                    i += 1
            # Join the list of new words into a sentence
            return ' '.join(new_words)
        else:
            return translated_sen

    def is_made_of_english_chars(self, word):
        allowed_special_chars = set(" .,!?:;'-")

        # Check if all characters are English letters, digits, or in the allowed set
        return all(
            char.isalpha() and char.isascii() or
            char.isdigit() or
            char in allowed_special_chars
            for char in word
        )

    def get_align_of_two_english_words(self, src_word_idx, tgt_sentence_list: List):
        for match in self.alignments:
            if match[0] == src_word_idx and self.is_made_of_english_chars(tgt_sentence_list[match[1]]):
                return match[1]

    def sentences_align(self, src_sentence: str, tgt_sentence: str, entity_align: bool = False):
        """
        Post-process translation func.
        Inputs:
        - src_sentence (sentence which was translated)
        - tgt_sentence (translated sentence)
        - entity_align (if applying NER model)
        Aligns between english/entities inside source sentence into target sentence in the correct manner.
        """
        src_sentence, tgt_sentence = self.apply_segmentation(src_sentence, tgt_sentence, entity_align)
        src_sentence_list, tgt_sentence_list = self.tokenize_sentences_to_words(src_sentence=src_sentence,
                                                                                tgt_sentence=tgt_sentence)
        # Remove prepositions from words in the source (to handle he-ar translation)
        if self.translit_map.src == 'he':
            src_sentence_list = [self.remove_hebrew_prepositions(word) for word in src_sentence_list if word]

        # Align
        self.alignments = self.get_words_alignment(src_sentence_list, tgt_sentence_list)
        for match in self.alignments:
            self.logger.log(
                f"{PRINT_START}{GREEN}{src_sentence_list[match[0]]} - {tgt_sentence_list[match[1]]}{PRINT_STOP}")

        # Parses translates sentence with english text and entities if `entity_align` requetsed.
        src_english_words_idxs = self.get_english_words_index(words=src_sentence_list)
        entities_map = self.align_entities(sentence=src_sentence,
                                           sentence_list=src_sentence_list,
                                           is_source=True) if entity_align else {}
        entities_tgt_map = self.align_entities(sentence=tgt_sentence,
                                               sentence_list=tgt_sentence_list,
                                               is_source=False) if entity_align else {}
        translated_sentence = self.swap_words_target(src_english_words_idxs=src_english_words_idxs,
                                                     entities_map=entities_map, src_sentence_list=src_sentence_list,
                                                     tgt_sentence_list=tgt_sentence_list)
        translated_sentence = self.unsegment_hebrew_sentence(translated_sen=translated_sentence)

        # Validates sentence response object.
        aligned_entities = list(entities_tgt_map.values()) + list(entities_map.values())
        aligned_sentence = Sentence(sentence=translated_sentence,
                                    entities=aligned_entities)
        aligned_sentence.update_entities_locations() if aligned_sentence.entities != [] else None
        return aligned_sentence
