from simalign import SentenceAligner
import re
from scripts.utils.print_colors import *
from typing import List


# List of common Hebrew prepositions
hebrew_prepositions = {"ב", "כ", "ל", "מ", "ש", "ו", "ה", "מה", "כש", "כשה"}


def sentence2words(text: str):
    return text.split(' ')


def is_english_word(word: str):
    # Pattern to match sequences of English words (and spaces between them) as a single match,
    # as well as email addresses.
    pattern = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(([A-Za-z]+(?:\s+[A-Za-z]+)*))')
    matches = pattern.findall(word)
    if len(matches) > 0:
        return True
    else:
        return False


class CustomSentenceAligner(SentenceAligner):
    def __init__(self, model="xlmr", token_type="bpe", matching_method="itermax", device=None):
        super().__init__(model=model, token_type=token_type, device=device)

        valid_methods = ["inter", "itermax", "mwmf"]
        if matching_method in valid_methods:
            self.matching_method = matching_method
        else:
            print(f"Warning: Invalid matching method '{matching_method}'. Defaulting to 'itermax'.")
            self.matching_method = "itermax"

    def get_words_alignment(self, src_sent: List[str], tgt_sent: List[str]):
        alignments = self.get_word_aligns(src_sent, tgt_sent)
        # The output is a dictionary with different matching methods.
        # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
        return alignments[self.matching_method]

    def sentences_align(self, src_sentence: str, tgt_sentence: str):

        # The source and target sentences should be tokenized to words.
        src_sentence_list = sentence2words(src_sentence)
        tgt_sentence_list = sentence2words(tgt_sentence)

        # Find english words in the source
        src_english_words_idxs = [idx for idx, word in enumerate(src_sentence_list) if is_english_word(word)]
        if len(src_english_words_idxs) == 0:
            return tgt_sentence

        # Align
        alignments = self.get_words_alignment(src_sentence_list, tgt_sentence_list)
        for match in alignments:
            print(f"{PRINT_START}{GREEN}{src_sentence_list[match[0]]} - {tgt_sentence_list[match[1]]}{PRINT_STOP}")

        # Create the new sentence - swap the original English word with the correlated translated one
        eng_matches = {}
        for eng_word_idx in src_english_words_idxs:
            has_match = False
            for match in alignments:
                src_word_idx, trg_word_idx = match
                if src_word_idx == eng_word_idx:
                    print(f"{PRINT_START}{RED}{src_sentence_list[src_word_idx]} <-> {tgt_sentence_list[trg_word_idx]}{PRINT_STOP}")
                    # Check if there is a preposition, and we need to keep it as a prefix to the translated word
                    prefix = ''
                    for preposition in hebrew_prepositions:
                        if tgt_sentence_list[trg_word_idx].startswith(preposition):
                            if (not prefix in preposition) or (prefix == ''):
                                prefix = preposition
                    if prefix != '':
                        prefix += '-'

                    # Keep track on the matches for the English words with a dictionary:
                    # keys - the target indexes we want to replace their words
                    # values - list of the words we want to put in these indexes,
                    # if there are multiple values we will join them by ' '
                    if not has_match:
                        if trg_word_idx not in eng_matches.keys():
                            eng_matches[trg_word_idx] = [prefix + src_sentence_list[src_word_idx]]
                        else:
                            eng_matches[trg_word_idx].append(src_sentence_list[src_word_idx])
                        has_match = True
                    else:
                        eng_matches[trg_word_idx] = ''

        # Replace words in the target and remove empty strings
        for idx, aligned_words in eng_matches.items():
            tgt_sentence_list[idx] = ' '.join(aligned_words)
        new_trans = ' '.join([word for word in tgt_sentence_list if word != ''])

        return new_trans
