import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then sorted unique words.
        """

        # Reset vocabulary
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        self.id_to_word = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token,
        }

        # Collect unique lowercase words
        words = set()
        for text in texts:
            words.update(text.lower().split())

        # Add words in sorted order
        next_id = 4
        for word in sorted(words):
            self.word_to_id[word] = next_id
            self.id_to_word[next_id] = word
            next_id += 1

        self.vocab_size = next_id

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Lowercase before tokenization.
        Unknown words become <UNK>.
        """

        unk_id = self.word_to_id[self.unk_token]

        return [
            self.word_to_id.get(word, unk_id)
            for word in text.lower().split()
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Unknown IDs become <UNK>.
        """

        return " ".join(
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        )