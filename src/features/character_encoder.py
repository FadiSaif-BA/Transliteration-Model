"""
Character encoding utilities for neural network input/output.
Converts characters to numerical indices and vice versa.
"""

import json
from typing import List, Dict, Optional, Tuple
import numpy as np


class CharacterEncoder:
    """Base class for character-to-index encoding."""

    # Special tokens
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    UNK_TOKEN = '<UNK>'

    def __init__(self, characters: Optional[List[str]] = None):
        """
        Initialize character encoder.

        Args:
            characters: List of characters to include in vocabulary
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        if characters:
            self.build_vocab(characters)

    def build_vocab(self, characters: List[str]):
        """
        Build vocabulary from character list.

        Args:
            characters: List of unique characters
        """
        # Start with special tokens
        vocab = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]

        # Add unique characters
        unique_chars = sorted(set(characters))
        vocab.extend(unique_chars)

        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str, add_start: bool = False, add_end: bool = False) -> List[int]:
        """
        Encode text to list of indices.

        Args:
            text: Input text
            add_start: Add START token at beginning
            add_end: Add END token at end

        Returns:
            List of character indices
        """
        indices = []

        if add_start:
            indices.append(self.char_to_idx[self.START_TOKEN])

        for char in text:
            idx = self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN])
            indices.append(idx)

        if add_end:
            indices.append(self.char_to_idx[self.END_TOKEN])

        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode list of indices to text.

        Args:
            indices: List of character indices
            remove_special: Remove special tokens from output

        Returns:
            Decoded text
        """
        chars = []
        special_tokens = {self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN}

        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]

                # Skip special tokens if requested
                if remove_special and char in special_tokens:
                    continue

                # Stop at END token
                if char == self.END_TOKEN:
                    break

                chars.append(char)

        return ''.join(chars)

    def pad_sequence(self, indices: List[int], max_length: int, 
                     pad_left: bool = False) -> List[int]:
        """
        Pad sequence to fixed length.

        Args:
            indices: List of indices
            max_length: Target length
            pad_left: Pad on left side instead of right

        Returns:
            Padded sequence
        """
        pad_idx = self.char_to_idx[self.PAD_TOKEN]

        if len(indices) >= max_length:
            return indices[:max_length]

        padding = [pad_idx] * (max_length - len(indices))

        if pad_left:
            return padding + indices
        else:
            return indices + padding

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def get_pad_idx(self) -> int:
        """Get padding token index."""
        return self.char_to_idx[self.PAD_TOKEN]

    def get_start_idx(self) -> int:
        """Get START token index."""
        return self.char_to_idx[self.START_TOKEN]

    def get_end_idx(self) -> int:
        """Get END token index."""
        return self.char_to_idx[self.END_TOKEN]

    def save(self, filepath: str):
        """
        Save encoder to file.

        Args:
            filepath: Path to save encoder
        """
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """
        Load encoder from file.

        Args:
            filepath: Path to encoder file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data['vocab_size']


class ArabicCharEncoder(CharacterEncoder):
    """Character encoder for Arabic text."""

    def __init__(self, characters: Optional[List[str]] = None):
        """
        Initialize Arabic character encoder.

        Args:
            characters: List of Arabic characters, or None to use defaults
        """
        if characters is None:
            # Default Arabic character set
            characters = self._get_default_arabic_chars()

        super().__init__(characters)

    @staticmethod
    def _get_default_arabic_chars() -> List[str]:
        """Get default Arabic character set."""
        # Basic Arabic letters
        letters = [
            'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
            'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
            'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ة', 'ى', 'ء', 'ئ', 'ؤ'
        ]

        # Diacritics
        diacritics = ['َ', 'ِ', 'ُ', 'ْ', 'ّ', 'ً', 'ٍ', 'ٌ']

        # Additional characters
        additional = [' ', '-']

        return letters + diacritics + additional

    def build_vocab_from_corpus(self, texts: List[str]):
        """
        Build vocabulary from corpus of texts.

        Args:
            texts: List of Arabic texts
        """
        # Collect all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        self.build_vocab(list(all_chars))


class EnglishCharEncoder(CharacterEncoder):
    """Character encoder for English transliteration output."""

    def __init__(self, characters: Optional[List[str]] = None):
        """
        Initialize English character encoder.

        Args:
            characters: List of English characters, or None to use defaults
        """
        if characters is None:
            # Default English character set for transliteration
            characters = self._get_default_english_chars()

        super().__init__(characters)

    @staticmethod
    def _get_default_english_chars() -> List[str]:
        """Get default English character set for transliteration."""
        # Lowercase letters
        lowercase = list('abcdefghijklmnopqrstuvwxyz')

        # Uppercase letters
        uppercase = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # Special characters for transliteration
        special = [' ', '-', "'", 'ā', 'ī', 'ū']

        return lowercase + uppercase + special

    def build_vocab_from_corpus(self, texts: List[str]):
        """
        Build vocabulary from corpus of texts.

        Args:
            texts: List of English transliterated texts
        """
        # Collect all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        self.build_vocab(list(all_chars))


class EncoderPair:
    """Pair of Arabic and English encoders for training."""

    def __init__(self, arabic_encoder: Optional[ArabicCharEncoder] = None,
                 english_encoder: Optional[EnglishCharEncoder] = None):
        """
        Initialize encoder pair.

        Args:
            arabic_encoder: Arabic character encoder
            english_encoder: English character encoder
        """
        self.arabic_encoder = arabic_encoder or ArabicCharEncoder()
        self.english_encoder = english_encoder or EnglishCharEncoder()

    def build_from_parallel_corpus(self, arabic_texts: List[str], 
                                   english_texts: List[str]):
        """
        Build both encoders from parallel corpus.

        Args:
            arabic_texts: List of Arabic texts
            english_texts: List of English texts
        """
        self.arabic_encoder.build_vocab_from_corpus(arabic_texts)
        self.english_encoder.build_vocab_from_corpus(english_texts)

    def encode_pair(self, arabic: str, english: str, 
                   max_arabic_len: int, max_english_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode Arabic-English pair for training.

        Args:
            arabic: Arabic text
            english: English text
            max_arabic_len: Maximum Arabic sequence length
            max_english_len: Maximum English sequence length

        Returns:
            Tuple of (encoder_input, decoder_input, decoder_target)
        """
        # Encode Arabic (encoder input)
        encoder_input = self.arabic_encoder.encode(arabic)
        encoder_input = self.arabic_encoder.pad_sequence(encoder_input, max_arabic_len)

        # Encode English with START token (decoder input)
        decoder_input = self.english_encoder.encode(english, add_start=True)
        decoder_input = self.english_encoder.pad_sequence(decoder_input, max_english_len)

        # Encode English with END token (decoder target)
        decoder_target = self.english_encoder.encode(english, add_end=True)
        decoder_target = self.english_encoder.pad_sequence(decoder_target, max_english_len)

        return (np.array(encoder_input), 
                np.array(decoder_input), 
                np.array(decoder_target))

    def encode_batch(self, arabic_texts: List[str], english_texts: List[str],
                    max_arabic_len: int, max_english_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode batch of Arabic-English pairs.

        Args:
            arabic_texts: List of Arabic texts
            english_texts: List of English texts
            max_arabic_len: Maximum Arabic sequence length
            max_english_len: Maximum English sequence length

        Returns:
            Tuple of batched arrays (encoder_inputs, decoder_inputs, decoder_targets)
        """
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for arabic, english in zip(arabic_texts, english_texts):
            enc_in, dec_in, dec_target = self.encode_pair(
                arabic, english, max_arabic_len, max_english_len
            )
            encoder_inputs.append(enc_in)
            decoder_inputs.append(dec_in)
            decoder_targets.append(dec_target)

        return (np.array(encoder_inputs), 
                np.array(decoder_inputs), 
                np.array(decoder_targets))

    def save(self, directory: str):
        """
        Save both encoders.

        Args:
            directory: Directory to save encoders
        """
        import os
        os.makedirs(directory, exist_ok=True)

        self.arabic_encoder.save(os.path.join(directory, 'arabic_encoder.json'))
        self.english_encoder.save(os.path.join(directory, 'english_encoder.json'))

    def load(self, directory: str):
        """
        Load both encoders.

        Args:
            directory: Directory containing encoders
        """
        import os

        self.arabic_encoder = ArabicCharEncoder()
        self.english_encoder = EnglishCharEncoder()

        self.arabic_encoder.load(os.path.join(directory, 'arabic_encoder.json'))
        self.english_encoder.load(os.path.join(directory, 'english_encoder.json'))
