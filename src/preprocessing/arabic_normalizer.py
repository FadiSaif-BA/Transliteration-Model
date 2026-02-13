"""
Arabic text normalization utilities.
Handles Unicode normalization, character variants, and text cleaning.
"""

import unicodedata
import re
from typing import Optional
from src.utils.config import get_config


class ArabicNormalizer:
    """Normalize Arabic text for consistent processing."""

    def __init__(self):
        """Initialize normalizer with configuration."""
        self.config = get_config()
        self._load_mappings()

    def _load_mappings(self):
        """Load character normalization mappings."""
        # Hamza variants normalization
        self.hamza_map = {
            'أ': 'ا',  # Alif with hamza above
            'إ': 'ا',  # Alif with hamza below
            'آ': 'ا',  # Alif with madda
            'ؤ': 'و',  # Waw with hamza
            'ئ': 'ي',  # Yaa with hamza
        }

        # Characters to remove (from config)
        ignore_chars = self.config.rules.get('ignore_chars', [])
        self.ignore_pattern = re.compile('[' + ''.join(ignore_chars) + ']')

    def normalize(self, text: str, 
                  normalize_hamza: bool = True,
                  remove_diacritics: bool = False,
                  remove_ignore_chars: bool = True) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input Arabic text
            normalize_hamza: Normalize hamza variants to base forms
            remove_diacritics: Remove short vowel diacritics
            remove_ignore_chars: Remove kashida and zero-width characters

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Unicode normalization (NFKC)
        text = unicodedata.normalize('NFKC', text)

        # Remove ignore characters (kashida, etc.)
        if remove_ignore_chars:
            text = self.ignore_pattern.sub('', text)

        # Normalize hamza variants
        if normalize_hamza:
            for variant, base in self.hamza_map.items():
                text = text.replace(variant, base)

        # Remove diacritics if requested
        if remove_diacritics:
            text = self.remove_diacritics(text)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritical marks (tashkeel).

        Args:
            text: Input text

        Returns:
            Text without diacritics
        """
        diacritics = self.config.rules.get('diacritics', {})
        diacritic_chars = ''.join(diacritics.values())

        # Create pattern to match any diacritic
        pattern = '[' + re.escape(diacritic_chars) + ']'
        return re.sub(pattern, '', text)

    def has_diacritics(self, text: str) -> bool:
        """
        Check if text contains any diacritical marks.

        Args:
            text: Input text

        Returns:
            True if diacritics present
        """
        diacritics = self.config.rules.get('diacritics', {})
        return any(char in text for char in diacritics.values())

    def is_arabic(self, char: str) -> bool:
        """
        Check if character is Arabic letter.

        Args:
            char: Single character

        Returns:
            True if Arabic letter
        """
        if not char:
            return False
        code = ord(char)
        # Arabic Unicode ranges: 0600-06FF, 0750-077F, 08A0-08FF
        return (0x0600 <= code <= 0x06FF or 
                0x0750 <= code <= 0x077F or 
                0x08A0 <= code <= 0x08FF)

    def extract_arabic_only(self, text: str) -> str:
        """
        Extract only Arabic characters from text.

        Args:
            text: Input text

        Returns:
            Only Arabic characters and spaces
        """
        return ''.join(char for char in text 
                      if self.is_arabic(char) or char.isspace())
