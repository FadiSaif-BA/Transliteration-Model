"""
Rule-based transliteration engine.
Applies deterministic Arabic-to-English transliteration rules.
"""

import re
from typing import List, Tuple, Optional
from .arabic_normalizer import ArabicNormalizer
from ..utils.config import get_config


class RuleEngine:
    """
    Apply rule-based transliteration for Arabic text.
    Handles matres lectionis, taa marbouta, definite article, and consonants.
    """

    def __init__(self):
        """Initialize rule engine with configuration and normalizer."""
        self.config = get_config()
        self.normalizer = ArabicNormalizer()
        self._load_mappings()

    def _load_mappings(self):
        """Load character mappings from configuration."""
        rules = self.config.rules

        # Consonant mappings
        self.consonant_map = rules.get('consonants', {})

        # Special endings
        special = rules.get('special_endings', {})
        self.taa_marbouta = special.get('taa_marbouta', {}).get('char', 'ة')
        self.taa_marbouta_trans = special.get('taa_marbouta', {}).get('transliteration', 'ah')
        self.alif_maqsura = special.get('alif_maqsura', {}).get('char', 'ى')

        # Definite article
        article = rules.get('definite_article', {})
        self.def_article_pattern = article.get('pattern', 'ال')
        self.def_article_trans = article.get('transliteration', 'Al')
        self.sun_letters = set(article.get('sun_letters', []))

        # Diacritics mapping
        diacritics = rules.get('diacritics', {})
        self.fatha = diacritics.get('fatha', 'َ')
        self.kasra = diacritics.get('kasra', 'ِ')
        self.damma = diacritics.get('damma', 'ُ')
        self.sukun = diacritics.get('sukun', 'ْ')
        self.shadda = diacritics.get('shadda', 'ّ')

    def apply_rules(self, arabic_text: str) -> str:
        """
        Apply all transliteration rules to Arabic text.

        Args:
            arabic_text: Input Arabic text

        Returns:
            Transliterated English text with rules applied
        """
        if not arabic_text:
            return ""

        # Step 1: Normalize text
        text = self.normalizer.normalize(
            arabic_text,
            normalize_hamza=True,
            remove_diacritics=False,  # Keep diacritics for now
            remove_ignore_chars=True
        )

        # Step 2: Process definite article at the beginning
        text, result = self._process_definite_article(text)

        # Step 3: Convert character by character with context
        result += self._transliterate_with_context(text)

        # Step 4: Post-process (clean up, capitalize)
        result = self._post_process(result)

        return result

    def _process_definite_article(self, text: str) -> Tuple[str, str]:
        """
        Process definite article at the start of text.

        Args:
            text: Arabic text

        Returns:
            Tuple of (remaining_text, transliteration_so_far)
        """
        if text.startswith(self.def_article_pattern):
            # Remove the article from text
            remaining = text[len(self.def_article_pattern):]

            # Check if next letter is a sun letter
            if remaining and remaining[0] in self.sun_letters:
                # Sun letter: article assimilates (al- + س = as-)
                sun_trans = self.consonant_map.get(remaining[0], remaining[0])
                return remaining, f"a{sun_trans}-"
            else:
                # Moon letter: keep 'al-'
                return remaining, f"{self.def_article_trans}-"

        return text, ""

    def _transliterate_with_context(self, text: str) -> str:
        """
        Transliterate text character by character with context awareness.

        Args:
            text: Arabic text (without initial article)

        Returns:
            Transliterated text
        """
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            # Skip diacritics (we'll use them for context but not output them separately)
            if self._is_diacritic(char):
                i += 1
                continue

            # Handle taa marbouta (ة)
            if char == self.taa_marbouta:
                result.append(self.taa_marbouta_trans)
                i += 1
                continue

            # Check for long vowel patterns (matres lectionis)
            long_vowel_result = self._process_long_vowel(text, i)
            if long_vowel_result:
                transliteration, chars_consumed = long_vowel_result
                result.append(transliteration)
                i += chars_consumed
                continue

            # Regular consonant
            if char in self.consonant_map:
                consonant_trans = self.consonant_map[char]

                # Check for explicit diacritic on this consonant
                vowel = self._get_vowel_after_consonant(text, i)

                result.append(consonant_trans + vowel)
                i += 1

            # Handle spaces and unknown characters
            elif char.isspace():
                result.append(' ')
                i += 1
            else:
                # Unknown character - pass through or skip
                i += 1

        return ''.join(result)

    def _process_long_vowel(self, text: str, pos: int) -> Optional[Tuple[str, int]]:
        """
        Detect and process long vowel patterns (matres lectionis).

        Args:
            text: Full text
            pos: Current position

        Returns:
            Tuple of (transliteration, characters_consumed) or None
        """
        if pos >= len(text):
            return None

        char = text[pos]

        # Alif (ا) - always long 'a'
        if char == 'ا':
            # Previous consonant should have fatha
            return ('a', 1)

        # Waw (و) - could be consonant 'w' or long vowel 'ū'
        if char == 'و':
            # Check if it's acting as a vowel
            # Look at next character - if it's a consonant, likely vowel
            # Look at previous - if there's a consonant with damma, definitely vowel
            if self._is_waw_as_vowel(text, pos):
                return ('ou', 1)
            # Otherwise treat as consonant 'w'
            return None

        # Yaa (ي) - could be consonant 'y' or long vowel 'ī'
        if char == 'ي':
            # Check if it's acting as a vowel
            if self._is_yaa_as_vowel(text, pos):
                return ('ee', 1)
            # Otherwise treat as consonant 'y'
            return None

        return None

    def _is_waw_as_vowel(self, text: str, pos: int) -> bool:
        """
        Determine if waw is a long vowel or consonant.

        Args:
            text: Full text
            pos: Position of waw

        Returns:
            True if waw is a vowel
        """
        # If at the start, likely consonant
        if pos == 0:
            return False

        # Check previous character for damma
        prev_pos = pos - 1
        while prev_pos >= 0 and self._is_diacritic(text[prev_pos]):
            if text[prev_pos] == self.damma:
                return True
            prev_pos -= 1

        # If followed by consonant (not another vowel letter), likely vowel
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            if next_char in self.consonant_map and next_char not in ['ا', 'و', 'ي']:
                return True

        # At end of word, could be vowel
        if pos == len(text) - 1:
            return True

        return False

    def _is_yaa_as_vowel(self, text: str, pos: int) -> bool:
        """
        Determine if yaa is a long vowel or consonant.

        Args:
            text: Full text
            pos: Position of yaa

        Returns:
            True if yaa is a vowel
        """
        # If at the start, likely consonant
        if pos == 0:
            return False

        # Check previous character for kasra
        prev_pos = pos - 1
        while prev_pos >= 0 and self._is_diacritic(text[prev_pos]):
            if text[prev_pos] == self.kasra:
                return True
            prev_pos -= 1

        # If followed by consonant, likely vowel
        if pos + 1 < len(text):
            next_char = text[pos + 1]
            if next_char in self.consonant_map and next_char not in ['ا', 'و', 'ي']:
                return True

        # At end of word with preceding consonant, likely vowel
        if pos == len(text) - 1 and pos > 0:
            prev_char = text[pos - 1]
            if prev_char in self.consonant_map:
                return True

        return False

    def _get_vowel_after_consonant(self, text: str, pos: int) -> str:
        """
        Get short vowel sound for a consonant based on following diacritic.

        Args:
            text: Full text
            pos: Position of consonant

        Returns:
            Vowel string ('a', 'i', 'u', or '')
        """
        # Check if next character is a diacritic
        if pos + 1 < len(text):
            next_char = text[pos + 1]

            if next_char == self.fatha:
                return 'a'
            elif next_char == self.kasra:
                return 'i'
            elif next_char == self.damma:
                return 'u'
            elif next_char == self.sukun:
                return ''

        # No explicit diacritic - use placeholder for ML model
        # For now, we'll leave it empty and let the model decide
        return ''

    def _is_diacritic(self, char: str) -> bool:
        """Check if character is a diacritic mark."""
        diacritics = self.config.rules.get('diacritics', {})
        return char in diacritics.values()

    def _post_process(self, text: str) -> str:
        """
        Post-process transliterated text.

        Args:
            text: Transliterated text

        Returns:
            Cleaned and formatted text
        """
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)

        # Clean up spaces around hyphens
        text = re.sub(r' - ', '-', text)
        text = re.sub(r'- ', '-', text)
        text = re.sub(r' -', '-', text)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Strip leading/trailing spaces
        text = text.strip()

        return text

    def get_coverage_stats(self, arabic_text: str) -> dict:
        """
        Get statistics on rule coverage for debugging.

        Args:
            arabic_text: Input Arabic text

        Returns:
            Dictionary with coverage statistics
        """
        normalized = self.normalizer.normalize(arabic_text)

        total_chars = len([c for c in normalized if not c.isspace()])
        taa_marbouta_count = normalized.count(self.taa_marbouta)
        long_vowels = normalized.count('ا') + normalized.count('و') + normalized.count('ي')

        return {
            'total_characters': total_chars,
            'taa_marbouta_count': taa_marbouta_count,
            'long_vowel_letters': long_vowels,
            'rule_coverage_estimate': min(100, (taa_marbouta_count + long_vowels) / max(1, total_chars) * 100)
        }
