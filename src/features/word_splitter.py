"""
Smart Word Splitter for Arabic-English Transliteration

Handles the key challenge: Arabic definite article "ال" is attached to words
but becomes a separate word "Al-" in English transliteration.

Examples:
- العيون (one word) → Al-Oyun (two English words: Al + Oyun)
- غيل الحاضنه (two words) → Ghayl Al-Hadina (three English words)
- بني حجاج (two words) → Bani Hajjaj (two English words)
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WordPair:
    """A paired Arabic word with its English transliteration."""
    arabic: str
    english: str


class ArabicWordSplitter:
    """
    Splits Arabic-English pairs into aligned word pairs for training.
    
    Key insight: Arabic "ال" prefix is attached but transliterates to "Al-" prefix.
    We need to detect this and split accordingly.
    """
    
    # Arabic definite article pattern (ال at start of word)
    AL_PATTERN = re.compile(r'^ال')
    
    # Common Al- patterns in English (various capitalization)
    AL_EN_PATTERNS = [
        re.compile(r'^[Aa][Ll][-\s]?', re.IGNORECASE),  # Al-, al-, Al , al 
        re.compile(r'^[Aa][Nn][-\s]?'),  # An- (for assimilation like الن → An-)
        re.compile(r'^[Aa][Ss][Ss]?[-\s]?'),  # As-, Ass- (الس → As-)
        re.compile(r'^[Aa][Tt][-\s]?'),  # At- (الت → At-)
        re.compile(r'^[Aa][Dd][-\s]?'),  # Ad- (الد → Ad-)
        re.compile(r'^[Aa][Rr][-\s]?'),  # Ar- (الر → Ar-)
    ]
    
    # Words that don't take the definite article pattern
    NON_AL_PREFIXES = ['Bani', 'Beni', 'Bayt', 'Wadi', 'Jabal', 'Dar', 'Shat', 
                        'Sha\'b', 'Sha\'ab', 'Sha\'bat', 'Hasab', 'Hassi', 'Hasib',
                        'Ghayl', 'Najd', 'Nawbat', 'Qanahu', 'Akmat', 'Qaryat',
                        'Hanna', 'Hannah', 'Nimrah', 'Tarrah', 'Hayjah', 'Hijat',
                        'Rahbah', 'Muqaddam', 'Abu', 'Umm', 'Bir', 'Jazirat']
    
    def __init__(self):
        """Initialize the word splitter."""
        pass
    
    def has_arabic_al(self, word: str) -> bool:
        """Check if Arabic word starts with definite article ال."""
        return bool(self.AL_PATTERN.match(word))
    
    def remove_arabic_al(self, word: str) -> str:
        """Remove the ال prefix from Arabic word."""
        return self.AL_PATTERN.sub('', word)
    
    def has_english_al(self, word: str) -> Tuple[bool, str, str]:
        """
        Check if English word starts with Al- pattern.
        Returns: (has_al, al_part, remaining)
        """
        for pattern in self.AL_EN_PATTERNS:
            match = pattern.match(word)
            if match:
                al_part = match.group()
                remaining = word[len(al_part):]
                # Normalize the al part
                normalized_al = al_part.rstrip('-').rstrip()
                return True, normalized_al, remaining
        return False, '', word
    
    def split_arabic_words(self, text: str) -> List[str]:
        """Split Arabic text into words on whitespace and dash."""
        words = re.split(r'[\s\-]+', text)
        return [w.strip() for w in words if w.strip()]
    
    def split_english_words(self, text: str) -> List[str]:
        """Split English text into words on whitespace, - and _, but preserve Al- prefix.
        
        Rules:
        - Split on whitespace (always a word separator)
        - Split on '_' (underscore is always a word separator)
        - Split on '-' EXCEPT when it follows Al-, An-, As-, At-, Ad-, Ar- patterns
        
        Examples:
        - "Al-Sawerah_Al-Kadm" -> ['Al-Sawerah', 'Al-Kadm']
        - "Al-Bara-Al-Kadf" -> ['Al-Bara', 'Al-Kadf']
        - "Wadi Al-Bir" -> ['Wadi', 'Al-Bir']
        """
        text = text.strip()
        
        # First split on whitespace and underscore
        parts = re.split(r'[\s_]+', text)
        
        result = []
        for part in parts:
            if not part.strip():
                continue
            
            # Now split on '-', but keep Al- patterns attached
            sub_parts = self._split_preserving_al(part.strip())
            result.extend(sub_parts)
        
        return [w.strip() for w in result if w.strip()]
        
        return [w.strip() for w in result if w.strip()]
    
    def _split_preserving_al(self, text: str) -> List[str]:
        """Split on '-' but keep Al- prefix attached.
        
        Al-Bara-Al-Kadf -> ['Al-Bara', 'Al-Kadf']
        Wadi-Al-Bir -> ['Wadi', 'Al-Bir']
        Al-Sawerah-Jahmh -> ['Al-Sawerah', 'Jahmh']
        """
        # Define Al- patterns to preserve (case insensitive)
        al_patterns = ['Al-', 'al-', 'An-', 'an-', 'As-', 'as-', 
                       'At-', 'at-', 'Ad-', 'ad-', 'Ar-', 'ar-']
        
        # First, temporarily replace Al- patterns to protect them
        protected = text
        placeholder = '\x00AL\x00'  # Temporary placeholder
        for pattern in al_patterns:
            protected = protected.replace(pattern, placeholder + pattern[:-1] + '\x01')
        
        # Now split on remaining '-'
        parts = protected.split('-')
        
        # Restore Al- patterns
        result = []
        for part in parts:
            if not part.strip():
                continue
            # Restore the Al- prefix
            restored = part.replace(placeholder, '').replace('\x01', '-')
            result.append(restored)
        
        return result
    
    def align_words(self, arabic: str, english: str) -> List[WordPair]:
        """
        Align Arabic and English words, handling the Al- mapping.
        
        Strategy:
        1. Split both into tokens
        2. For each Arabic word with ال, expect Al-X in English
        3. For Arabic words without ال, expect direct mapping
        """
        ar_words = self.split_arabic_words(arabic)
        en_words = self.split_english_words(english)
        
        pairs = []
        ar_idx = 0
        en_idx = 0
        
        while ar_idx < len(ar_words) and en_idx < len(en_words):
            ar_word = ar_words[ar_idx]
            en_word = en_words[en_idx]
            
            # Check if Arabic word has ال prefix
            if self.has_arabic_al(ar_word):
                # Arabic word like "العيون" should map to English "Al-Oyun" or "Al Oyun"
                # Check if English word has Al- prefix
                has_al, al_part, remaining = self.has_english_al(en_word)
                
                if has_al:
                    # If remaining is empty, Al is a separate word
                    if not remaining:
                        # Al is separate word, combine with next
                        if en_idx + 1 < len(en_words):
                            combined_en = f"{al_part}-{en_words[en_idx + 1]}"
                            pairs.append(WordPair(ar_word, combined_en))
                            en_idx += 2
                        else:
                            # Just Al with nothing following - fallback
                            pairs.append(WordPair(ar_word, en_word))
                            en_idx += 1
                    else:
                        # Al- is part of the word (e.g., "Al-Qal'ah")
                        pairs.append(WordPair(ar_word, en_word))
                        en_idx += 1
                else:
                    # No Al in English but Arabic has ال - this is unusual
                    # Just pair them directly
                    pairs.append(WordPair(ar_word, en_word))
                    en_idx += 1
            else:
                # No ال in Arabic word
                # Check if English word starts with Al (it shouldn't normally)
                has_al, al_part, remaining = self.has_english_al(en_word)
                
                if has_al and not remaining and en_idx + 1 < len(en_words):
                    # "Al" is separate but Arabic doesn't have ال
                    # This might be a multi-word name like "Bani Ali"
                    # Just take one English word
                    pairs.append(WordPair(ar_word, en_word))
                    en_idx += 1
                else:
                    # Normal word-to-word mapping
                    pairs.append(WordPair(ar_word, en_word))
                    en_idx += 1
            
            ar_idx += 1
        
        # Handle remaining words (if any)
        while ar_idx < len(ar_words):
            # More Arabic words than English - append remaining Arabic
            # This shouldn't happen in well-aligned data
            ar_idx += 1
        
        while en_idx < len(en_words):
            # More English words than Arabic
            # This might happen with complex names - attach to last pair
            if pairs:
                pairs[-1] = WordPair(
                    pairs[-1].arabic,
                    pairs[-1].english + ' ' + en_words[en_idx]
                )
            en_idx += 1
        
        return pairs
    
    def extract_training_pairs(self, arabic: str, english: str, 
                               min_length: int = 1) -> List[Tuple[str, str]]:
        """
        Extract word-level training pairs from a full name.
        
        Args:
            arabic: Full Arabic name
            english: Full English transliteration
            min_length: Minimum Arabic word length to include
            
        Returns:
            List of (arabic_word, english_word) tuples
        """
        word_pairs = self.align_words(arabic, english)
        
        result = []
        for pair in word_pairs:
            # Filter out very short words
            if len(pair.arabic) >= min_length:
                result.append((pair.arabic.strip(), pair.english.strip()))
        
        return result


def create_word_level_dataset(df, arabic_col: str = 'arabic_name', 
                               english_col: str = 'english_name',
                               min_word_length: int = 2) -> 'pd.DataFrame':
    """
    Convert a name-level dataset to word-level for training.
    
    Args:
        df: DataFrame with arabic and english name columns
        arabic_col: Name of Arabic column
        english_col: Name of English column
        min_word_length: Minimum word length to include
        
    Returns:
        DataFrame with word-level pairs
    """
    import pandas as pd
    
    splitter = ArabicWordSplitter()
    word_pairs = []
    
    for idx, row in df.iterrows():
        arabic = str(row[arabic_col]).strip()
        english = str(row[english_col]).strip()
        
        try:
            pairs = splitter.extract_training_pairs(arabic, english, min_word_length)
            for ar_word, en_word in pairs:
                word_pairs.append({
                    'arabic_word': ar_word,
                    'english_word': en_word,
                    'source_arabic': arabic,
                    'source_english': english,
                    'source_idx': idx
                })
        except Exception as e:
            # Skip problematic entries
            print(f"Warning: Could not process row {idx}: {e}")
            continue
    
    return pd.DataFrame(word_pairs)


def validate_alignment(df_words: 'pd.DataFrame', sample_size: int = 20):
    """
    Print sample alignments for validation.
    """
    print(f"\n{'='*60}")
    print("WORD-LEVEL ALIGNMENT VALIDATION")
    print(f"{'='*60}")
    
    # Get unique source entries
    unique_sources = df_words.groupby('source_idx').first().head(sample_size)
    
    for idx, row in unique_sources.iterrows():
        # Get all words for this source
        source_words = df_words[df_words['source_idx'] == idx]
        
        print(f"\nSource: {row['source_arabic']} → {row['source_english']}")
        print("Words:")
        for _, word_row in source_words.iterrows():
            print(f"  {word_row['arabic_word']:15} → {word_row['english_word']}")


# Quick test
if __name__ == '__main__':
    # Test cases
    test_cases = [
        ("العيون", "Al-Oyun"),  # ال attached → Al-
        ("غيل الحاضنه", "Ghayl Al-Hadina"),  # Two words, second has ال
        ("بني حجاج", "Bani Hajjaj"),  # No ال in either
        ("وادي البير", "Wadi Al-Bir"),  # Second word has ال
        ("الصنمه", "Al-Sunmah"),  # Single word with ال
        ("حسي بن علوان", "Hassi Bin Alwan"),  # Three Arabic words, three English
        ("الشبارة", "Al-Shabara"),  # ال word
        ("نوبة عامر", "Nawbat Amer"),  # No ال
    ]
    
    splitter = ArabicWordSplitter()
    
    print("Word Alignment Tests:")
    print("=" * 60)
    
    for ar, en in test_cases:
        pairs = splitter.extract_training_pairs(ar, en)
        print(f"\n{ar} → {en}")
        for ar_word, en_word in pairs:
            status = "✓" if len(ar_word) > 0 else "✗"
            print(f"  {status} {ar_word} → {en_word}")
