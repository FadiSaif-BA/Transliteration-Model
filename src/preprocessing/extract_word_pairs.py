"""
Robust word pair extraction from Yemeni villages dataset.
Extracts Arabic-English word pairs for transliteration model training.

Addresses issues:
- Spacing inconsistencies
- Attached Arabic words
- Non-transliteration entries (translations)
- Word count mismatches
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter


# ==========================================
# CONFIGURATION
# ==========================================

# Paths (relative to project root)
INPUT_FILE = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'yemeni_villages.csv'
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PAIRS = OUTPUT_DIR / 'clean_word_pairs.csv'
OUTPUT_REPORT = OUTPUT_DIR / 'extraction_report.txt'
OUTPUT_REVIEW = OUTPUT_DIR / 'manual_review.csv'

# Arabic prefixes that often attach to following words
ARABIC_ATTACHED_PREFIXES = ['دار', 'بيت', 'وادي', 'جبل', 'حصب', 'شعب', 'نجد', 'نوبة', 'هيجة', 'شعبة', 'قحفة', 'اكمة']

# English prefixes (definite articles) that should be hyphenated
ENGLISH_PREFIXES = r"(Al|El|Ad|As|Ash|Ath|Az|An|Ar|At)"

# Non-transliteration words to filter out
TRANSLATION_WORDS = {
    'city', 'station', 'residential', 'steam', 'bedouin', 'nomads',
    'the', 'camp', 'bridge', 'lower', 'upper'
}


# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class ExtractionStats:
    """Track extraction statistics."""
    total_rows: int = 0
    processed_rows: int = 0
    skipped_compound_mismatch: int = 0
    skipped_word_mismatch: int = 0
    skipped_translations: int = 0
    skipped_too_short: int = 0
    total_pairs_extracted: int = 0
    unique_pairs: int = 0
    low_confidence_pairs: int = 0
    skip_reasons: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_skip_reason(self, reason: str, entry: str):
        if reason not in self.skip_reasons:
            self.skip_reasons[reason] = []
        if len(self.skip_reasons[reason]) < 10:  # Keep only first 10 examples
            self.skip_reasons[reason].append(entry)


# ==========================================
# PRE-PROCESSING
# ==========================================

def normalize_spacing(text: str) -> str:
    """Fix spacing issues in text while preserving word-internal hyphens."""
    if not isinstance(text, str):
        return ""
    # Remove Kashida/Tatweel character (ـ U+0640) - decorative stretching
    text = text.replace('\u0640', '')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Standardize compound separator: normalize " - " variations
    # Only match hyphens with whitespace on at least one side (compound separators)
    # This preserves word-internal hyphens like "Al-Ghayl"
    text = re.sub(r'\s+-\s*|\s*-\s+', ' - ', text)
    return text.strip()


def split_arabic_attached(text: str) -> str:
    """
    Split attached Arabic words.
    E.g., "دارالقعموس" -> "دار القعموس"
    """
    for prefix in ARABIC_ATTACHED_PREFIXES:
        # Pattern: prefix directly followed by Arabic letter (no space)
        pattern = rf'({prefix})(?=[ء-ي])'
        text = re.sub(pattern, r'\1 ', text)
    return text


def is_translation_entry(english: str) -> bool:
    """Check if entry contains translation rather than transliteration."""
    english_lower = english.lower()
    # Check for translation words
    for word in TRANSLATION_WORDS:
        if word in english_lower:
            return True
    # Check for parenthetical annotations
    if '(' in english and ')' in english:
        return True
    return False


def standardize_english_prefixes(text: str) -> str:
    """
    Standardize English prefixes to hyphenated format.
    "Al Oyun" -> "Al-Oyun", "ad dhayjah" -> "Ad-Dhayjah"
    """
    if not isinstance(text, str):
        return ""
    
    # Pattern: prefix followed by space then word
    pattern = re.compile(rf"\b{ENGLISH_PREFIXES}\s+([A-Za-z']+)", re.IGNORECASE)
    
    def replacer(match):
        prefix = match.group(1).capitalize()  # Normalize to "Al", "Ad", etc.
        word = match.group(2)
        return f"{prefix}-{word}"
    
    return pattern.sub(replacer, text)


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize_arabic(text: str) -> List[str]:
    """Tokenize Arabic text into words."""
    text = split_arabic_attached(text)
    return [w.strip() for w in text.split() if w.strip()]


def tokenize_english(text: str) -> List[str]:
    """
    Tokenize English text, preserving hyphenated words as single tokens.
    "Al-Ghayl Al-Majarib" -> ["Al-Ghayl", "Al-Majarib"]
    """
    text = standardize_english_prefixes(text)
    return [w.strip() for w in text.split() if w.strip()]


# ==========================================
# ALIGNMENT & VALIDATION
# ==========================================

def calculate_confidence(ar_word: str, en_word: str) -> str:
    """
    Calculate confidence score for a word pair.
    Returns: 'high', 'medium', or 'low'
    """
    # Length ratio check
    ar_len = len(ar_word)
    en_len = len(en_word.replace('-', ''))  # Ignore hyphens for length
    
    # Reasonable Arabic:English length ratio is roughly 1:2 to 1:4
    ratio = en_len / ar_len if ar_len > 0 else 0
    
    if 1.0 <= ratio <= 5.0:
        return 'high'
    elif 0.5 <= ratio <= 6.0:
        return 'medium'
    else:
        return 'low'


def extract_word_pairs(df: pd.DataFrame, stats: ExtractionStats) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main extraction logic.
    Returns tuple of (high/medium confidence pairs, low confidence pairs).
    """
    valid_pairs = []
    review_pairs = []
    
    stats.total_rows = len(df)
    
    for idx, row in df.iterrows():
        ar_full = normalize_spacing(str(row['arabic_name']))
        en_full = normalize_spacing(str(row['english_name']))
        
        # Skip translation entries
        if is_translation_entry(en_full):
            stats.skipped_translations += 1
            stats.add_skip_reason('translation', f"{ar_full} | {en_full}")
            continue
        
        # Split compound locations (e.g., "الغيل - المجارب")
        if ' - ' in ar_full and ' - ' in en_full:
            ar_parts = ar_full.split(' - ')
            en_parts = en_full.split(' - ')
            
            if len(ar_parts) != len(en_parts):
                stats.skipped_compound_mismatch += 1
                stats.add_skip_reason('compound_mismatch', f"{ar_full} | {en_full}")
                continue
        else:
            ar_parts = [ar_full]
            en_parts = [en_full]
        
        # Process each part
        for ar_part, en_part in zip(ar_parts, en_parts):
            ar_tokens = tokenize_arabic(ar_part)
            en_tokens = tokenize_english(en_part)
            
            if len(ar_tokens) != len(en_tokens):
                stats.skipped_word_mismatch += 1
                stats.add_skip_reason('word_mismatch', 
                    f"AR({len(ar_tokens)}): {ar_part} | EN({len(en_tokens)}): {en_part}")
                continue
            
            # Align word pairs
            for ar_word, en_word in zip(ar_tokens, en_tokens):
                # Filter single-character noise
                if len(ar_word) < 2 or len(en_word) < 2:
                    stats.skipped_too_short += 1
                    continue
                
                confidence = calculate_confidence(ar_word, en_word)
                
                pair = {
                    'arabic_word': ar_word,
                    'english_word': en_word,
                    'source_id': idx,
                    'confidence': confidence
                }
                
                if confidence == 'low':
                    review_pairs.append(pair)
                    stats.low_confidence_pairs += 1
                else:
                    valid_pairs.append(pair)
        
        stats.processed_rows += 1
    
    stats.total_pairs_extracted = len(valid_pairs)
    
    return pd.DataFrame(valid_pairs), pd.DataFrame(review_pairs)


# ==========================================
# REPORTING
# ==========================================

def generate_report(stats: ExtractionStats) -> str:
    """Generate extraction statistics report."""
    lines = [
        "=" * 50,
        "WORD PAIR EXTRACTION REPORT",
        "=" * 50,
        "",
        "INPUT STATISTICS:",
        f"  Total rows in CSV:            {stats.total_rows}",
        f"  Rows processed:               {stats.processed_rows}",
        "",
        "SKIP REASONS:",
        f"  Compound mismatch:            {stats.skipped_compound_mismatch}",
        f"  Word count mismatch:          {stats.skipped_word_mismatch}",
        f"  Translation (not translit.):  {stats.skipped_translations}",
        f"  Too short (< 2 chars):        {stats.skipped_too_short}",
        "",
        "OUTPUT STATISTICS:",
        f"  Total pairs extracted:        {stats.total_pairs_extracted}",
        f"  Unique pairs:                 {stats.unique_pairs}",
        f"  Low confidence (for review):  {stats.low_confidence_pairs}",
        "",
    ]
    
    # Add example skip reasons
    if stats.skip_reasons:
        lines.append("EXAMPLE SKIPPED ENTRIES:")
        lines.append("-" * 50)
        for reason, examples in stats.skip_reasons.items():
            lines.append(f"\n{reason.upper()} ({len(examples)} examples shown):")
            for ex in examples[:5]:  # Show max 5 examples per reason
                lines.append(f"  • {ex[:80]}{'...' if len(ex) > 80 else ''}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main entry point."""
    print(f"Loading data from: {INPUT_FILE}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    
    # Initialize stats
    stats = ExtractionStats()
    
    # Extract pairs
    print("Extracting word pairs...")
    df_pairs, df_review = extract_word_pairs(df, stats)
    
    # Deduplicate
    print("Deduplicating...")
    df_unique = df_pairs.drop_duplicates(subset=['arabic_word', 'english_word'])
    stats.unique_pairs = len(df_unique)
    
    # Save outputs
    df_unique.to_csv(OUTPUT_PAIRS, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df_unique)} unique pairs to: {OUTPUT_PAIRS}")
    
    if len(df_review) > 0:
        df_review.to_csv(OUTPUT_REVIEW, index=False, encoding='utf-8-sig')
        print(f"Saved {len(df_review)} pairs for review to: {OUTPUT_REVIEW}")
    
    # Generate and save report
    report = generate_report(stats)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report to: {OUTPUT_REPORT}")
    
    # Print report to console
    print("\n" + report)
    
    # Preview
    print("\nSAMPLE PAIRS:")
    print(df_unique.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
