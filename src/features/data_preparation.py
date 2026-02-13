"""
Data preparation utilities for training corpus.
Handles validation, preprocessing, splitting, and quality checks.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from ..preprocessing import ArabicNormalizer, RuleEngine
from ..utils.config import get_config


class DataValidator:
    """Validate and clean parallel corpus data."""

    def __init__(self):
        """Initialize validator with normalizer."""
        self.normalizer = ArabicNormalizer()
        self.issues = []

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate parallel corpus dataframe.

        Args:
            df: DataFrame with 'arabic_name' and 'english_name' columns

        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        self.issues = []
        original_count = len(df)

        # Check required columns
        required_cols = ['arabic_name', 'english_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove null values
        df = df.dropna(subset=required_cols)
        null_count = original_count - len(df)
        if null_count > 0:
            self.issues.append(f"Removed {null_count} rows with null values")

        # Remove empty strings
        df = df[df['arabic_name'].str.strip() != '']
        df = df[df['english_name'].str.strip() != '']
        empty_count = original_count - null_count - len(df)
        if empty_count > 0:
            self.issues.append(f"Removed {empty_count} rows with empty strings")

        # Check for Arabic characters
        df['has_arabic'] = df['arabic_name'].apply(self._contains_arabic)
        non_arabic_count = (~df['has_arabic']).sum()
        if non_arabic_count > 0:
            self.issues.append(f"Warning: {non_arabic_count} rows without Arabic characters")
        df = df.drop('has_arabic', axis=1)

        # Remove duplicates
        duplicates = df.duplicated(subset=['arabic_name', 'english_name'])
        dup_count = duplicates.sum()
        if dup_count > 0:
            df = df[~duplicates]
            self.issues.append(f"Removed {dup_count} duplicate rows")

        # Reset index
        df = df.reset_index(drop=True)

        # Generate report
        report = {
            'original_count': original_count,
            'final_count': len(df),
            'removed_count': original_count - len(df),
            'issues': self.issues,
            'status': 'valid' if len(df) > 0 else 'invalid'
        }

        return df, report

    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        return any(self.normalizer.is_arabic(char) for char in text)

    def check_length_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze length distribution of names.

        Args:
            df: DataFrame with arabic_name and english_name

        Returns:
            Dictionary with length statistics
        """
        arabic_lengths = df['arabic_name'].str.len()
        english_lengths = df['english_name'].str.len()

        return {
            'arabic': {
                'min': int(arabic_lengths.min()),
                'max': int(arabic_lengths.max()),
                'mean': float(arabic_lengths.mean()),
                'median': float(arabic_lengths.median()),
                'std': float(arabic_lengths.std())
            },
            'english': {
                'min': int(english_lengths.min()),
                'max': int(english_lengths.max()),
                'mean': float(english_lengths.mean()),
                'median': float(english_lengths.median()),
                'std': float(english_lengths.std())
            }
        }

    def check_character_coverage(self, df: pd.DataFrame) -> Dict:
        """
        Analyze character coverage in corpus.

        Args:
            df: DataFrame with arabic_name and english_name

        Returns:
            Dictionary with character statistics
        """
        # Collect all unique characters
        arabic_chars = set()
        english_chars = set()

        for text in df['arabic_name']:
            arabic_chars.update(text)

        for text in df['english_name']:
            english_chars.update(text)

        return {
            'arabic_unique_chars': len(arabic_chars),
            'arabic_chars': sorted(list(arabic_chars)),
            'english_unique_chars': len(english_chars),
            'english_chars': sorted(list(english_chars))
        }


class DataPreprocessor:
    """Preprocess parallel corpus for training."""

    def __init__(self):
        """Initialize preprocessor."""
        self.normalizer = ArabicNormalizer()
        self.rule_engine = RuleEngine()

    def preprocess_corpus(self, df: pd.DataFrame, 
                         normalize_arabic: bool = True,
                         apply_rules: bool = False) -> pd.DataFrame:
        """
        Preprocess parallel corpus.

        Args:
            df: DataFrame with arabic_name and english_name
            normalize_arabic: Apply Arabic normalization
            apply_rules: Apply rule engine to get rule-based transliteration

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Normalize Arabic text
        if normalize_arabic:
            df['arabic_normalized'] = df['arabic_name'].apply(
                lambda x: self.normalizer.normalize(x, normalize_hamza=True)
            )
        else:
            df['arabic_normalized'] = df['arabic_name']

        # Apply rule engine if requested
        if apply_rules:
            df['rule_based_trans'] = df['arabic_normalized'].apply(
                self.rule_engine.apply_rules
            )

            # Calculate rule coverage
            df['rule_coverage'] = df['arabic_normalized'].apply(
                lambda x: self.rule_engine.get_coverage_stats(x)['rule_coverage_estimate']
            )

        # Strip whitespace from English
        df['english_cleaned'] = df['english_name'].str.strip()

        # Calculate lengths
        df['arabic_length'] = df['arabic_normalized'].str.len()
        df['english_length'] = df['english_cleaned'].str.len()

        return df


class DataSplitter:
    """Split data into train/validation/test sets."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        config = get_config()
        data_cfg = config.get('data', {})

        self.train_split = data_cfg.get('train_split', 0.7)
        self.val_split = data_cfg.get('val_split', 0.15)
        self.test_split = data_cfg.get('test_split', 0.15)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: train+val vs test
        train_val_size = self.train_split + self.val_split

        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        # Second split: train vs val
        val_size_adjusted = self.val_split / train_val_size

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_seed
        )

        return train_df, val_df, test_df


class GroupBasedSplitter:
    """
    Split data by grouping similar names to prevent data leakage.
    
    This ensures that similar Arabic names (e.g., same prefix/root) 
    don't appear in both training and test sets, giving a more 
    realistic estimate of model generalization.
    """
    
    def __init__(self, random_seed: int = 42, 
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15):
        """
        Initialize group-based splitter.
        
        Args:
            random_seed: Random seed for reproducibility
            train_split: Fraction for training
            val_split: Fraction for validation  
            test_split: Fraction for testing
        """
        self.random_seed = random_seed
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        np.random.seed(random_seed)
    
    def extract_group_key(self, arabic_name: str) -> str:
        """
        Extract group key from Arabic name for grouping similar names.
        
        Uses first 3-4 characters (or prefix before space) as group key.
        This catches common roots and prefixes like ال (Al-).
        
        Args:
            arabic_name: Arabic name string
            
        Returns:
            Group key string
        """
        name = str(arabic_name).strip()
        
        # Remove definite article for grouping
        if name.startswith('ال'):
            name = name[2:]
        
        # Take first 3 chars or until first space
        if ' ' in name:
            key = name.split()[0][:4]
        else:
            key = name[:4] if len(name) >= 4 else name
        
        return key
    
    def split_data(self, df: pd.DataFrame, 
                   arabic_col: str = 'arabic_name') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by groups to prevent leakage.
        
        Args:
            df: Input DataFrame
            arabic_col: Column name containing Arabic text
            
        Returns:
            Tuple of (train_df, val_df, test_df) with no group overlap
        """
        df = df.copy()
        
        # Assign group keys
        df['_group_key'] = df[arabic_col].apply(self.extract_group_key)
        
        # Get unique groups
        unique_groups = df['_group_key'].unique()
        np.random.shuffle(unique_groups)
        
        # Split groups
        n_groups = len(unique_groups)
        train_end = int(n_groups * self.train_split)
        val_end = int(n_groups * (self.train_split + self.val_split))
        
        train_groups = set(unique_groups[:train_end])
        val_groups = set(unique_groups[train_end:val_end])
        test_groups = set(unique_groups[val_end:])
        
        # Split data
        train_df = df[df['_group_key'].isin(train_groups)].drop('_group_key', axis=1)
        val_df = df[df['_group_key'].isin(val_groups)].drop('_group_key', axis=1)
        test_df = df[df['_group_key'].isin(test_groups)].drop('_group_key', axis=1)
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def check_overlap(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, col: str = 'arabic_name') -> dict:
        """
        Check for overlap between splits (for validation).
        
        Args:
            train_df, val_df, test_df: Split DataFrames
            col: Column to check for overlap
            
        Returns:
            Dictionary with overlap statistics
        """
        train_set = set(train_df[col])
        val_set = set(val_df[col])
        test_set = set(test_df[col])
        
        train_val_overlap = train_set.intersection(val_set)
        train_test_overlap = train_set.intersection(test_set)
        val_test_overlap = val_set.intersection(test_set)
        
        return {
            'train_val_overlap': len(train_val_overlap),
            'train_test_overlap': len(train_test_overlap),
            'val_test_overlap': len(val_test_overlap),
            'total_overlap': len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap),
            'samples': list(train_test_overlap)[:5]  # Sample of overlapping items
        }


class NameSplitter:
    """Split compound Yemeni geographical names on ' - ' separator."""

    def __init__(self):
        self.separator = ' - '

    def split_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split compound names into atomic parts.

        Args:
            df: DataFrame with 'arabic_name' and 'english_name' columns

        Returns:
            Expanded DataFrame with split names as separate rows
        """
        new_rows = []

        for idx, row in df.iterrows():
            arabic = str(row['arabic_name']).strip()
            english = str(row['english_name']).strip()

            arabic_parts = arabic.split(self.separator)
            english_parts = english.split(self.separator)

            # Only split if both have matching number of parts
            if len(arabic_parts) == len(english_parts) and len(arabic_parts) > 1:
                # Create separate rows for each part
                for ar_part, en_part in zip(arabic_parts, english_parts):
                    new_rows.append({
                        'arabic_name': ar_part.strip(),
                        'english_name': en_part.strip(),
                        'source': 'split',
                        'original_arabic': arabic,
                        'original_english': english
                    })
            else:
                # Keep original if no split
                new_rows.append({
                    'arabic_name': arabic,
                    'english_name': english,
                    'source': 'original',
                    'original_arabic': None,
                    'original_english': None
                })

        return pd.DataFrame(new_rows)

    def analyze_splits(self, df: pd.DataFrame) -> dict:
        """
        Analyze splitting potential (for statistics).

        Args:
            df: DataFrame with 'arabic_name' and 'english_name' columns

        Returns:
            Dictionary with split statistics
        """
        arabic_has_sep = df['arabic_name'].astype(str).str.contains(self.separator, regex=False)
        english_has_sep = df['english_name'].astype(str).str.contains(self.separator, regex=False)
        both_have_sep = arabic_has_sep & english_has_sep

        splittable_count = 0
        total_parts = 0

        for idx, row in df[both_have_sep].iterrows():
            ar_parts = str(row['arabic_name']).split(self.separator)
            en_parts = str(row['english_name']).split(self.separator)

            if len(ar_parts) == len(en_parts):
                splittable_count += 1
                total_parts += len(ar_parts)

        estimated_new_size = len(df) - splittable_count + total_parts

        return {
            'total_names': len(df),
            'can_be_split': splittable_count,
            'estimated_new_size': estimated_new_size,
            'data_increase_pct': ((estimated_new_size / len(df)) - 1) * 100 if len(df) > 0 else 0
        }


class DatasetBuilder:
    """Build complete dataset pipeline."""

    def __init__(self, output_dir: str = 'data/processed'):
        """
        Initialize dataset builder.

        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.splitter = DataSplitter()

    def build_dataset(self, input_file: str, 
                     normalize: bool = True,
                     apply_rules: bool = False,
                     split_compounds: bool = False,
                     save_splits: bool = True) -> Dict:
        """
        Complete dataset building pipeline.

        Args:
            input_file: Path to input CSV file
            normalize: Apply normalization
            apply_rules: Apply rule engine
            save_splits: Save train/val/test splits

        Returns:
            Dictionary with dataset information and statistics
        """
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples")

        # Step 1: Validate
        print("\nValidating data...")
        df, validation_report = self.validator.validate_dataframe(df)
        print(f"Validation complete: {validation_report['status']}")
        print(f"Final count: {validation_report['final_count']}")
        if validation_report['issues']:
            for issue in validation_report['issues']:
                print(f"  - {issue}")

        # ← ADD STEP 1.5: Split compounds
        if split_compounds:
            print("\nSplitting compound names on ' - '...")
            splitter = NameSplitter()
            split_stats = splitter.analyze_splits(df)

            print(f"Found {split_stats['can_be_split']} splittable names")
            print(
                f"Dataset will grow from {split_stats['total_names']} to {split_stats['estimated_new_size']} (+{split_stats['data_increase_pct']:.1f}%)")

            df = splitter.split_dataframe(df)
            print(f"After splitting: {len(df)} total names")

        # Step 2: Check distributions
        print("\nAnalyzing data distributions...")
        length_stats = self.validator.check_length_distribution(df)
        char_coverage = self.validator.check_character_coverage(df)

        print(f"Arabic length: {length_stats['arabic']['mean']:.1f} ± {length_stats['arabic']['std']:.1f}")
        print(f"English length: {length_stats['english']['mean']:.1f} ± {length_stats['english']['std']:.1f}")
        print(f"Unique Arabic characters: {char_coverage['arabic_unique_chars']}")
        print(f"Unique English characters: {char_coverage['english_unique_chars']}")

        # Step 3: Preprocess
        print("\nPreprocessing data...")
        df = self.preprocessor.preprocess_corpus(df, normalize, apply_rules)

        # Step 4: Split
        print("\nSplitting data...")
        train_df, val_df, test_df = self.splitter.split_data(df)

        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

        # Step 5: Save
        if save_splits:
            print("\nSaving processed data...")
            train_file = self.output_dir / 'train.csv'
            val_file = self.output_dir / 'val.csv'
            test_file = self.output_dir / 'test.csv'

            train_df.to_csv(train_file, index=False, encoding='utf-8')
            val_df.to_csv(val_file, index=False, encoding='utf-8')
            test_df.to_csv(test_file, index=False, encoding='utf-8')

            print(f"✓ Saved to {self.output_dir}/")

            # Save metadata
            metadata = {
                'validation_report': validation_report,
                'length_stats': length_stats,
                'character_coverage': char_coverage,
                'total_samples': len(df),
                'splits': {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                }
            }

            metadata_file = self.output_dir / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved metadata to {metadata_file}")

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'metadata': {
                'validation': validation_report,
                'lengths': length_stats,
                'characters': char_coverage
            }
        }


def create_sample_dataset(output_file: str = 'data/raw/sample_data.csv', num_samples: int = 20):
    """
    Create a sample dataset for testing.

    Args:
        output_file: Path to save sample dataset
        num_samples: Number of samples (max 20)
    """
    # Sample Yemeni place names (real examples)
    sample_data = [
        ('عدن', 'Aden'),
        ('صنعاء', 'Sanaa'),
        ('الحديدة', 'Al-Hudaydah'),
        ('تعز', 'Taiz'),
        ('إب', 'Ibb'),
        ('ذمار', 'Dhamar'),
        ('المكلا', 'Al-Mukalla'),
        ('حضرموت', 'Hadramawt'),
        ('صعدة', 'Saada'),
        ('عمران', 'Amran'),
        ('حجة', 'Hajjah'),
        ('المحويت', 'Al-Mahwit'),
        ('الضالع', 'Ad-Dhale'),
        ('لحج', 'Lahj'),
        ('أبين', 'Abyan'),
        ('شبوة', 'Shabwah'),
        ('المهرة', 'Al-Mahrah'),
        ('ريمة', 'Raymah'),
        ('الجوف', 'Al-Jawf'),
        ('مأرب', 'Marib'),
    ]

    # Create DataFrame
    df = pd.DataFrame(sample_data[:num_samples], columns=['arabic_name', 'english_name'])

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"✓ Created sample dataset: {output_file}")
    print(f"  Samples: {len(df)}")

    return df
