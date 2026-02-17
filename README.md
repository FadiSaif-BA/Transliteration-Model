# Arabic to English Transliteration System (TransliterateModel)

A robust, production-grade neural machine transliteration system designed to convert Arabic names (specifically Yemeni administrative areas) into standardized English text following UNGEGN guidelines.

This project combines a **Rule-Based Engine** for deterministic cases with a **Sequence-to-Sequence (Seq2Seq) Neural Network** for handling ambiguity, ensuring high accuracy and consistency.

## 🚀 Key Features

- **Hybrid Architecture**: 
  - **Rule Engine**: Handles deterministic patterns (e.g., *Al-* prefix, *Taa Marbouta* → *ah*) with 100% accuracy.
  - **Neural Model**: Seq2Seq LSTM with Bahdanau Attention for learning complex phonetic mappings and vowel restoration.
- **UNGEGN Compliance**: Produces simplified English output (e.g., *ā, ī, ū* for long vowels, no complex diacritics on consonants) suitable for official use.
- **Optimized Performance**: 
  - **Vectorized Validation**: Fast GPU-accelerated batch evaluation using TensorFlow.
  - **Beam Search**: High-quality decoding for difficult names.
  - **Caching**: LRU caching for repeatedly accessed tokens.
- **Trainable**: easy-to-use training pipeline for custom datasets.

---

## 📂 Project Structure

```
TransliterateModel/
├── 📁 configs/                 # Configuration files
│   ├── transliteration_rules.yaml  # Linguistic rules & mappings
│   └── model_config.yaml           # Neural network hyperparameters
│
├── 📁 data/                    # Data directory
│   ├── raw/                        # Original input datasets
│   └── processed/                  # Cleaned & splitted data (train/val/test)
│
├── 📁 models/                  # Saved models & checkpoints
│   └── new_model/                  # Current active model artifacts
│
├── 📁 src/                     # Source code
│   ├── features/                   # Data encoding & processing
│   │   ├── character_encoder.py    # Char <-> Index conversion
│   │   └── word_splitter.py        # Word alignment utilities
│   ├── models/                     # Model definitions
│   │   └── seq2seq_model.py        # LSTM + Attention architecture
│   ├── preprocessing/              # Text cleaning & rules
│   │   ├── arabic_normalizer.py    # Normalization (Unicodes, Hamzas)
│   │   └── rule_engine.py          # Deterministic transliteration logic
│   └── utils/                      # Helper scripts
│
├── 📜 test_model.py            # Main inference & validation script (Fast!)
├── 📜 train_word_model.py      # Training script
└── 📜 requirements.txt         # Dependencies
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd TransliterateModel
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage

### 1. Validation & Testing

To evaluate the model on the test dataset or run interactive checking:

```bash
python test_model.py
```

- **Metrics**: Calculates Exact Match, Case-Insensitive Match, Word Accuracy, and Edit Distance.
- **Modes**:
  - **Fast Validation**: Uses vectorized TensorFlow operations for speed.
  - **Interactive Mode**: Type any Arabic word to see the result (Greedy & Beam Search).

### 2. Training a New Model

1. **Prepare Data**: Place your CSV files in `data/processed/` (`train.csv`, `val.csv`, `test.csv`).
   - Format: `arabic_name`, `english_name` columns.
2. **Run Training**:
   ```bash
   python train_word_model.py
   ```
   - This will preprocess data, build vocabularies, and train the Seq2Seq model.
   - Artifacts are saved to `models/`.

---

## 🧠 Model Details

### Architecture
- **Embedding Layer**: Learnable character embeddings (dim=128).
- **Encoder**: Bidirectional LSTM (256 units) to capture context from both sides.
- **Attention**: Bahdanau Attention to align Arabic characters with English output positions.
- **Decoder**: LSTM (256 units) with attention context concatenation.

### Optimization
The validation pipeline has been highly optimized:
- **Batch Processing**: Instead of looping one-by-one, we process thousands of words in parallel tensors.
- **Vectorized Metrics**: Levenshtein distance and character accuracy are computed using matrix operations (`tf.edit_distance`, `numpy`).

---

## 📊 Performance (Typical)

On a test set of ~12k Yemeni place names:
- **Exact Match**: ~17-20% (Strict)
- **Functional Accuracy**: ~55-60% (Acceptable for search/matching)
- **Character Accuracy**: >90%

*Note: Performance depends heavily on the training data quality and size.*

---

## 👥 Authors

**Fadi Ali Qasem Saif**  
*Business and Data Analytics Specialist*

Developed for Yemeni administrative area transliteration standardization.
