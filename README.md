# Arabic to English Transliteration System
## Yemeni Administrative Areas - UN Standard

A professional-grade hybrid transliteration system combining rule-based linguistics with deep learning for accurate Arabic-to-English transliteration of Yemeni geographic names.

---

## 🎯 Project Goals

- Transliterate Yemeni administrative areas (villages, sub-districts) to English
- Follow UN UNGEGN transliteration standards
- Avoid diacritics on English letters (simplified system)
- Combine linguistic rules with machine learning for accuracy
- Handle missing Arabic vowel marks (diacritics) intelligently

---

## 🔬 How It Works

### Hybrid Architecture

```
Input: حضرموت (Hadramawt in Arabic, no vowels marked)
    ↓
┌─────────────────────────────┐
│   RULE ENGINE               │
│  - Detect ة → ah            │
│  - Detect ا/و/ي patterns    │
│  - Apply definite article   │
│  - Map consonants           │
└─────────────────────────────┘
    ↓ (Handles ~60-70%)
Partially transliterated: "H_d_r_m_wt"
(underscores = unknown vowels)
    ↓
┌─────────────────────────────┐
│   ML MODEL (Seq2Seq)        │
│  - Bidirectional LSTM       │
│  - Attention mechanism      │
│  - Learns vowel patterns    │
└─────────────────────────────┘
    ↓ (Predicts remaining ~30-40%)
Output: "Hadramawt"
```

### Rule-Based Logic

1. **Taa marbouta (ة)**: Always "ah" - 100% deterministic
2. **Long vowels**: 
   - If see ا after consonant → that consonant has 'a', alif is 'ā'
   - If see و in middle/end + preceded by consonant → 'ū'
   - If see ي in middle/end + preceded by consonant → 'ī'
3. **Definite article**: ال at start → "al-" (with sun letter check)

### Machine Learning Component

**When rules aren't enough:**
- Consonants without following vowel letters
- Ambiguous و/ي (consonant vs. vowel)
- Context-dependent vowel choice (a vs. i vs. u)

**Model learns from training data:**
- Common morphological patterns
- Yemeni dialectal preferences
- N-gram context (surrounding letters)



---

**Status**: 🟡 In Development  
**Completed**: Configuration, Normalization, Rule Engine, Character Encoders  
