# Seq2Seq Model Architecture

## Overview
Sequence-to-Sequence model with Bahdanau Attention for Arabic-to-English transliteration.

## Key Components
1. **Bidirectional LSTM Encoder** - Processes Arabic input
2. **Bahdanau Attention** - Learns character alignment  
3. **LSTM Decoder** - Generates English output character-by-character
4. **Teacher Forcing** - Training strategy for faster convergence

## Model Parameters
- Total parameters: ~7.8M
- Encoder: ~2.5M (embedding + BiLSTM)
- Decoder: ~5.3M (embedding + attention + LSTM + output)

## Performance
- Expected accuracy: 84-90% (based on similar systems)
- Character-level processing handles unseen names
- Attention mechanism learns complex character mappings
