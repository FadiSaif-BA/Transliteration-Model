"""
Word-Level Training Script for Transliteration Model

This script trains the transliteration model on word-level data,
which should improve exact match accuracy on test sequences.

Key improvements:
1. Word-level training (shorter sequences = higher exact match)
2. Uses smart word alignment for Arabic-English mapping
3. Includes all previous fixes (ExactMatchCallback, ScheduledSampling)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from src.features.character_encoder import ArabicCharEncoder, EnglishCharEncoder, EncoderPair
from src.features.word_splitter import ArabicWordSplitter, create_word_level_dataset, validate_alignment
from src.models.seq2seq_model import (
    build_model,
    ExactMatchCallback,
    ScheduledSamplingCallback
)


def prepare_word_level_data(train_path: str, val_path: str, test_path: str):
    """
    Convert name-level data to word-level data.
    """
    print("\n" + "="*60)
    print("PREPARING WORD-LEVEL DATA")
    print("="*60)

    # Load original data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Original data sizes:")
    print(f"  Train: {len(train_df)} names")
    print(f"  Val:   {len(val_df)} names")
    print(f"  Test:  {len(test_df)} names")

    # Process each dataset
    train_words = create_word_level_dataset(train_df, 'arabic_name', 'english_name')
    val_words = create_word_level_dataset(val_df, 'arabic_name', 'english_name')
    test_words = create_word_level_dataset(test_df, 'arabic_name', 'english_name')

    print(f"\nWord-level data sizes:")
    print(f"  Train: {len(train_words)} word pairs")
    print(f"  Val:   {len(val_words)} word pairs")
    print(f"  Test:  {len(test_words)} word pairs")

    # Validate alignment on sample
    print("\n--- Sample Alignments (Validation) ---")
    validate_alignment(val_words, sample_size=10)

    # Remove duplicates to get unique word pairs
    train_unique = train_words.drop_duplicates(subset=['arabic_word', 'english_word'])
    val_unique = val_words.drop_duplicates(subset=['arabic_word', 'english_word'])
    test_unique = test_words.drop_duplicates(subset=['arabic_word', 'english_word'])

    print(f"\nUnique word pairs:")
    print(f"  Train: {len(train_unique)} unique")
    print(f"  Val:   {len(val_unique)} unique")
    print(f"  Test:  {len(test_unique)} unique")

    return train_unique, val_unique, test_unique


def build_vocabularies(train_words: pd.DataFrame):
    """Build character vocabularies from training data."""
    print("\n" + "="*60)
    print("BUILDING VOCABULARIES")
    print("="*60)
    
    # Collect all unique characters from training data
    arabic_texts = train_words['arabic_word'].tolist()
    english_texts = train_words['english_word'].tolist()
    
    # Get unique characters
    arabic_chars = set()
    for text in arabic_texts:
        arabic_chars.update(text)
    
    english_chars = set()
    for text in english_texts:
        english_chars.update(text)
    
    # Build Arabic vocabulary
    arabic_encoder = ArabicCharEncoder()
    arabic_encoder.build_vocab(list(arabic_chars))
    
    # Build English vocabulary  
    english_encoder = EnglishCharEncoder()
    english_encoder.build_vocab(list(english_chars))
    
    encoder_pair = EncoderPair(arabic_encoder, english_encoder)
    
    print(f"Arabic vocabulary: {arabic_encoder.vocab_size} characters")
    print(f"English vocabulary: {english_encoder.vocab_size} characters")
    
    return encoder_pair


def encode_sequences(encoder, texts, max_length, add_start=False, add_end=False):
    """Encode a batch of texts to padded sequences."""
    sequences = []
    for text in texts:
        indices = encoder.encode(text, add_start=add_start, add_end=add_end)
        padded = encoder.pad_sequence(indices, max_length)
        sequences.append(padded)
    return np.array(sequences, dtype='int32')


def prepare_training_data(words_df: pd.DataFrame, encoder_pair: EncoderPair, 
                          max_input_len: int = 20, max_output_len: int = 25):
    """Prepare encoded sequences for training."""
    
    arabic_encoder = encoder_pair.arabic_encoder
    english_encoder = encoder_pair.english_encoder
    
    arabic_texts = words_df['arabic_word'].tolist()
    english_texts = words_df['english_word'].tolist()
    
    # Encode Arabic (input)
    encoder_inputs = encode_sequences(arabic_encoder, arabic_texts, max_input_len)
    
    # Encode English with START token (decoder input)
    decoder_inputs = encode_sequences(english_encoder, english_texts, max_output_len, 
                                       add_start=True, add_end=False)
    
    # Encode English with END token (decoder target)
    decoder_targets = encode_sequences(english_encoder, english_texts, max_output_len,
                                        add_start=False, add_end=True)
    
    return encoder_inputs, decoder_inputs, decoder_targets


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("WORD-LEVEL TRANSLITERATION MODEL TRAINING")
    print("="*60)
    
    # Paths
    data_dir = Path("data/processed")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # 1. Prepare word-level data
    train_words, val_words, test_words = prepare_word_level_data(
        data_dir / "train.csv",
        data_dir / "val.csv", 
        data_dir / "test.csv"
    )
    
    # 2. Build vocabularies
    encoder_pair = build_vocabularies(train_words)
    
    # 3. Prepare training data
    max_input_len = 20  # Max Arabic word length
    max_output_len = 25  # Max English word length
    
    print("\n" + "="*60)
    print("PREPARING TRAINING SEQUENCES")
    print("="*60)
    
    train_enc, train_dec, train_tgt = prepare_training_data(
        train_words, encoder_pair, max_input_len, max_output_len
    )
    val_enc, val_dec, val_tgt = prepare_training_data(
        val_words, encoder_pair, max_input_len, max_output_len
    )
    
    print(f"Training shapes: input={train_enc.shape}, dec_input={train_dec.shape}, target={train_tgt.shape}")
    print(f"Validation shapes: input={val_enc.shape}, dec_input={val_dec.shape}, target={val_tgt.shape}")
    
    # 4. Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    model_config = {
        'embedding_dim': 128,
        'hidden_units': 256,
        'attention_units': 64,
        'dropout': 0.2,
        'training': {
            'learning_rate': 0.001
        }
    }
    
    # model = build_model(
    #     arabic_vocab_size=encoder_pair.arabic_encoder.vocab_size,
    #     english_vocab_size=encoder_pair.english_encoder.vocab_size,
    #     config=model_config
    # )
    #
    # # Build model with sample input
    # sample_enc = np.zeros((1, max_input_len), dtype='int32')
    # sample_dec = np.zeros((1, max_output_len), dtype='int32')
    # _ = model([sample_enc, sample_dec])
    #
    # print(f"Model parameters: {model.count_params():,}")
    #
    # # 5. Setup callbacks
    # print("\n" + "="*60)
    # print("SETTING UP TRAINING")
    # print("="*60)
    #
    # # Create validation dataframe for ExactMatchCallback
    # # The callback expects 'arabic_normalized' and 'english_cleaned' columns
    # val_df_for_callback = val_words.rename(columns={
    #     'arabic_word': 'arabic_normalized',
    #     'english_word': 'english_cleaned'
    # })
    #
    # callbacks = [
    #     # Early stopping
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=10,
    #         restore_best_weights=True,
    #         verbose=1
    #     ),
    #     # Learning rate reduction
    #     tf.keras.callbacks.ReduceLROnPlateau(
    #         monitor='val_loss',
    #         factor=0.5,
    #         patience=5,
    #         min_lr=1e-6,
    #         verbose=1
    #     ),
    #     # Exact match monitoring (smaller sample for word-level)
    #     ExactMatchCallback(
    #         val_df=val_df_for_callback,
    #         encoder_pair=encoder_pair,
    #         sample_size=50,  # Use more samples for word-level (faster)
    #         max_length=max_output_len
    #     ),
    #     # Scheduled sampling to bridge teacher forcing gap
    #     ScheduledSamplingCallback(
    #         initial_prob=0.0,
    #         final_prob=0.4,
    #         warmup_epochs=10
    #     )
    # ]
    #
    # # 6. Train
    # print("\n" + "="*60)
    # print("TRAINING")
    # print("="*60)
    #
    # batch_size = 64
    # epochs = 80
    #
    # history = model.fit(
    #     [train_enc, train_dec],
    #     train_tgt,
    #     validation_data=([val_enc, val_dec], val_tgt),
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     callbacks=callbacks,
    #     verbose=1
    # )
    #
    # # 7. Save model and encoders
    # print("\n" + "="*60)
    # print("SAVING MODEL")
    # print("="*60)
    #
    # model.save_weights(str(model_dir / "word_model.weights.h5"))
    # encoder_pair.save(str(model_dir / "word_encoders"))
    #
    # # Save model config
    # import json
    # config_path = model_dir / "word_model_config.json"
    # full_config = {
    #     **model_config,
    #     'arabic_vocab_size': encoder_pair.arabic_encoder.vocab_size,
    #     'english_vocab_size': encoder_pair.english_encoder.vocab_size,
    #     'max_input_len': max_input_len,
    #     'max_output_len': max_output_len
    # }
    # with open(config_path, 'w') as f:
    #     json.dump(full_config, f, indent=2)
    #
    # print(f"✓ Model weights saved to: {model_dir / 'word_model.weights.h5'}")
    # print(f"✓ Encoders saved to: {model_dir / 'word_encoders'}")
    # print(f"✓ Config saved to: {config_path}")
    #
    # # 8. Print training summary
    # print("\n" + "="*60)
    # print("TRAINING COMPLETE")
    # print("="*60)
    #
    # final_loss = history.history['loss'][-1]
    # final_val_loss = history.history['val_loss'][-1]
    # final_acc = history.history.get('accuracy', [0])[-1]
    #
    # print(f"Final training loss: {final_loss:.4f}")
    # print(f"Final validation loss: {final_val_loss:.4f}")
    # print(f"Final training accuracy: {final_acc*100:.1f}%")
    #
    # print("\nTo evaluate: python evaluate_word_model.py")


if __name__ == '__main__':
    main()
