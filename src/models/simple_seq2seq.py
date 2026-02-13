"""
Simple working seq2seq model for transliteration.
Uses Keras Functional API for clean train/inference compatibility.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_simple_seq2seq(arabic_vocab_size: int,
                         english_vocab_size: int,
                         embedding_dim: int = 32,
                         hidden_dim: int = 128):
    """
    Build simple seq2seq model that works for both training and inference.

    Args:
        arabic_vocab_size: Size of Arabic vocabulary
        english_vocab_size: Size of English vocabulary
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension

    Returns:
        Training model
    """
    # Encoder
    encoder_input = layers.Input(shape=(None,), name='encoder_input')
    encoder_embedding = layers.Embedding(
        arabic_vocab_size,
        embedding_dim,
        mask_zero=True
    )(encoder_input)

    encoder_lstm = layers.LSTM(
        hidden_dim,
        return_state=True,
        dropout=0.2,
        name='encoder_lstm'
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_input = layers.Input(shape=(None,), name='decoder_input')
    decoder_embedding = layers.Embedding(
        english_vocab_size,
        embedding_dim,
        mask_zero=True
    )(decoder_input)

    decoder_lstm = layers.LSTM(
        hidden_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=encoder_states
    )

    decoder_dense = layers.Dense(
        english_vocab_size,
        name='decoder_output'
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    # Training model
    model = keras.Model([encoder_input, decoder_input], decoder_outputs)

    return model


def build_inference_models(training_model):
    """
    Build separate encoder and decoder models for inference.

    Args:
        training_model: Trained seq2seq model

    Returns:
        Tuple of (encoder_model, decoder_model)
    """
    # Get layers from training model
    encoder_input = training_model.get_layer('encoder_input').output
    encoder_embedding = training_model.get_layer('embedding')(encoder_input)
    encoder_lstm = training_model.get_layer('encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_embedding)

    # Encoder inference model
    encoder_model = keras.Model(
        encoder_input,
        [state_h, state_c]
    )

    # Decoder inference model
    decoder_state_input_h = layers.Input(shape=(128,))
    decoder_state_input_c = layers.Input(shape=(128,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_input = training_model.get_layer('decoder_input').output
    decoder_embedding_layer = training_model.get_layer('embedding_1')
    decoder_embedding = decoder_embedding_layer(decoder_input)

    decoder_lstm = training_model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding,
        initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]

    decoder_dense = training_model.get_layer('decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = keras.Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def transliterate(arabic_text, encoder_pair, encoder_model, decoder_model, max_len=50):
    """
    Transliterate Arabic text to English.

    Args:
        arabic_text: Input Arabic text
        encoder_pair: EncoderPair instance
        encoder_model: Encoder inference model
        decoder_model: Decoder inference model
        max_len: Maximum output length

    Returns:
        Transliterated English text
    """
    # Encode input
    input_seq = encoder_pair.arabic_encoder.encode(arabic_text, max_len)
    if len(input_seq) < max_len:
        input_seq = input_seq + [0] * (max_len - len(input_seq))
    input_seq = tf.constant([input_seq], dtype=tf.int32)

    # Get encoder states
    states = encoder_model.predict(input_seq, verbose=0)

    # Generate output
    target_seq = tf.constant([[1]], dtype=tf.int32)  # <START>
    result = []

    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states,
            verbose=0
        )

        # Get next token
        next_token = int(tf.argmax(output_tokens[0, -1, :]))

        if next_token == 2:  # <END>
            break

        if next_token not in [0, 1, 2]:  # Not special tokens
            char = encoder_pair.english_encoder.idx_to_char.get(next_token, '')
            if char:
                result.append(char)

        # Update target sequence and states
        target_seq = tf.concat([target_seq, [[next_token]]], axis=1)
        states = [h, c]

    return ''.join(result)
