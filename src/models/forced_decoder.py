"""
Use teacher forcing during inference by providing skeleton as decoder input.
"""
import numpy as np
import tensorflow as tf


def predict_with_forced_decoding(model, encoder_pair, skeleton_text, max_length=50):
    """
    Use skeleton as both encoder input AND decoder input (forced decoding).
    """
    # Encode skeleton for encoder
    encoder_input = encoder_pair.arabic_encoder.encode(skeleton_text)
    encoder_input = encoder_pair.arabic_encoder.pad_sequence(encoder_input, max_length)
    encoder_input = np.array([encoder_input])

    # Also use skeleton for decoder (forced decoding approximation)
    # Prepend START token
    decoder_input_ids = [encoder_pair.english_encoder.get_start_idx()] + list(encoder_input[0])
    decoder_input = np.array([decoder_input_ids[:max_length]])

    # Get encoder outputs
    encoder_outputs, encoder_states = model.encoder(encoder_input, training=False)

    # Run decoder in one shot
    predictions, _, _ = model.decoder(
        decoder_input,
        encoder_outputs,
        encoder_states[-1],
        training=False
    )

    # Take argmax for each position
    predicted_ids = tf.argmax(predictions[0], axis=-1).numpy()

    # Decode
    result = []
    for token_id in predicted_ids:
        if token_id == encoder_pair.english_encoder.get_end_idx():
            break
        if token_id != encoder_pair.english_encoder.get_start_idx():
            result.append(int(token_id))

    return encoder_pair.english_encoder.decode(result)
