"""
Sequence-to-Sequence model with attention for Arabic transliteration.
Implements bidirectional LSTM encoder and LSTM decoder with Bahdanau attention.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict
from ..utils.config import get_config


class BahdanauAttention(layers.Layer):
    """
    Bahdanau (additive) attention mechanism.
    Computes attention weights between decoder state and encoder outputs.
    """

    def __init__(self, units, **kwargs):
        """
        Initialize attention layer.

        Args:
            units: Dimensionality of attention layer
        """
        super().__init__(**kwargs)  # Changed this line
        self.units = units

        # Dense layers for attention computation
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values, training=None):
        """
        Compute attention.

        Args:
            query: Decoder hidden state (batch_size, hidden_size)
            values: Encoder outputs (batch_size, max_length, hidden_size)
            training: Whether in training mode (unused but required by Keras)

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # If query is 1D, expand it to 2D
        if len(query.shape) == 1:
            query = tf.expand_dims(query, 0)

        # Expand query to match values dimensions
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape: (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        )

        # attention_weights shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape: (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({"units": self.units})
        return config


class Encoder(keras.Model):
    """
    Bidirectional LSTM encoder for Arabic text.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_size: int, num_layers: int = 2, 
                 dropout: float = 0.3, **kwargs):
        """
        Initialize encoder.

        Args:
            vocab_size: Size of input vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(Encoder, self).__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = layers.Embedding(
            vocab_size, 
            embedding_dim,
            mask_zero=True  # Mask padding tokens
        )

        # Bidirectional LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            # Return sequences for all but last layer
            return_sequences = (i < num_layers - 1) or True  # Always return sequences for attention

            lstm = layers.Bidirectional(
                layers.LSTM(
                    hidden_size,
                    return_sequences=return_sequences,
                    return_state=True,
                    dropout=dropout,
                    recurrent_dropout=0.2
                )
            )
            self.lstm_layers.append(lstm)

    def call(self, x, training=False):
        """
        Forward pass through encoder.

        Args:
            x: Input sequences (batch_size, max_length)
            training: Whether in training mode

        Returns:
            Tuple of (encoder_outputs, final_states)
        """
        # Embedding
        x = self.embedding(x)

        # Pass through LSTM layers
        states = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            if i == 0:
                lstm_output = lstm_layer(x, training=training)
            else:
                lstm_output = lstm_layer(lstm_output[0], training=training)

            # lstm_output: (outputs, forward_h, forward_c, backward_h, backward_c)
            outputs = lstm_output[0]
            forward_h, forward_c = lstm_output[1], lstm_output[2]
            backward_h, backward_c = lstm_output[3], lstm_output[4]

            # Concatenate forward and backward states
            state_h = layers.Concatenate()([forward_h, backward_h])
            state_c = layers.Concatenate()([forward_c, backward_c])

            states.append((state_h, state_c))

        return outputs, states


class Decoder(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int = 2,
                 dropout: float = 0.3, attention_units: int = 128, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.attention = BahdanauAttention(attention_units)

        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.LSTM(hidden_size, return_sequences=True,
                               return_state=True, dropout=dropout, recurrent_dropout=0.2)
            self.lstm_layers.append(lstm)

        self.concat = layers.Concatenate()
        self.fc = layers.Dense(vocab_size)

    def call(self, x, encoder_outputs, states=None, training=False):
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, emb_dim)
        batch_size = tf.shape(x)[0]

        # ✅ FIX: Initialize states properly
        if states is None:
            # Create zero states for all layers
            current_states = []
            for _ in range(self.num_layers):
                h = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
                c = tf.zeros((batch_size, self.hidden_size), dtype=tf.float32)
                current_states.append((h, c))
        elif isinstance(states, tuple) and len(states) == 2:
            # Single tuple (h, c) - replicate for all layers
            current_states = [states] * self.num_layers
        elif isinstance(states, list):
            # Already a list of tuples
            current_states = states
        else:
            # Fallback: replicate whatever we got
            current_states = [states] * self.num_layers

        # Rest of the code stays the same...
        x_timesteps = tf.unstack(x, axis=1)
        outputs = []

        for x_t in x_timesteps:
            x_t = tf.expand_dims(x_t, 1)
            query = current_states[0][0]
            context_vector, _ = self.attention(query, encoder_outputs)
            context_vector = tf.expand_dims(context_vector, 1)
            lstm_input = self.concat([x_t, context_vector])

            for i, lstm_layer in enumerate(self.lstm_layers):
                lstm_output, h, c = lstm_layer(
                    lstm_input,
                    initial_state=current_states[i],  # Now safely within range
                    training=training
                )
                current_states[i] = (h, c)
                lstm_input = lstm_output

            outputs.append(lstm_output)

        output = tf.concat(outputs, axis=1)
        return self.fc(output), current_states, None


class Seq2SeqTransliterator(keras.Model):
    """
    Complete Seq2Seq model with attention for Arabic-English transliteration.
    """

    def __init__(self, arabic_vocab_size: int, english_vocab_size: int,
                 config: Optional[Dict] = None, **kwargs):
        """
        Initialize seq2seq model.

        Args:
            arabic_vocab_size: Size of Arabic vocabulary
            english_vocab_size: Size of English vocabulary
            config: Configuration dictionary (or use default from config file)
        """
        super(Seq2SeqTransliterator, self).__init__(**kwargs)

        # Load configuration
        if config is None:
            cfg = get_config()
            config = cfg.model

        # Extract hyperparameters
        embedding_cfg = config.get('embedding', {})
        encoder_cfg = config.get('encoder', {})
        decoder_cfg = config.get('decoder', {})
        attention_cfg = config.get('attention', {})

        arabic_emb_dim = embedding_cfg.get('arabic_dim', 64)
        english_emb_dim = embedding_cfg.get('english_dim', 64)
        encoder_hidden = encoder_cfg.get('hidden_size', 256)
        encoder_layers = encoder_cfg.get('num_layers', 2)
        encoder_dropout = encoder_cfg.get('dropout', 0.3)
        decoder_hidden = decoder_cfg.get('hidden_size', 256)
        decoder_layers = decoder_cfg.get('num_layers', 2)
        decoder_dropout = decoder_cfg.get('dropout', 0.3)
        attention_size = attention_cfg.get('attention_size', 128)

        # Initialize encoder and decoder
        self.encoder = Encoder(
            vocab_size=arabic_vocab_size,
            embedding_dim=arabic_emb_dim,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=encoder_dropout
        )

        # Decoder hidden size should be 2x encoder (for bidirectional)
        decoder_hidden_size = encoder_hidden * 2

        self.decoder = Decoder(
            vocab_size=english_vocab_size,
            embedding_dim=english_emb_dim,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            attention_units=attention_size
        )

    def call(self, inputs, training=False):
        """Forward pass through entire model."""
        encoder_input, decoder_input = inputs

        # Encode
        encoder_outputs, encoder_states = self.encoder(encoder_input, training=training)

        # ← FIX: Extract the last layer's state from the list
        # encoder_states is a list: [(h1, c1), (h2, c2), ...]
        # We want the last one for the decoder
        initial_state = encoder_states[-1]  # Get last layer's (h, c)

        # Decode - feed ENTIRE sequence at once
        predictions, _, _ = self.decoder(
            decoder_input,
            encoder_outputs,
            initial_state,  # ← Pass tuple (h, c), not list of tuples
            training=training
        )

        return predictions

    def predict_sequence(self, encoder_input, english_encoder,
                         max_length: int = 50) -> Tuple[str, np.ndarray]:
        """
        Generate transliteration for input sequence (inference).
        """
        # Encode
        encoder_outputs, encoder_states = self.encoder(encoder_input, training=False)

        # ✅ FIX: Initialize decoder states properly
        # Use only the last encoder layer state, replicated for all decoder layers
        initial_state = encoder_states[-1]  # (h, c) tuple
        decoder_states = [initial_state] * self.decoder.num_layers

        # Start with START token
        current_token = english_encoder.get_start_idx()

        result = []
        attention_weights_list = []

        for step in range(max_length):
            # ✅ FIX: Create decoder input for single timestep
            decoder_input = tf.constant([[current_token]])  # Shape: (1, 1)

            # ✅ FIX: Call decoder for single timestep
            predictions, decoder_states, _ = self.decoder(
                decoder_input,
                encoder_outputs,
                decoder_states,
                training=False
            )

            # Get predicted token from last timestep
            predicted_id = tf.argmax(predictions[0, -1]).numpy()

            # Stop if END token
            if predicted_id == english_encoder.get_end_idx():
                break

            # Store result
            result.append(predicted_id)

            # ✅ FIX: Use predicted token as next input
            current_token = predicted_id

        # Decode result
        transliteration = english_encoder.decode(result)

        return transliteration, np.array([])  # Return empty attention for now

    def predict_sequence_greedy(self, encoder_input, english_encoder, max_length=50):
        """Greedy decoding without proper autoregression (for testing)."""
        # Encode
        encoder_outputs, encoder_states = self.encoder(encoder_input, training=False)

        # Create full START sequence
        decoder_input = tf.constant([[english_encoder.get_start_idx()] * max_length])

        # Get predictions for full sequence
        predictions, _, _ = self.decoder(
            decoder_input,
            encoder_outputs,
            encoder_states[-1],
            training=False
        )

        # Take argmax for each position
        predicted_ids = tf.argmax(predictions[0], axis=-1).numpy()

        # Find END token and truncate
        result = []
        for token_id in predicted_ids:
            if token_id == english_encoder.get_end_idx():
                break
            result.append(token_id)

        return english_encoder.decode(result), np.array([])

    def predict_sequence_beam(self, encoder_input, english_encoder, 
                              max_length: int = 50, beam_width: int = 3) -> Tuple[str, np.ndarray]:
        """
        Generate transliteration using beam search decoding.
        
        Explores multiple hypotheses at each step instead of just taking
        the single best prediction (greedy). This often produces better results.
        
        Args:
            encoder_input: Encoded Arabic input
            english_encoder: English character encoder
            max_length: Maximum output length
            beam_width: Number of hypotheses to track
            
        Returns:
            Best transliteration and attention weights
        """
        import heapq
        
        # Encode input
        encoder_outputs, encoder_states = self.encoder(encoder_input, training=False)
        initial_state = encoder_states[-1]
        
        start_token = english_encoder.get_start_idx()
        end_token = english_encoder.get_end_idx()
        
        # ✅ CRITICAL FIX: Create INDEPENDENT copies for the initial beam
        # Old code: [initial_state] * self.decoder.num_layers (BAD - Shared Reference)
        initial_states = [
            (tf.identity(initial_state[0]), tf.identity(initial_state[1]))
            for _ in range(self.decoder.num_layers)
        ]
        
        # Each beam: (neg_log_prob, sequence, decoder_states)
        initial_beam = (0.0, [start_token], initial_states)
        beams = [initial_beam]
        completed = []
        
        for step in range(max_length):
            all_candidates = []
            
            for score, sequence, states in beams:
                if sequence[-1] == end_token:
                    completed.append((score, sequence, states))
                    continue
                
                # Prepare decoder input for last token
                current_token = sequence[-1]
                decoder_input = tf.constant([[current_token]])
                
                # Get predictions
                # Note: The decoder call must basically be stateless here
                predictions, new_states, _ = self.decoder(
                    decoder_input,
                    encoder_outputs,
                    states,
                    training=False
                )
                
                # Get log probabilities (use log_softmax for numerical stability)
                # Shape: (1, vocab_size)
                logits = predictions[0, -1]
                log_probs = tf.nn.log_softmax(logits).numpy()
                
                # Get top-k tokens
                # We want indices with highest probability (least negative log prob)
                top_k_indices = np.argsort(log_probs)[-beam_width:]
                
                for token_idx in top_k_indices:
                    # We are MINIMIZING score (negative log likelihood)
                    # score is current accumulated cost
                    # log_probs[token_idx] is negative number (e.g. -0.5)
                    # We subtract it: score - (-0.5) = score + 0.5 (Cost increases)
                    new_score = score - log_probs[token_idx]
                    
                    new_sequence = sequence + [int(token_idx)]
                    all_candidates.append((new_score, new_sequence, new_states))
            
            if not all_candidates:
                break
                
            # Select top-k beams with SMALLEST score (highest probability)
            beams = heapq.nsmallest(beam_width, all_candidates, key=lambda x: x[0])
            
            # Stop if we have enough completed beams
            if len(completed) >= beam_width:
                break
        
        # Add any remaining active beams to completed list
        completed.extend(beams)
        
        if not completed:
            return "", np.array([])
        
        # Get best sequence (lowest negative log prob)
        best = min(completed, key=lambda x: x[0])
        best_sequence = best[1]
        
        # Remove special tokens and decode
        result = [t for t in best_sequence if t != start_token and t != end_token]
        transliteration = english_encoder.decode(result)
        
        return transliteration, np.array([])


def build_model(arabic_vocab_size: int, english_vocab_size: int,
                config: Optional[Dict] = None) -> Seq2SeqTransliterator:
    """
    Build and compile seq2seq model.

    Args:
        arabic_vocab_size: Size of Arabic vocabulary
        english_vocab_size: Size of English vocabulary
        config: Configuration dictionary

    Returns:
        Compiled model
    """
    model = Seq2SeqTransliterator(
        arabic_vocab_size=arabic_vocab_size,
        english_vocab_size=english_vocab_size,
        config=config
    )

    # Get training config
    if config is None:
        cfg = get_config()
        config = cfg.model

    training_cfg = config.get('training', {})
    learning_rate = training_cfg.get('learning_rate', 0.001)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


class ExactMatchCallback(keras.callbacks.Callback):
    """
    Compute exact sequence match accuracy during training.
    
    This gives a realistic view of model performance by measuring
    how many complete sequences are predicted correctly, not just
    individual characters.
    """
    
    def __init__(self, val_df, encoder_pair, sample_size=100, max_length=50):
        """
        Initialize callback.
        
        Args:
            val_df: Validation DataFrame with 'arabic_normalized' and 'english_cleaned'
            encoder_pair: EncoderPair with arabic and english encoders
            sample_size: Number of samples to evaluate each epoch
            max_length: Maximum sequence length for encoding
        """
        super().__init__()
        self.val_df = val_df
        self.encoder_pair = encoder_pair
        self.sample_size = min(sample_size, len(val_df))
        self.max_length = max_length
        self.history = {'exact_match': [], 'case_insensitive_match': []}
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute exact match accuracy at end of each epoch."""
        logs = logs or {}
        
        exact_matches = 0
        case_insensitive_matches = 0
        
        # Sample random indices
        indices = np.random.choice(len(self.val_df), self.sample_size, replace=False)
        
        for idx in indices:
            row = self.val_df.iloc[idx]
            arabic = row['arabic_normalized']
            true_english = row['english_cleaned']
            
            # Encode input
            encoder_input = self.encoder_pair.arabic_encoder.encode(arabic)
            encoder_input = self.encoder_pair.arabic_encoder.pad_sequence(
                encoder_input, self.max_length
            )
            encoder_input = np.array([encoder_input])
            
            # Predict
            try:
                pred_english, _ = self.model.predict_sequence(
                    encoder_input,
                    self.encoder_pair.english_encoder,
                    max_length=self.max_length
                )
                
                # Check matches
                if pred_english.strip() == true_english.strip():
                    exact_matches += 1
                    case_insensitive_matches += 1
                elif pred_english.strip().lower() == true_english.strip().lower():
                    case_insensitive_matches += 1
            except Exception:
                pass  # Skip failed predictions
        
        exact_acc = exact_matches / self.sample_size * 100
        case_acc = case_insensitive_matches / self.sample_size * 100
        
        self.history['exact_match'].append(exact_acc)
        self.history['case_insensitive_match'].append(case_acc)
        
        logs['exact_match_acc'] = exact_acc
        logs['case_insensitive_acc'] = case_acc
        
        print(f" - exact_match: {exact_acc:.1f}% - case_insensitive: {case_acc:.1f}%")


class ScheduledSamplingCallback(keras.callbacks.Callback):
    """
    Implement scheduled sampling to bridge the teacher forcing gap.
    
    During training, gradually increases the probability of using 
    the model's own predictions instead of ground truth tokens.
    This helps the model learn to recover from its own mistakes.
    
    Note: This callback modifies training behavior. The model must be
    trained with a custom training loop or must support sampling_probability.
    """
    
    def __init__(self, initial_prob: float = 0.0, final_prob: float = 0.5,
                 warmup_epochs: int = 5, total_epochs: int = 50):
        """
        Initialize scheduled sampling.
        
        Args:
            initial_prob: Starting probability of using model predictions (0 = pure teacher forcing)
            final_prob: Final probability of using model predictions
            warmup_epochs: Epochs before sampling begins
            total_epochs: Total training epochs for scheduling
        """
        super().__init__()
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_prob = initial_prob
    
    def on_epoch_begin(self, epoch, logs=None):
        """Update sampling probability at start of each epoch."""
        if epoch < self.warmup_epochs:
            self.current_prob = self.initial_prob
        else:
            # Linear schedule from initial to final
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            self.current_prob = self.initial_prob + progress * (self.final_prob - self.initial_prob)
        
        # Store in model for use during training
        if hasattr(self.model, 'sampling_probability'):
            self.model.sampling_probability = self.current_prob
        
        print(f"Epoch {epoch + 1}: scheduled sampling probability = {self.current_prob:.2%}")
    
    def get_sampling_probability(self) -> float:
        """Get current sampling probability."""
        return self.current_prob

# class HybridTransliterator:
#     """Combines rule-based consonants with neural vowel prediction."""
#
#     def __init__(self, seq2seq_model, rule_engine, english_encoder):
#         self.seq2seq_model = seq2seq_model
#         self.rule_engine = rule_engine
#         self.english_encoder = english_encoder
#
#     def transliterate(self, arabic_text: str, arabic_encoder) -> str:
#         # Get rule-based skeleton (consonants)
#         rule_trans = self.rule_engine.apply_rules(arabic_text)
#
#         # Get neural prediction (full)
#         encoder_input = arabic_encoder.encode(arabic_text)
#         encoder_input = arabic_encoder.pad_sequence(encoder_input, 50)
#         encoder_input = tf.constant([encoder_input])
#         neural_trans, _ = self.seq2seq_model.predict_sequence(
#             encoder_input, self.english_encoder, max_length=50
#         )
#
#         # Merge: use rule consonants + neural vowels/spacing
#         return self._merge(rule_trans, neural_trans)
#
#     def _merge(self, rules, neural):
#         # Start with neural (better vowels/spacing)
#         # Then verify consonants against rules
#         # For Yemeni data, you might weight rules higher
#         return neural  # Placeholder - implement alignment algorithm



import numpy as np
import tensorflow as tf
from typing import Tuple


class ConsonantMapper:
    """Rule-based consonant skeleton extractor."""

    def __init__(self):
        # Consonant-only mappings (deterministic)
        self.consonant_map = {
            'ء': "'",
            'أ': 'a',
            'إ': 'i',
            'ا': 'a',
            'ب': 'b',
            'ت': 't',
            'ث': 'th',
            'ج': 'J',
            'ح': 'h',
            'خ': 'kh',
            'د': 'd',
            'ذ': 'dh',
            'ر': 'r',
            'ز': 'z',
            'س': 's',
            'ش': 'sh',
            'ص': 's',
            'ض': 'd',
            'ط': 'T',
            'ظ': 'dh',
            'ع': "'",
            'غ': 'gh',
            'ف': 'f',
            'ق': 'q',
            'ك': 'k',
            'ل': 'l',
            'م': 'm',
            'ن': 'n',
            'ه': 'h',
            'و': 'w',
            'ي': 'y',
            'ى': 'a',
            'ة': 'ah',
        }

        # Vowel markers (to skip)
        self.vowels = {'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ً', 'ٌ', 'ٍ'}

    def get_skeleton(self, arabic_text: str) -> Tuple[str, list]:
        """
        Extract consonant skeleton from Arabic text.

        Args:
            arabic_text: Arabic string

        Returns:
            Tuple of (skeleton_string, consonant_list)
            Example: "القلعه" -> ("ALQL'H", ['Al', 'Q', 'L', "'", 'H'])
        """
        consonants = []

        # Handle definite article ال
        text = arabic_text.strip()
        if text.startswith('ال'):
            consonants.append('Al')
            text = text[2:]

        # Extract consonants
        for char in text:
            if char in self.vowels:
                continue  # Skip vowel markers
            elif char == ' ':
                consonants.append(' ')
            elif char in self.consonant_map:
                consonants.append(self.consonant_map[char])

        skeleton_str = ''.join(consonants)
        return skeleton_str, consonants


class HybridTransliterator:
    """
    Combines rule-based consonant mapping with neural vowel insertion.
    """

    def __init__(self, vowel_model, vowel_encoder_pair):
        """
        Args:
            seq2seq_model: Trained Seq2Seq model
            encoder_pair: EncoderPair with Arabic and English encoders
            vowel_model: Model trained on skeleton→English task
            vowel_encoder_pair: Encoders for skeleton input
        """

        self.consonant_mapper = ConsonantMapper()
        self.vowel_model = vowel_model
        self.encoder_pair = vowel_encoder_pair

    def transliterate(self, arabic_text: str, max_length: int = 50) -> str:
        """
        Transliterate Arabic to English using hybrid approach.

        Args:
            arabic_text: Arabic input string
            max_length: Maximum output length

        Returns:
            English transliteration
        """
        # Stage 1: Get skeleton using rules (100% accurate)
        skeleton, _ = self.consonant_mapper.get_skeleton(arabic_text)

        # Stage 2: Neural model inserts vowels
        encoder_input = self.encoder_pair.arabic_encoder.encode(skeleton)
        encoder_input = self.encoder_pair.arabic_encoder.pad_sequence(encoder_input, max_length)
        encoder_input = np.array([encoder_input])

        result, _ = self.vowel_model.predict_sequence(
            encoder_input,
            self.encoder_pair.english_encoder,
            max_length=max_length
        )

        return result

    def _apply_skeleton_constraint(self, neural_output: str,
                                   consonants: list) -> str:
        """
        Force neural output to use correct consonants.

        Strategy: Keep neural output but fix consonants that don't match.
        """
        # Simple strategy: if neural output has wrong consonants,
        # rebuild using skeleton + best guess vowels

        # Extract consonants from neural output
        neural_consonants = self._extract_consonants(neural_output)

        # If skeletons match closely, use neural output
        if self._skeletons_similar(neural_consonants, consonants):
            return neural_output

        # Otherwise, rebuild from skeleton
        return self._rebuild_from_skeleton(consonants)

    def _extract_consonants(self, text: str) -> list:
        """Extract consonant letters from English text."""
        consonants = []
        skip_next = False

        for i, char in enumerate(text):
            if skip_next:
                skip_next = False
                continue

            if char in 'aeiouAEIOU-':
                continue  # Skip vowels and hyphens

            # Check for digraphs (Sh, Th, Kh, Gh, Dh)
            if i < len(text) - 1 and text[i:i + 2] in ['Sh', 'Th', 'Kh', 'Gh', 'Dh', 'sh', 'th', 'kh', 'gh', 'dh']:
                consonants.append(text[i:i + 2])
                skip_next = True
            else:
                consonants.append(char)

        return consonants

    def _skeletons_similar(self, neural_cons: list, rule_cons: list) -> bool:
        """Check if two consonant sequences are similar enough."""
        # Allow small differences (1-2 consonants different)
        if len(neural_cons) != len(rule_cons):
            return False

        differences = sum(1 for a, b in zip(neural_cons, rule_cons) if a.upper() != b.upper())
        return differences <= 2  # Allow up to 2 consonant errors

    def _rebuild_from_skeleton(self, consonants: list) -> str:
        """
        Rebuild transliteration from consonant skeleton.
        Uses simple heuristics for vowel placement.
        """
        result = []

        for i, cons in enumerate(consonants):
            if cons == ' ':
                result.append(' ')
                continue

            # Add consonant
            result.append(cons)

            # Add vowel after consonant (simple heuristic)
            if i < len(consonants) - 1 and consonants[i + 1] != ' ':
                # Default to 'a' between consonants
                result.append('a')

        output = ''.join(result)

        # Clean up
        output = output.replace('Ala', 'Al-')  # Fix definite article
        output = output.rstrip('a')  # Remove trailing 'a'

        return output
