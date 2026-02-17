"""
Fast Validation Script for Kaggle-Trained Transliteration Model

Features:
- Batch greedy decoding (fast)
- tf.function decoding (faster)
- LRU cache for repeated Arabic tokens
- Optional beam search (slower, for quality checks)
"""

import os
import json
from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf
from difflib import SequenceMatcher

from src.features.character_encoder import EncoderPair
from src.models.seq2seq_model import build_model


# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class ModelValidator:
    """Validates transliteration model with comprehensive metrics."""

    def __init__(self, model, encoder_pair, max_input_len=20, max_output_len=25, cache_size=50000):
        self.model = model
        self.encoder_pair = encoder_pair
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        # LRU cache for repeated words
        self._cached_transliterate = lru_cache(maxsize=cache_size)(self._transliterate_word_uncached)

    def transliterate_word(self, arabic_word: str, use_beam=False, beam_width=3) -> str:
        """Public entry with cache.

        Handles multi-word inputs by splitting on whitespace and transliterating each word.
        """
        # Normalize whitespace
        arabic_word = " ".join(arabic_word.strip().split())
        if not arabic_word:
            return ""

        # If compound, handle per word
        if " " in arabic_word:
            parts = arabic_word.split(" ")
            return " ".join(self.transliterate_word(p, use_beam=use_beam, beam_width=beam_width) for p in parts)

        if use_beam:
            # Beam search is not cached (can be, but results vary with beam_width)
            return self._transliterate_word_uncached(arabic_word, use_beam=use_beam, beam_width=beam_width)
        return self._cached_transliterate(arabic_word)

    def _transliterate_word_uncached(self, arabic_word: str, use_beam=False, beam_width=3) -> str:
        arabic_encoder = self.encoder_pair.arabic_encoder
        english_encoder = self.encoder_pair.english_encoder

        indices = arabic_encoder.encode(arabic_word)
        encoder_input = np.array([arabic_encoder.pad_sequence(indices, self.max_input_len)], dtype="int32")

        start_idx = english_encoder.get_start_idx()
        end_idx = english_encoder.get_end_idx()

        if use_beam:
            return self._beam_search(encoder_input, start_idx, end_idx, beam_width)
        return self._greedy_decode(encoder_input, start_idx, end_idx)

    def _greedy_decode(self, encoder_input, start_idx, end_idx) -> str:
        english_encoder = self.encoder_pair.english_encoder
        decoder_input = np.zeros((1, self.max_output_len), dtype="int32")
        decoder_input[0, 0] = start_idx

        result = []
        for i in range(self.max_output_len - 1):
            predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
            next_token = int(np.argmax(predictions[0, i, :]))

            if next_token == end_idx:
                break

            result.append(next_token)
            if i + 1 < self.max_output_len:
                decoder_input[0, i + 1] = next_token

        return english_encoder.decode(result)

    def _beam_search(self, encoder_input, start_idx, end_idx, beam_width=3) -> str:
        english_encoder = self.encoder_pair.english_encoder
        beams = [([start_idx], 0.0)]

        for _ in range(self.max_output_len - 1):
            all_candidates = []

            for seq, score in beams:
                if seq[-1] == end_idx:
                    all_candidates.append((seq, score))
                    continue

                decoder_input = np.zeros((1, self.max_output_len), dtype="int32")
                for i, token in enumerate(seq):
                    if i < self.max_output_len:
                        decoder_input[0, i] = token

                predictions = self.model.predict([encoder_input, decoder_input], verbose=0)

                # Use log-softmax for numerical stability
                logits = predictions[0, len(seq) - 1, :]
                log_probs = tf.nn.log_softmax(logits).numpy()

                top_k = np.argsort(log_probs)[-beam_width:]

                for token in top_k:
                    new_seq = seq + [int(token)]
                    # length-normalized score to reduce repetition bias
                    new_score = (score + float(log_probs[token])) / max(1, len(new_seq))
                    all_candidates.append((new_seq, new_score))

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]

            if all(seq[-1] == end_idx for seq, _ in beams):
                break

        best_seq = beams[0][0]
        result = [t for t in best_seq if t not in [start_idx, end_idx]]
        return english_encoder.decode(result)

    def _greedy_decode_batch(self, encoder_input, start_idx, end_idx) -> list[str]:
        english_encoder = self.encoder_pair.english_encoder
        batch_size = encoder_input.shape[0]
        decoder_input = np.zeros((batch_size, self.max_output_len), dtype="int32")
        decoder_input[:, 0] = start_idx

        results = [[] for _ in range(batch_size)]
        finished = np.zeros((batch_size,), dtype=bool)

        for i in range(self.max_output_len - 1):
            predictions = self.model.predict([encoder_input, decoder_input], verbose=0)
            next_tokens = np.argmax(predictions[:, i, :], axis=-1)

            for b, token in enumerate(next_tokens):
                if finished[b]:
                    continue
                if token == end_idx:
                    finished[b] = True
                    continue
                results[b].append(int(token))
                if i + 1 < self.max_output_len:
                    decoder_input[b, i + 1] = token

            if finished.all():
                break

        return [english_encoder.decode(seq) for seq in results]

    @tf.function
    def _greedy_decode_batch_tf(self, encoder_input, start_idx, end_idx):
        batch_size = tf.shape(encoder_input)[0]
        decoder_input = tf.zeros((batch_size, self.max_output_len), dtype=tf.int32)

        start_positions = tf.stack([tf.range(batch_size), tf.zeros(batch_size, dtype=tf.int32)], axis=1)
        decoder_input = tf.tensor_scatter_nd_update(
            decoder_input,
            start_positions,
            tf.fill([batch_size], tf.cast(start_idx, tf.int32))
        )

        results = tf.TensorArray(tf.int32, size=self.max_output_len - 1)
        finished = tf.zeros((batch_size,), dtype=tf.bool)

        def cond(i, dec_in, done, res):
            return tf.logical_and(i < self.max_output_len - 1, tf.logical_not(tf.reduce_all(done)))

        def body(i, dec_in, done, res):
            preds = self.model([encoder_input, dec_in], training=False)
            next_tokens = tf.argmax(preds[:, i, :], axis=-1, output_type=tf.int32)

            finished_now = tf.equal(next_tokens, tf.cast(end_idx, tf.int32))
            done = tf.logical_or(done, finished_now)

            update_mask = tf.logical_not(done)
            indices = tf.where(~finished)  # shape: [n, 1]
            updates = tf.gather(next_tokens, indices[:, 0])
            scatter_indices = tf.concat(
                [
                    indices,
                    tf.cast(
                        tf.fill(
                            [tf.shape(indices)[0], 1],
                            tf.cast(i + 1, dtype=indices.dtype)
                        ),
                        dtype=indices.dtype
                    )
                ],
                axis=1
            )
            dec_in = tf.tensor_scatter_nd_update(dec_in, scatter_indices, updates)
            res = res.write(i, next_tokens)
            return i + 1, dec_in, done, res

        i0 = tf.constant(0)
        _, _, _, results = tf.while_loop(cond, body, [i0, decoder_input, finished, results])
        tokens = tf.transpose(results.stack(), [1, 0])  # [batch, steps]
        return tokens

    def calculate_similarity(self, pred: str, true: str) -> float:
        return SequenceMatcher(None, pred.lower(), true.lower()).ratio()

    def calculate_char_accuracy(self, pred: str, true: str) -> float:
        min_len = min(len(pred), len(true))
        max_len = max(len(pred), len(true))
        if max_len == 0:
            return 1.0
        matches = sum(1 for i in range(min_len) if pred[i].lower() == true[i].lower())
        return matches / max_len

    def categorize_error(self, pred: str, true: str) -> str:
        if pred == true:
            return "exact"
        if pred.lower() == true.lower():
            return "case_only"
        if pred.lower().rstrip("ah") == true.lower().rstrip("ah"):
            return "ending_h_a"

        similarity = self.calculate_similarity(pred, true)
        if similarity >= 0.9:
            return "near_miss"
        if similarity >= 0.7:
            return "minor_error"
        if similarity >= 0.5:
            return "moderate_error"
        return "major_error"

    def validate(
        self,
        test_df: pd.DataFrame,
        arabic_col="arabic_word",
        english_col="english_word",
        use_beam=False,
        sample_size=None,
        verbose=True,
        batch_size=64,
        use_tf=True,
    ) -> dict:
        if sample_size:
            test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

        results = []
        error_categories = Counter()
        total_similarity = 0.0
        total_char_acc = 0.0

        arabic_list = test_df[arabic_col].astype(str).tolist()
        true_list = test_df[english_col].astype(str).tolist()
        total = len(arabic_list)

        arabic_encoder = self.encoder_pair.arabic_encoder
        english_encoder = self.encoder_pair.english_encoder
        start_idx = english_encoder.get_start_idx()
        end_idx = english_encoder.get_end_idx()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if verbose:
                print(f"  Processing {end}/{total}...")

            batch_arabic = arabic_list[start:end]
            batch_true = true_list[start:end]

            if use_beam:
                batch_pred = [self.transliterate_word(a, use_beam=True) for a in batch_arabic]
            else:
                batch_indices = [
                    arabic_encoder.pad_sequence(arabic_encoder.encode(a), self.max_input_len)
                    for a in batch_arabic
                ]
                encoder_input = np.array(batch_indices, dtype="int32")
                if use_tf:
                    token_matrix = self._greedy_decode_batch_tf(encoder_input, start_idx, end_idx).numpy()
                    batch_pred = []
                    for row in token_matrix:
                        seq = []
                        for t in row:
                            if t == end_idx:
                                break
                            seq.append(int(t))
                        batch_pred.append(english_encoder.decode(seq))
                else:
                    batch_pred = self._greedy_decode_batch(encoder_input, start_idx, end_idx)

            for arabic, true_english, pred_english in zip(batch_arabic, batch_true, batch_pred):
                similarity = self.calculate_similarity(pred_english, true_english)
                char_acc = self.calculate_char_accuracy(pred_english, true_english)
                error_type = self.categorize_error(pred_english, true_english)

                error_categories[error_type] += 1
                total_similarity += similarity
                total_char_acc += char_acc

                results.append(
                    {
                        "arabic": arabic,
                        "true": true_english,
                        "predicted": pred_english,
                        "similarity": similarity,
                        "char_accuracy": char_acc,
                        "error_type": error_type,
                    }
                )

        n = len(results)
        metrics = {
            "total_samples": n,
            "exact_match": error_categories["exact"] / n * 100,
            "case_insensitive": (error_categories["exact"] + error_categories["case_only"]) / n * 100,
            "functional": (
                error_categories["exact"]
                + error_categories["case_only"]
                + error_categories["ending_h_a"]
                + error_categories["near_miss"]
            )
            / n
            * 100,
            "avg_similarity": total_similarity / n * 100,
            "avg_char_accuracy": total_char_acc / n * 100,
            "error_breakdown": dict(error_categories),
            "results": results,
        }
        return metrics


def load_kaggle_model(model_dir: Path):
    config_path = model_dir / "word_model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    encoder_pair = EncoderPair()
    encoder_pair.load(str(model_dir / "word_encoders"))

    model = build_model(
        arabic_vocab_size=config["arabic_vocab_size"],
        english_vocab_size=config["english_vocab_size"],
        config=config,
    )

    max_input_len = config.get("max_input_len", 20)
    max_output_len = config.get("max_output_len", 25)
    sample_enc = np.zeros((1, max_input_len), dtype="int32")
    sample_dec = np.zeros((1, max_output_len), dtype="int32")
    _ = model([sample_enc, sample_dec])

    model.load_weights(str(model_dir / "word_model.weights.h5"))
    return model, encoder_pair, config


def print_validation_report(metrics: dict, title: str = "Validation Results"):
    print(f"\n{'='*60}")
    print(title.upper())
    print(f"{'='*60}")

    print("\n📊 ACCURACY METRICS")
    print("-" * 40)
    print(f"  Total samples:        {metrics['total_samples']:,}")
    print(f"  Exact Match:          {metrics['exact_match']:.1f}%")
    print(f"  Case-Insensitive:     {metrics['case_insensitive']:.1f}%")
    print(f"  Functional Accuracy:  {metrics['functional']:.1f}%")
    print(f"  Avg Similarity:       {metrics['avg_similarity']:.1f}%")
    print(f"  Avg Char Accuracy:    {metrics['avg_char_accuracy']:.1f}%")


def interactive_test(validator):
    """Interactive testing mode."""
    print(f"\n{'=' * 60}")
    print("INTERACTIVE TESTING MODE")
    print("=" * 60)
    print("Enter Arabic words to transliterate. Type 'quit' to exit.\n")

    while True:
        arabic = input("Arabic word: ").strip()
        if arabic.lower() in ['quit', 'exit', 'q']:
            break

        greedy = validator.transliterate_word(arabic, use_beam=False)
        beam = validator.transliterate_word(arabic, use_beam=True, beam_width=5 )

        print(f"  Greedy: {greedy}")
        print(f"  Beam:   {beam}")
        print()


def main():
    print("\n" + "=" * 60)
    print("FAST TRANSLITERATION MODEL VALIDATION")
    print("=" * 60)

    model_dir = Path("models/new_model")
    data_dir = Path("models/new_model")

    required_files = [
        model_dir / "word_model.weights.h5",
        model_dir / "word_model_config.json",
        model_dir / "word_encoders" / "arabic_encoder.json",
        model_dir / "word_encoders" / "english_encoder.json",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("\n❌ Missing model files:")
        for f in missing:
            print(f"   - {f}")
        print("\n💡 Please place model artifacts into 'models/new_model/'.")
        return

    print("\n📥 Loading model...")
    model, encoder_pair, config = load_kaggle_model(model_dir)
    print(f"   Arabic vocab: {config['arabic_vocab_size']}")
    print(f"   English vocab: {config['english_vocab_size']}")

    validator = ModelValidator(
        model,
        encoder_pair,
        config.get("max_input_len", 20),
        config.get("max_output_len", 25),
    )

    print("\n📄 Loading test data...")
    test_df = pd.read_csv(data_dir / "test.csv")
    test_df = test_df.rename(columns={"arabic_name": "arabic_word", "english_name": "english_word"})
    test_unique = test_df.drop_duplicates(subset=["arabic_word", "english_word"])
    print(f"   {len(test_df)} word pairs → {len(test_unique)} unique")

    sample_size = 500  # Set to None for full validation
    print(f"\n🔄 Running validation on {sample_size or 'all'} samples...")
    metrics = validator.validate(
        test_unique,
        sample_size=sample_size,
        use_beam=False,
        batch_size=256,
        use_tf=True,
    )
    print_validation_report(metrics, "Optimized Greedy Decoding Results")

    results_df = pd.DataFrame(metrics["results"])
    results_path = model_dir / "validation_results_fast.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Detailed results saved to: {results_path}")

    # Optional: Interactive testing
    response = input("\n🎯 Enter interactive testing mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_test(validator)

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()