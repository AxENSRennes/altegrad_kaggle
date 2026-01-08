#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for molecular captioning.

Provides:
- BLEU-4: Corpus-level BLEU score with 4-gram matching
- METEOR: Sentence-level METEOR score (averaged)
- Simple text similarity metrics
"""

from typing import List, Dict, Optional
import re


def tokenize_simple(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace and punctuation.

    Args:
        text: Input text string

    Returns:
        List of lowercase tokens
    """
    # Lowercase and split on non-alphanumeric characters
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_bleu4(
    predictions: List[str],
    references: List[str],
    smooth: bool = True,
) -> float:
    """
    Compute corpus-level BLEU-4 score.

    Args:
        predictions: List of predicted captions
        references: List of reference captions
        smooth: Whether to apply smoothing for zero counts

    Returns:
        BLEU-4 score (0-100 scale)
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        print("NLTK not installed, using fallback BLEU computation")
        return compute_bleu4_fallback(predictions, references)

    # Prepare references (list of list of tokens) and hypotheses (list of tokens)
    refs = [[tokenize_simple(r)] for r in references]
    hyps = [tokenize_simple(p) for p in predictions]

    # Compute BLEU-4
    smoothing = SmoothingFunction().method1 if smooth else None
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-4 grams

    try:
        bleu = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoothing)
    except ZeroDivisionError:
        bleu = 0.0

    return bleu * 100  # Convert to percentage


def compute_bleu4_fallback(predictions: List[str], references: List[str]) -> float:
    """
    Fallback BLEU-4 computation without NLTK.

    Args:
        predictions: List of predicted captions
        references: List of reference captions

    Returns:
        Approximate BLEU-4 score (0-100 scale)
    """
    from collections import Counter
    import math

    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    total_score = 0.0
    count = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_simple(pred)
        ref_tokens = tokenize_simple(ref)

        if not pred_tokens or not ref_tokens:
            continue

        # Compute precision for each n-gram order
        precisions = []
        for n in range(1, 5):
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            # Clipped count
            clipped = sum(min(pred_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in pred_ngrams)
            total = sum(pred_ngrams.values())
            precisions.append(clipped / total if total > 0 else 0.0)

        # Geometric mean with smoothing
        if all(p > 0 for p in precisions):
            log_precision = sum(math.log(p) for p in precisions) / 4
            geo_mean = math.exp(log_precision)
        else:
            # Smoothing: add 1 to numerator and denominator
            geo_mean = 0.0

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

        total_score += bp * geo_mean
        count += 1

    return (total_score / count * 100) if count > 0 else 0.0


def compute_meteor(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute average METEOR score.

    Args:
        predictions: List of predicted captions
        references: List of reference captions

    Returns:
        Average METEOR score (0-100 scale)
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        import nltk
        # Ensure wordnet is downloaded
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    except ImportError:
        print("NLTK not installed, using fallback METEOR computation")
        return compute_meteor_fallback(predictions, references)

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_simple(pred)
        ref_tokens = tokenize_simple(ref)

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        try:
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        except Exception:
            scores.append(0.0)

    return (sum(scores) / len(scores) * 100) if scores else 0.0


def compute_meteor_fallback(predictions: List[str], references: List[str]) -> float:
    """
    Fallback METEOR-like computation based on F1 of unigram matches.

    Args:
        predictions: List of predicted captions
        references: List of reference captions

    Returns:
        Approximate METEOR score (0-100 scale)
    """
    scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize_simple(pred))
        ref_tokens = set(tokenize_simple(ref))

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        # Compute precision and recall
        matches = len(pred_tokens & ref_tokens)
        precision = matches / len(pred_tokens) if pred_tokens else 0.0
        recall = matches / len(ref_tokens) if ref_tokens else 0.0

        # F1 score (weighted harmonic mean)
        if precision + recall > 0:
            f1 = (10 * precision * recall) / (9 * precision + recall)
        else:
            f1 = 0.0

        scores.append(f1)

    return (sum(scores) / len(scores) * 100) if scores else 0.0


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted captions
        references: List of reference captions

    Returns:
        Exact match percentage (0-100)
    """
    if not predictions:
        return 0.0

    matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return matches / len(predictions) * 100


def compute_token_f1(predictions: List[str], references: List[str]) -> float:
    """
    Compute token-level F1 score.

    Args:
        predictions: List of predicted captions
        references: List of reference captions

    Returns:
        Average token F1 score (0-100)
    """
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize_simple(pred))
        ref_tokens = set(tokenize_simple(ref))

        if not pred_tokens and not ref_tokens:
            f1_scores.append(100.0)
            continue
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1 * 100)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_metrics(
    predictions: List[str],
    references: List[str],
    include_bert_score: bool = False,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: List of predicted captions
        references: List of reference captions
        include_bert_score: Whether to compute BERTScore (slower)

    Returns:
        Dictionary with metric names and values
    """
    metrics = {
        "bleu4": compute_bleu4(predictions, references),
        "meteor": compute_meteor(predictions, references),
        "token_f1": compute_token_f1(predictions, references),
        "exact_match": compute_exact_match(predictions, references),
    }

    # Optional: BERTScore (computationally expensive)
    if include_bert_score:
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
            metrics["bert_score_f1"] = F1.mean().item() * 100
        except ImportError:
            print("bert_score not installed, skipping BERTScore")

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metric values
        prefix: Optional prefix for metric names
    """
    print(f"\n{'=' * 40}")
    print(f"Evaluation Metrics{f' ({prefix})' if prefix else ''}")
    print(f"{'=' * 40}")
    for name, value in metrics.items():
        print(f"  {name:15s}: {value:6.2f}")
    print(f"{'=' * 40}\n")
