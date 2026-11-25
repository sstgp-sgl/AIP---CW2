"""Agreement between the global head and the lesion head.

The score is large when the global head is confident and the two softmaxes
put their mass on the same classes. A monotone function of max-softmax alone
would leave the ranking unchanged, so a drop means the lesion head disagreed.
"""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def consistency_scores(
    global_logits: np.ndarray, lesion_logits: np.ndarray
) -> np.ndarray:
    global_probabilities = _softmax(global_logits)
    lesion_probabilities = _softmax(lesion_logits)
    confidence = global_probabilities.max(axis=1)
    agreement = (global_probabilities * lesion_probabilities).sum(axis=1)
    return confidence * agreement
