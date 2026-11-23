"""Known-class and open-set metrics for the baseline."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def rejection_scores(
    logits: np.ndarray,
    rule: str = "energy",
    temperature: float = 1.0,
) -> np.ndarray:
    """Known-ness scores; larger means more likely known."""
    z = np.asarray(logits, dtype=np.float64)

    if rule == "max_softmax":
        shifted = z - z.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities.max(axis=1)

    if rule == "max_logit":
        return z.max(axis=1)

    if rule == "energy":
        pivot = z.max(axis=1, keepdims=True)
        return temperature * (
            pivot[:, 0] / temperature
            + np.log(np.exp((z - pivot) / temperature).sum(axis=1))
        )

    raise ValueError(f"Unknown rejection rule: {rule}")


def auroc_unknown_positive(
    scores_known: np.ndarray,
    scores_unknown: np.ndarray,
) -> float:
    """AUROC with unknown as positive and -s(x) as the score."""
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    n_known, n_unknown = s_known.size, s_unknown.size
    if n_known == 0 or n_unknown == 0:
        return float("nan")

    combined = np.concatenate([s_known, s_unknown])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    _, inverse, counts = np.unique(
        combined, return_inverse=True, return_counts=True
    )
    rank_sums = np.bincount(inverse, weights=ranks)
    ranks = (rank_sums / counts)[inverse]

    u_statistic = ranks[:n_known].sum() - n_known * (n_known + 1) / 2.0
    return float(u_statistic / (n_known * n_unknown))


def auoscr(
    scores_known: np.ndarray,
    correct_known: np.ndarray,
    scores_unknown: np.ndarray,
) -> float:
    """Area under the OSCR curve (CCR vs unknown false-acceptance rate)."""
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    correct = np.asarray(correct_known, dtype=bool).ravel()
    if s_known.size == 0 or s_unknown.size == 0:
        return float("nan")

    thresholds = np.unique(np.concatenate([s_known, s_unknown]))[::-1]
    fpr = [0.0]
    ccr = [0.0]
    for threshold in thresholds:
        accepted_known = s_known >= threshold
        accepted_unknown = s_unknown >= threshold
        ccr.append(float(np.sum(accepted_known & correct)) / s_known.size)
        fpr.append(float(np.mean(accepted_unknown)))
    fpr.append(1.0)
    ccr.append(float(np.mean(correct)))

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(np.asarray(ccr), np.asarray(fpr)))
    return float(np.trapz(np.asarray(ccr), np.asarray(fpr)))


def known_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict:
    predictions = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(labels).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
    }


@torch.no_grad()
def collect_outputs(model, loader, device) -> dict:
    model.eval()
    logits, labels, unknown = [], [], []
    groups, shared, image_ids = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits.append(model(images).detach().cpu())
        labels.append(batch["label"])
        unknown.append(batch["is_unknown"])
        groups.extend(batch["unknown_group"])
        shared.extend(batch["disease_shared_with_known"].tolist())
        image_ids.extend(batch["image_id"])

    return {
        "logits": torch.cat(logits).numpy(),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "is_unknown": torch.cat(unknown).numpy().astype(bool),
        "unknown_group": np.asarray(groups),
        "disease_shared": np.asarray(shared, dtype=bool),
        "image_ids": image_ids,
    }


def group_metrics(
    outputs: dict,
    rule: str,
    scores_known_val: np.ndarray | None = None,
    acceptance: float = 0.95,
) -> dict:
    """AUROC/AUOSCR and optional fixed operating point for each unknown group."""
    known_mask = ~outputs["is_unknown"]
    labels = outputs["labels"]
    groups = outputs["unknown_group"]
    shared = outputs["disease_shared"]

    scores_known = rejection_scores(outputs["logits"][known_mask], rule)
    correct_known = (
        outputs["logits"][known_mask].argmax(axis=1) == labels[known_mask]
    )

    masks = {
        "familiar_host": groups == "familiar_host",
        "familiar_shared": (groups == "familiar_host") & shared,
        "familiar_novel": (groups == "familiar_host") & ~shared,
        "unfamiliar_host": groups == "unfamiliar_host",
    }

    results = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        scores_unknown = rejection_scores(outputs["logits"][mask], rule)
        entry = {
            "auroc": auroc_unknown_positive(scores_known, scores_unknown),
            "auoscr": auoscr(scores_known, correct_known, scores_unknown),
            "n_unknown": int(mask.sum()),
        }
        if scores_known_val is not None:
            threshold = float(
                np.quantile(scores_known_val, 1.0 - acceptance)
            )
            accepted_known = scores_known >= threshold
            entry["operating_point"] = {
                "threshold": threshold,
                "known_acceptance": float(accepted_known.mean()),
                "correct_and_accepted_rate": float(
                    np.sum(accepted_known & correct_known) / scores_known.size
                ),
                "unknown_false_acceptance_rate": float(
                    np.mean(scores_unknown >= threshold)
                ),
            }
        results[name] = entry
    return results
