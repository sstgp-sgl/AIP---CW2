"""Evaluate global-head scores and the lesion-agreement score.

Known-class predictions come from the global head. The consistency score is
global max-softmax times the agreement of the two softmaxes. Its 95% threshold
uses known validation images stored in the checkpoint.

Example:
    python -m lesion_consistency.evaluate --checkpoint runs/A_lesion_consistency_seed0/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline.data import BaselineDataset, eval_transform, load_manifest, split_rows
from baseline.metrics import (
    auroc_unknown_positive,
    auoscr,
    group_metrics,
    known_metrics,
    rejection_scores,
)
from lesion_consistency.network import ConsistencyResNet18
from lesion_consistency.scores import consistency_scores
from lesion_consistency.train import collect_pair

RULES = ("max_softmax", "max_logit", "energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def consistency_groups(
    outputs: dict, scores: np.ndarray, scores_known_val: np.ndarray
) -> dict:
    known = ~outputs["is_unknown"]
    labels = outputs["labels"]
    groups = outputs["unknown_group"]
    shared = outputs["disease_shared"]
    correct = outputs["logits"][known].argmax(axis=1) == labels[known]
    scores_known = scores[known]
    masks = {
        "familiar_host": groups == "familiar_host",
        "familiar_shared": (groups == "familiar_host") & shared,
        "familiar_novel": (groups == "familiar_host") & ~shared,
        "unfamiliar_host": groups == "unfamiliar_host",
    }
    threshold = float(np.quantile(scores_known_val, 0.05))
    results = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        scores_unknown = scores[mask]
        accepted_known = scores_known >= threshold
        results[name] = {
            "auroc": auroc_unknown_positive(scores_known, scores_unknown),
            "auoscr": auoscr(scores_known, correct, scores_unknown),
            "n_unknown": int(mask.sum()),
            "operating_point": {
                "threshold": threshold,
                "known_acceptance": float(accepted_known.mean()),
                "correct_and_accepted_rate": float(
                    np.sum(accepted_known & correct) / scores_known.size
                ),
                "unknown_false_acceptance_rate": float(
                    np.mean(scores_unknown >= threshold)
                ),
            },
        }
    return results


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    assignment = config["assignment"]
    data_root = args.data_root or Path(config["data_root"])
    manifest = args.manifest or Path(config["manifest"])

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    frame = load_manifest(manifest)
    rows = split_rows(frame, args.split)
    dataset = BaselineDataset(rows, data_root, eval_transform(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    class_names = checkpoint["class_names"]
    model = ConsistencyResNet18(
        num_classes=len(class_names),
        dropout=config["dropout"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    outputs = collect_pair(model, loader, device)
    known = ~outputs["is_unknown"]
    global_known = known_metrics(
        outputs["logits"][known], outputs["labels"][known], len(class_names)
    )
    lesion_known = known_metrics(
        outputs["lesion_logits"][known],
        outputs["labels"][known],
        len(class_names),
    )
    agreement = float(
        np.mean(
            outputs["logits"][known].argmax(axis=1)
            == outputs["lesion_logits"][known].argmax(axis=1)
        )
    )

    val_known = ~checkpoint["val_is_unknown"].numpy().astype(bool)
    val_global = checkpoint["val_logits"].numpy()
    val_lesion = checkpoint["val_lesion_logits"].numpy()
    scores = consistency_scores(outputs["logits"], outputs["lesion_logits"])
    val_scores = consistency_scores(val_global, val_lesion)

    results = {
        "method": "lesion_consistency",
        "assignment": assignment,
        "seed": config["seed"],
        "split": args.split,
        "lambda_mask": config.get("lambda_mask"),
        "known_global": global_known,
        "known_lesion": lesion_known,
        "known_head_agreement": agreement,
        "groups": {},
        "consistency": consistency_groups(outputs, scores, val_scores[val_known]),
    }
    for rule in RULES:
        scores_known_val = rejection_scores(val_global[val_known], rule)
        results["groups"][rule] = group_metrics(
            outputs,
            rule,
            scores_known_val=scores_known_val,
            acceptance=0.95,
        )

    output = args.output or (args.checkpoint.parent / f"{args.split}_metrics.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    familiar = results["consistency"]["familiar_host"]
    global_familiar = results["groups"]["max_softmax"]["familiar_host"]
    print(
        f"{assignment} lesion_consistency seed {config['seed']} | {args.split} | "
        f"global macro-F1 {global_known['macro_f1']:.4f} | "
        f"lesion macro-F1 {lesion_known['macro_f1']:.4f} | "
        f"known agreement {agreement:.4f}"
    )
    print(
        f"consistency familiar AUOSCR {familiar['auoscr']:.4f} "
        f"false-accept {familiar['operating_point']['unknown_false_acceptance_rate']:.4f}"
    )
    print(
        f"global max-softmax familiar AUOSCR {global_familiar['auoscr']:.4f} "
        f"false-accept {global_familiar['operating_point']['unknown_false_acceptance_rate']:.4f}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
