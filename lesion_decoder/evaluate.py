"""Evaluate a mask-supervised checkpoint on validation or test.

Inference uses the classification head only. The fixed operating-point
threshold comes from known validation scores stored in the checkpoint.

Example:
    python -m lesion_decoder.evaluate --checkpoint runs/A_mask_seed0/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baseline.data import BaselineDataset, eval_transform, load_manifest, split_rows
from baseline.metrics import (
    collect_outputs,
    group_metrics,
    known_metrics,
    rejection_scores,
)
from lesion_decoder.network import MaskResNet18

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
    model = MaskResNet18(
        num_classes=len(class_names),
        dropout=config["dropout"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    outputs = collect_outputs(model, loader, device)
    known = known_metrics(
        outputs["logits"][~outputs["is_unknown"]],
        outputs["labels"][~outputs["is_unknown"]],
        len(class_names),
    )

    val_logits = checkpoint["val_logits"].numpy()
    val_known = val_logits[~checkpoint["val_is_unknown"].numpy().astype(bool)]

    results = {
        "method": "mask",
        "assignment": assignment,
        "seed": config["seed"],
        "split": args.split,
        "lambda_mask": config.get("lambda_mask"),
        "known": known,
        "groups": {},
    }
    for rule in RULES:
        scores_known_val = rejection_scores(val_known, rule)
        results["groups"][rule] = group_metrics(
            outputs,
            rule,
            scores_known_val=scores_known_val,
            acceptance=0.95,
        )

    output = args.output or (args.checkpoint.parent / f"{args.split}_metrics.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(
        f"{assignment} mask seed {config['seed']} | {args.split} | "
        f"known macro-F1 {known['macro_f1']:.4f}"
    )
    print(f"{'rule':12s} {'group':18s} {'AUROC':>7s} {'AUOSCR':>7s} {'n':>5s}")
    for rule in RULES:
        for name, values in results["groups"][rule].items():
            print(
                f"{rule:12s} {name:18s} "
                f"{values['auroc']:7.4f} {values['auoscr']:7.4f} "
                f"{values['n_unknown']:5d}"
            )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
