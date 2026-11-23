"""Train one baseline ResNet-18 run.

Example (from the repository root):

    python -m baseline.train --assignment A --seed 0

The run directory contains best.pt, history.json and summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from baseline.data import (
    BaselineDataset,
    class_names,
    class_weights,
    eval_transform,
    load_manifest,
    split_rows,
    train_transform,
)
from baseline.metrics import collect_outputs, group_metrics, known_metrics
from baseline.model import BaselineResNet18

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", choices=("A", "B"), default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "plantseg-v7" / "plantseg",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "runs"
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or (
        REPO_ROOT / "artifacts" / "splits" / f"manifest_{args.assignment}.csv"
    )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    manifest_sha256 = file_sha256(manifest)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    set_seed(args.seed)

    frame = load_manifest(manifest)
    train_rows = split_rows(frame, "train")
    val_rows = split_rows(frame, "validation")

    names = class_names(train_rows)
    num_classes = len(names)
    weights = class_weights(train_rows, num_classes)

    run_dir = (
        args.output_dir
        / f"{args.assignment}_baseline_seed{args.seed}"
    )
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} exists; delete it or choose another output directory"
        )
    run_dir.mkdir(parents=True)

    config = {
        "assignment": args.assignment,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "data_root": str(args.data_root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "pretrained": "IMAGENET1K_V1",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    train_dataset = BaselineDataset(
        train_rows, args.data_root, train_transform(args.image_size)
    )
    val_dataset = BaselineDataset(
        val_rows, args.data_root, eval_transform(args.image_size)
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = BaselineResNet18(
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=args.label_smoothing
    )
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )

    print(
        f"assignment={args.assignment} seed={args.seed} device={device} "
        f"train={len(train_dataset)} val={len(val_dataset)} classes={num_classes}"
    )
    print(f"run dir: {run_dir}")

    history = []
    best_f1 = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach())
            batches += 1
        scheduler.step()

        val_outputs = collect_outputs(model, val_loader, device)
        known = known_metrics(
            val_outputs["logits"][~val_outputs["is_unknown"]],
            val_outputs["labels"][~val_outputs["is_unknown"]],
            num_classes,
        )
        familiar = group_metrics(val_outputs, "energy").get(
            "familiar_host", {}
        ).get("auoscr", float("nan"))

        record = {
            "epoch": epoch,
            "train_loss": running_loss / batches,
            "val_macro_f1": known["macro_f1"],
            "val_accuracy": known["accuracy"],
            "val_energy_familiar_auoscr": familiar,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch:02d} | loss {record['train_loss']:.4f} "
            f"| val macro-F1 {record['val_macro_f1']:.4f} "
            f"| energy familiar AUOSCR {familiar:.4f}"
        )

        if known["macro_f1"] > best_f1:
            best_f1 = known["macro_f1"]
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "config": config,
                    "class_names": names,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_logits": torch.from_numpy(val_outputs["logits"]),
                    "val_labels": torch.from_numpy(val_outputs["labels"]),
                    "val_is_unknown": torch.from_numpy(
                        val_outputs["is_unknown"]
                    ),
                    "val_unknown_group": val_outputs[
                        "unknown_group"
                    ].tolist(),
                    "val_disease_shared": val_outputs[
                        "disease_shared"
                    ].tolist(),
                    "val_image_ids": val_outputs["image_ids"],
                },
                best_path,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "assignment": args.assignment,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(best_path),
        "run_dir": str(run_dir),
        "manifest_sha256": manifest_sha256,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("summary:", summary)


if __name__ == "__main__":
    main()
