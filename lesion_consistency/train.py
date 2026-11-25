"""Train global and lesion heads with binary mask supervision.

Example (from the repository root):

    python -m lesion_consistency.train --assignment A --seed 0

Checkpoint selection uses the global head's known-validation macro-F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
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
)
from baseline.metrics import auoscr, known_metrics
from lesion_consistency.network import ConsistencyResNet18
from lesion_consistency.scores import consistency_scores
from lesion_decoder.image_mask import ImageMaskDataset
from lesion_decoder.loss import ClassificationPlusMaskLoss

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
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "plantseg-v7" / "plantseg",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def collect_pair(model, loader, device) -> dict:
    model.eval()
    global_logits, lesion_logits = [], []
    labels, unknown = [], []
    groups, shared, image_ids = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        global_batch, lesion_batch, _ = model(images, return_all=True)
        global_logits.append(global_batch.detach().cpu())
        lesion_logits.append(lesion_batch.detach().cpu())
        labels.append(batch["label"])
        unknown.append(batch["is_unknown"])
        groups.extend(batch["unknown_group"])
        shared.extend(batch["disease_shared_with_known"].tolist())
        image_ids.extend(batch["image_id"])
    return {
        "logits": torch.cat(global_logits).numpy(),
        "lesion_logits": torch.cat(lesion_logits).numpy(),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "is_unknown": torch.cat(unknown).numpy().astype(bool),
        "unknown_group": np.asarray(groups),
        "disease_shared": np.asarray(shared, dtype=bool),
        "image_ids": image_ids,
    }


def familiar_auoscr(outputs: dict, scores: np.ndarray) -> float:
    known = ~outputs["is_unknown"]
    familiar = outputs["unknown_group"] == "familiar_host"
    correct = outputs["logits"][known].argmax(axis=1) == outputs["labels"][known]
    return auoscr(scores[known], correct, scores[familiar])


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
    if not bool(train_rows["mask_supervision_allowed"].all()):
        raise RuntimeError("Training rows include samples without mask supervision")

    names = class_names(train_rows)
    num_classes = len(names)
    weights = class_weights(train_rows, num_classes)

    run_dir = args.output_dir / f"{args.assignment}_lesion_consistency_seed{args.seed}"
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} exists; delete it or choose another output directory"
        )
    run_dir.mkdir(parents=True)

    config = {
        "method": "lesion_consistency",
        "assignment": args.assignment,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "lambda_mask": args.lambda_mask,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "data_root": str(args.data_root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "pretrained": "IMAGENET1K_V1",
        "decoder": "conv3x3-128, relu, conv1x1, bilinear upsample, no skips",
        "lesion_pool": "sum(features * sigmoid(7x7 lesion logits)) / sum(sigmoid)",
        "rejection": "global max-softmax times softmax agreement with the lesion head",
        "mask_loss": "binary cross-entropy with logits",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    train_dataset = ImageMaskDataset(train_rows, args.data_root, args.image_size)
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

    model = ConsistencyResNet18(
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)
    criterion = ClassificationPlusMaskLoss(
        weights.to(device),
        label_smoothing=args.label_smoothing,
        lambda_mask=args.lambda_mask,
    )
    lesion_classification = torch.nn.CrossEntropyLoss(
        weight=weights.to(device), label_smoothing=args.label_smoothing
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    print(
        f"assignment={args.assignment} seed={args.seed} device={device} "
        f"train={len(train_dataset)} val={len(val_dataset)} classes={num_classes} "
        f"lambda_mask={args.lambda_mask}"
    )
    print(f"run dir: {run_dir}")

    history = []
    best_f1 = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total = 0.0
        running_global = 0.0
        running_lesion = 0.0
        running_mask = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            global_logits, lesion_logits, mask_logits = model(images, return_all=True)
            global_total, global_loss, mask_loss = criterion(
                global_logits, labels, mask_logits, masks
            )
            lesion_loss = lesion_classification(lesion_logits, labels)
            total = global_total + lesion_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_total += float(total.detach())
            running_global += float(global_loss.detach())
            running_lesion += float(lesion_loss.detach())
            running_mask += float(mask_loss.detach())
            batches += 1
        scheduler.step()

        val_outputs = collect_pair(model, val_loader, device)
        known = ~val_outputs["is_unknown"]
        global_known = known_metrics(
            val_outputs["logits"][known],
            val_outputs["labels"][known],
            num_classes,
        )
        lesion_known = known_metrics(
            val_outputs["lesion_logits"][known],
            val_outputs["labels"][known],
            num_classes,
        )
        scores = consistency_scores(
            val_outputs["logits"], val_outputs["lesion_logits"]
        )
        familiar = familiar_auoscr(val_outputs, scores)

        record = {
            "epoch": epoch,
            "train_loss": running_total / batches,
            "train_global_loss": running_global / batches,
            "train_lesion_loss": running_lesion / batches,
            "train_mask_loss": running_mask / batches,
            "val_macro_f1": global_known["macro_f1"],
            "val_lesion_macro_f1": lesion_known["macro_f1"],
            "val_consistency_familiar_auoscr": familiar,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch:02d} | loss {record['train_loss']:.4f} "
            f"(global {record['train_global_loss']:.4f} "
            f"+ lesion {record['train_lesion_loss']:.4f} "
            f"+ mask {record['train_mask_loss']:.4f}) "
            f"| val macro-F1 {record['val_macro_f1']:.4f} "
            f"| lesion F1 {record['val_lesion_macro_f1']:.4f} "
            f"| consistency familiar AUOSCR {familiar:.4f}"
        )

        if global_known["macro_f1"] > best_f1:
            best_f1 = global_known["macro_f1"]
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
                    "val_lesion_logits": torch.from_numpy(val_outputs["lesion_logits"]),
                    "val_labels": torch.from_numpy(val_outputs["labels"]),
                    "val_is_unknown": torch.from_numpy(val_outputs["is_unknown"]),
                    "val_unknown_group": val_outputs["unknown_group"].tolist(),
                    "val_disease_shared": val_outputs["disease_shared"].tolist(),
                    "val_image_ids": val_outputs["image_ids"],
                },
                best_path,
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )

    summary = {
        "method": "lesion_consistency",
        "assignment": args.assignment,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda_mask": args.lambda_mask,
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
