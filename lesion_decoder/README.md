# Lesion decoder

This folder is one way of using the binary lesion mask: an auxiliary lesion decoder. It is not the only mechanism allowed by the task sheet.

The classification backbone matches the baseline. The decoder is trained on known-class masks. Validation and test inference use RGB images and the classification logits only. Scores are computed with `baseline.metrics`.

## 1. Method

The decoder reads the last 7×7×512 feature map, before global pooling, and predicts a lesion / background map. There are no skip connections.

- Total loss = class-weighted cross-entropy + `lambda_mask` × binary cross-entropy.
- `lambda_mask` is 1 in the runs below.
- Crops, flips and rotations are shared by the photo and the mask. The mask uses nearest-neighbour interpolation. Colour jitter stays on the photo.
- Backbone, dropout, optimiser, epoch count, batch size, checkpoint rule and rejection metrics match the baseline.
- Each run keeps the checkpoint with the best known-validation macro-F1.

## 2. Files

| File | Purpose |
| --- | --- |
| `image_mask.py` | photo and binary lesion mask, shared geometry |
| `network.py` | ResNet-18, classification head, lesion decoder |
| `loss.py` | classification loss plus mask loss |
| `train.py` | one training run and checkpointing |
| `evaluate.py` | validation or test evaluation from a checkpoint |

Manifest loading, the validation image pipeline and the metrics live in `baseline/`.

## 3. Commands

Run from the repository root. Pass `--data-root` when PlantSeg is not in the default location.

Train one assignment and seed:

```bash
python -m lesion_decoder.train --assignment A --seed 0 --data-root /path/to/plantseg
```

A one-epoch smoke run:

```bash
python -m lesion_decoder.train --assignment A --seed 0 --epochs 1 --data-root /path/to/plantseg
```

Evaluate a saved checkpoint:

```bash
python -m lesion_decoder.evaluate --checkpoint runs/A_mask_seed0/best.pt --split test
```

A training directory looks like this:

```text
runs/A_mask_seed0/
    config.json
    history.json
    summary.json
    best.pt
    test_metrics.json
```

`test_metrics.json` is written by `evaluate.py`. The 95% known-validation threshold comes from scores stored in `best.pt`. Delete a run directory before starting the same assignment and seed again.

## 4. Results

Six runs: assignments A and B, seeds 0, 1 and 2, 25 epochs, `lambda_mask` 1. Values below are mean ± standard deviation over the three seeds.

The primary outcome is test familiar-host AUOSCR with the max-softmax rule. Known-class macro-F1 is measured before rejection. Baseline numbers are copied from `baseline/README.md`. They are not a rerun on the manifests used here; see the hashes below.

### 4.1 Main comparison

| Metric | A, lesion decoder | A, baseline README | B, lesion decoder | B, baseline README |
| --- | ---: | ---: | ---: | ---: |
| Familiar-host AUOSCR | 0.7001 ± 0.0408 | 0.6874 ± 0.0061 | 0.6182 ± 0.0118 | 0.6188 ± 0.0129 |
| Test known-class macro-F1 | 0.8133 ± 0.0148 | 0.8129 ± 0.0115 | 0.7356 ± 0.0119 | 0.7378 ± 0.0025 |

Assignment A is slightly higher on average, but seed 0 is lower than the baseline mean, so the gain is not consistent across seeds. Assignment B matches the baseline. Known-class macro-F1 stays about the same on both assignments.

### 4.2 Per-seed test results

| Assignment | Seed | Known-class macro-F1 | Familiar-host AUOSCR |
| --- | ---: | ---: | ---: |
| A | 0 | 0.7965 | 0.6532 |
| A | 1 | 0.8191 | 0.7267 |
| A | 2 | 0.8242 | 0.7205 |
| B | 0 | 0.7404 | 0.6117 |
| B | 1 | 0.7220 | 0.6110 |
| B | 2 | 0.7444 | 0.6318 |

### 4.3 Manifests

These runs used the manifests currently in `artifacts/splits/`:

```text
e5b24a5931b038f8df5da119a8e3e3965e4cb087e6685e5c07e9a19d03106155  manifest_A.csv
a7364605280baa8364c545b4c8ddbc14315a06f173bac6632c8767b7f5f15e5d  manifest_B.csv
```

`baseline/README.md` records different hashes for the runs behind its table:

```text
96030f4d03eedbb4fa6e92eb1222375b48a2e4cc193a249cc558e3e321aca22b  manifest_A.csv
f670b44e5c0f3dea9860d206083331630808909a14997a9d7e78101230d009f5  manifest_B.csv
```

Finished outputs are `runs/A_mask_seed0` through `runs/B_mask_seed2`.

## 5. Requirements

Same packages as `baseline/requirements.txt`:

```bash
pip install -r lesion_decoder/requirements.txt
```
