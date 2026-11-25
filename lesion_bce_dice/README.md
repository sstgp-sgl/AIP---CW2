# Lesion decoder with BCE and Dice

This folder keeps the auxiliary lesion decoder and the baseline classification head. The network is the same 7×7 decoder as `lesion_decoder`. The change is the mask loss.

`lesion_decoder` uses binary cross-entropy alone. Lesion pixels are a small part of each photo, so that average can stay small when most of the background is already correct. This folder adds a per-image soft Dice term, which scores overlap between the predicted spot and the true spot.

The combined loss follows AgroXAI-MTNet, which uses BCE plus Dice for lesion segmentation on PlantSeg so that the loss is less dominated by background pixels (Ndong and Chopade, 2026). This folder borrows that loss only. It does not use their attention backbone, severity head, or explainability branch. CuFormer U-Net also combines binary cross-entropy and soft Dice for lesion segmentation (Frontiers in Plant Science, 2026).

Validation and test inference use RGB images and the classification logits only. The predicted map is not an input to the 25 class scores. Scores are computed with `baseline.metrics`.

## 1. Method

- Total loss = class-weighted cross-entropy + `lambda_bce` × BCE + `lambda_dice` × (1 − soft Dice).
- Both weights are 1. The Dice smooth term is 1. Dice is computed per image, then averaged over the batch.
- The decoder reads the last 7×7×512 map: conv 512→128, ReLU, conv 128→1, then bilinear upsample to 224×224. There are no skip connections.
- The classification head still globally pools that 7×7 map, then uses dropout 0.2 and a linear layer.
- The network and the photo/mask loader are imported from `lesion_decoder`.
- Backbone, optimiser, epoch count, batch size, checkpoint rule and rejection metrics match `lesion_decoder`.
- Each run keeps the checkpoint with the best known-validation macro-F1.

## 2. Files

| File | Purpose |
| --- | --- |
| `loss.py` | classification loss plus BCE and per-image soft Dice |
| `train.py` | one training run and checkpointing |
| `evaluate.py` | validation or test evaluation from a checkpoint |
| `requirements.txt` | packages used to train and evaluate |

`lesion_decoder` must be importable. This folder uses its ResNet-18 decoder and its photo/mask loader. Manifest loading, the validation image pipeline and the metrics live in `baseline/`.

## 3. Commands

Run from the repository root. Pass `--data-root` when PlantSeg is not in the default location.

Train one assignment and seed:

```bash
python -m lesion_bce_dice.train --assignment A --seed 0 --data-root /path/to/plantseg
```

Evaluate a saved checkpoint:

```bash
python -m lesion_bce_dice.evaluate --checkpoint runs/A_lesion_bce_dice_seed0/best.pt --split test
```

Install the packages:

```bash
pip install -r lesion_bce_dice/requirements.txt
```

A training directory looks like this:

```text
runs/A_lesion_bce_dice_seed0/
    config.json
    history.json
    summary.json
    best.pt
    test_metrics.json
```

`test_metrics.json` is written by `evaluate.py`. The 95% known-validation threshold comes from scores stored in `best.pt`. Delete a run directory before starting the same assignment and seed again. Checkpoints are not part of this folder.

## 4. Results

Six runs: assignments A and B, seeds 0, 1 and 2, 25 epochs, `lambda_bce` 1, `lambda_dice` 1. Values below are mean ± sample standard deviation over the three seeds.

The primary outcome is test familiar-host AUOSCR with the max-softmax rule. Known-class macro-F1 is measured before rejection. The BCE-only numbers are from `lesion_decoder/README.md`, on these same manifests. Baseline numbers are from `baseline/README.md` and use different manifest hashes.

### 4.1 Main comparison

| Metric | A, BCE+Dice | A, BCE only | A, baseline README | B, BCE+Dice | B, BCE only | B, baseline README |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Familiar-host AUOSCR | 0.6894 ± 0.0231 | 0.7001 ± 0.0408 | 0.6874 ± 0.0061 | 0.6432 ± 0.0069 | 0.6182 ± 0.0118 | 0.6188 ± 0.0129 |
| Test known-class macro-F1 | 0.8148 ± 0.0064 | 0.8133 ± 0.0148 | 0.8129 ± 0.0115 | 0.7430 ± 0.0117 | 0.7356 ± 0.0119 | 0.7378 ± 0.0025 |

Assignment A stays near the baseline and below the BCE-only decoder mean. Assignment B is higher than the BCE-only decoder and the baseline README on all three seeds. Known-class macro-F1 stays with the other methods. Adding Dice does not raise familiar-host rejection on both assignments.

### 4.2 Per-seed test results

| Assignment | Seed | Known-class macro-F1 | Familiar-host AUOSCR |
| --- | ---: | ---: | ---: |
| A | 0 | 0.8083 | 0.6677 |
| A | 1 | 0.8211 | 0.6868 |
| A | 2 | 0.8149 | 0.7137 |
| B | 0 | 0.7352 | 0.6353 |
| B | 1 | 0.7373 | 0.6476 |
| B | 2 | 0.7564 | 0.6467 |

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

Finished outputs are `runs/A_lesion_bce_dice_seed0` through `runs/B_lesion_bce_dice_seed2`. These runs were started after the BCE-only decoder and the FPN decoder had already been evaluated on the test split.

## 5. Requirements

Same packages as `baseline/requirements.txt`:

```bash
pip install -r lesion_bce_dice/requirements.txt
```

Ndong, M. M. E., and Chopade, S. A. (2026). AgroXAI-MTNet: an explainable attention-guided multitask deep learning framework for crop disease classification, lesion segmentation, and severity estimation. International Journal of Computer Applications, 187(145). https://www.ijcaonline.org/archives/volume187/number145/ndong-ijca-2026-c97998ffa7bb.pdf
