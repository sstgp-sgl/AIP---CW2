# Baseline: ResNet-18 classification

Matched baseline for CW2 Option 2: ResNet-18 with ImageNet `IMAGENET1K_V1`
weights and a classification head. The model has **no lesion decoder**, takes
only RGB images, and predicts the 25 known classes of the current assignment.

## Configuration

- Backbone: ResNet-18, ImageNet pretrained;
- head: global average pooling → dropout 0.2 → linear (25 classes);
- loss: class-weighted cross-entropy with label smoothing 0.05;
- optimiser: AdamW, lr 3e-4, weight decay 0.01, cosine schedule;
- training: 25 epochs, batch 32, 224×224;
- checkpoint selection: known-validation macro-F1;
- runs: 2 assignments × 3 seeds;
- primary rejection rule: max-softmax, selected on baseline validation
  familiar-host AUOSCR.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | manifest loading, split filtering, RGB dataset, transforms |
| `model.py` | ResNet-18 baseline |
| `metrics.py` | rejection scores, AUROC, AUOSCR, known metrics |
| `train.py` | one training run and checkpointing |
| `evaluate.py` | validation/test evaluation from a checkpoint |

## Data

The extracted PlantSeg v7 folder contains `Metadata.csv`, `images/` and
`annotations/`. Manifest paths are relative to that folder:

```text
data/raw/plantseg-v7/plantseg/
    Metadata.csv
    images/
    annotations/
artifacts/splits/manifest_A.csv
artifacts/splits/manifest_B.csv
```

If the extracted data is stored elsewhere, pass `--data-root /path/to/plantseg`
to `train.py` and `evaluate.py`.

Counts after the duplicate and mask exclusions in the current manifest:

| Assignment | Train known | Validation known | Val familiar | Val unfamiliar | Test known | Test familiar | Test unfamiliar |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1704 | 280 | 54 | 42 | 499 | 147 | 27 |
| B | 1547 | 258 | 70 | 42 | 464 | 132 | 27 |

## Train

Run from the repository root:

```bash
python -m baseline.train --assignment A --seed 0
python -m baseline.train --assignment A --seed 1
python -m baseline.train --assignment A --seed 2

python -m baseline.train --assignment B --seed 0
python -m baseline.train --assignment B --seed 1
python -m baseline.train --assignment B --seed 2
```

Outputs are written under the chosen `--output-dir` (default `runs/`):

```text
runs/A_baseline_seed0/
    config.json
    history.json
    summary.json
    best.pt
```

## Evaluate

```bash
python -m baseline.evaluate --checkpoint runs/A_baseline_seed0/best.pt --split test
```

This writes `test_metrics.json` next to the checkpoint. The fixed operating
point uses known-validation scores stored in the checkpoint.

## Results

Six runs with the configuration above. Values are mean ± standard deviation
over seeds 0, 1 and 2.

### Main metrics

| Metric | Assignment A | Assignment B |
|---|---:|---:|
| Known-val macro-F1 | 0.8146 ± 0.0184 | 0.7431 ± 0.0322 |
| Test accuracy | 0.8277 ± 0.0080 | 0.7766 ± 0.0025 |
| Test known macro-F1 | 0.8129 ± 0.0115 | 0.7378 ± 0.0025 |
| Familiar-host AUOSCR (max-softmax) | 0.6874 ± 0.0061 | 0.6188 ± 0.0129 |
| Familiar-host AUROC (max-softmax) | 0.7763 ± 0.0068 | 0.7285 ± 0.0216 |
| Unfamiliar-host AUOSCR (max-softmax) | 0.6555 ± 0.0399 | 0.6958 ± 0.0103 |
| Unfamiliar-host AUROC (max-softmax) | 0.7418 ± 0.0534 | 0.8359 ± 0.0188 |

### Primary rule selection

Familiar-host validation AUOSCR:

| Assignment | max-softmax | max-logit | energy |
|---|---:|---:|---:|
| A | **0.6865** | 0.6768 | 0.6697 |
| B | **0.6147** | 0.6016 | 0.5930 |

The primary rule is **max-softmax** for both assignments.

### Test AUOSCR by rule

| Group | max-softmax | max-logit | energy |
|---|---:|---:|---:|
| A, familiar-host | 0.6874 ± 0.0061 | 0.6790 ± 0.0071 | 0.6711 ± 0.0094 |
| B, familiar-host | 0.6188 ± 0.0129 | 0.6117 ± 0.0120 | 0.6012 ± 0.0127 |
| A, unfamiliar-host | 0.6555 ± 0.0399 | 0.6491 ± 0.0286 | 0.6413 ± 0.0304 |
| B, unfamiliar-host | 0.6958 ± 0.0103 | 0.6849 ± 0.0149 | 0.6703 ± 0.0166 |

### Fixed operating point

Threshold from known validation at approximately 95% acceptance:

| Metric | Assignment A | Assignment B |
|---|---:|---:|
| Test known acceptance | 0.9439 ± 0.0112 | 0.9239 ± 0.0076 |
| Correct-and-accepted known rate | 0.8036 ± 0.0035 | 0.7507 ± 0.0050 |
| Familiar unknown false-acceptance rate | 0.7778 ± 0.0579 | 0.7778 ± 0.0558 |
| Unfamiliar unknown false-acceptance rate | 0.7778 ± 0.0642 | 0.6914 ± 0.0932 |

### Per-seed headline values

| Assignment | Seed | Val macro-F1 | Test macro-F1 | Familiar AUOSCR (max-softmax) |
|---|---:|---:|---:|---:|
| A | 0 | 0.8343 | 0.8258 | 0.6902 |
| A | 1 | 0.7979 | 0.8036 | 0.6803 |
| A | 2 | 0.8117 | 0.8093 | 0.6916 |
| B | 0 | 0.7711 | 0.7359 | 0.6044 |
| B | 1 | 0.7504 | 0.7368 | 0.6226 |
| B | 2 | 0.7079 | 0.7406 | 0.6293 |

### Manifest provenance

The results above use these manifest files:

```text
96030f4d03eedbb4fa6e92eb1222375b48a2e4cc193a249cc558e3e321aca22b  manifest_A.csv
f670b44e5c0f3dea9860d206083331630808909a14997a9d7e78101230d009f5  manifest_B.csv
```

## Evaluation notes

- Familiar test-unknown groups have 147 (A) and 132 (B) images; per-class
  support ranges from 5 to 44. The shared-disease subgroup has 18 images in A.
- The unfamiliar-host test control has only 27 potato images.
- AUOSCR/AUROC estimates for these groups have wide uncertainty; subgroup
  differences should be treated as exploratory.
- Near-duplicate leakage is not audited in the current dataset split. Absolute
  test metrics may therefore be optimistic; both methods use the same split,
  so the paired comparison is less affected.
- Batch size 32 must match the mask-supervised model.

## Requirements

```bash
pip install -r baseline/requirements.txt
```
