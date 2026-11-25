# Global and lesion consistency

This folder keeps a baseline global classifier and adds a second head that sees only the predicted lesion. It is the task sheet's third example: consistency between global features and lesion-focused features.

The global head is global average pooling, dropout 0.2, and a linear layer. The lesion head pools

```text
sum(features * sigmoid(lesion logits)) / sum(sigmoid(lesion logits))
```

so background locations with a near-zero weight drop out of that average. The binary mask supervises the 7×7 lesion map with binary cross-entropy. Known-class macro-F1 uses the global head.

The mask-specific rejection score is

```text
global max-softmax × agreement of the two softmaxes
```

When the lesion head matches the global head, this score ranks images the same way as max-softmax. It falls when the lesion head disagrees. The 95% threshold for this score is taken from known validation images. Max-softmax of the global head is also reported, using the same rule as the baseline.

Validation and test read RGB images only. Both scores need the model's own lesion map. Ground-truth masks are used only while training. Scores other than the consistency score are computed with `baseline.metrics`.

## 1. Method

- The lesion map is predicted from the last 7×7×512 feature map: conv 512→128, ReLU, conv 128→1. There are no skip connections.
- The global head classifies the global average of that map.
- The lesion head classifies the mask-weighted average above. It has its own dropout and linear layer.
- Total loss = global cross-entropy + lesion-head cross-entropy + `lambda_mask` × binary cross-entropy. `lambda_mask` is 1. Both classification terms use the baseline class weights and label smoothing.
- Photo and mask loading, and the global-plus-BCE loss, are imported from `lesion_decoder`.
- Backbone, optimiser, epoch count, batch size and checkpoint rule match `lesion_decoder`.
- Each run keeps the checkpoint with the best known-validation macro-F1 of the global head.

## 2. Files

| File | Purpose |
| --- | --- |
| `network.py` | ResNet-18, global head, lesion map, lesion-only head |
| `scores.py` | global max-softmax times softmax agreement |
| `train.py` | one training run and checkpointing |
| `evaluate.py` | global-head metrics and the consistency score |
| `requirements.txt` | packages used to train and evaluate |

`lesion_decoder` must be importable. This folder uses its photo/mask loader and its classification-plus-BCE loss. Manifest loading, the validation image pipeline and the standard rejection metrics live in `baseline/`.

## 3. Commands

Run from the repository root. Pass `--data-root` when PlantSeg is not in the default location.

Train one assignment and seed:

```bash
python -m lesion_consistency.train --assignment A --seed 0 --data-root /path/to/plantseg
```

Evaluate a saved checkpoint:

```bash
python -m lesion_consistency.evaluate --checkpoint runs/A_lesion_consistency_seed0/best.pt --split test
```

Install the packages:

```bash
pip install -r lesion_consistency/requirements.txt
```

A training directory looks like this:

```text
runs/A_lesion_consistency_seed0/
    config.json
    history.json
    summary.json
    best.pt
    test_metrics.json
```

`test_metrics.json` is written by `evaluate.py`. It stores both the global-head rejection rules and the consistency score. Delete a run directory before starting the same assignment and seed again. Checkpoints are not part of this folder.

## 4. Results

Six runs: assignments A and B, seeds 0, 1 and 2, 25 epochs, `lambda_mask` 1. Values below are mean ± sample standard deviation over the three seeds.

Known-class macro-F1 uses the global head. In the table below, familiar-host AUOSCR is the consistency score. The same runs' global max-softmax AUOSCR is 0.6870 ± 0.0235 on A and 0.6225 ± 0.0095 on B. Baseline numbers are from `baseline/README.md` and use different manifest hashes. The BCE-only numbers are from `lesion_decoder/README.md`, on these same manifests.

### 4.1 Main comparison

| Metric | A, consistency | A, BCE-only decoder | A, baseline README | B, consistency | B, BCE-only decoder | B, baseline README |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Familiar-host AUOSCR | 0.6920 ± 0.0220 | 0.7001 ± 0.0408 | 0.6874 ± 0.0061 | 0.6299 ± 0.0102 | 0.6182 ± 0.0118 | 0.6188 ± 0.0129 |
| Test known-class macro-F1 | 0.8098 ± 0.0101 | 0.8133 ± 0.0148 | 0.8129 ± 0.0115 | 0.7510 ± 0.0059 | 0.7356 ± 0.0119 | 0.7378 ± 0.0025 |

The lesion head agrees with the global head on about 95% of known test images, so the consistency score stays close to global max-softmax. The lift over that same model's global max-softmax is about 0.005 on A and 0.007 on B. Assignment A does not exceed the BCE-only decoder. Assignment B is above the baseline README and the BCE-only decoder, and below the BCE+Dice runs (0.6432 ± 0.0069). Fixed-threshold familiar-host false acceptance remains high: A 0.7029 ± 0.0875, B 0.7955 ± 0.0592.

### 4.2 Per-seed test results

Familiar-host AUOSCR in this table is the consistency score. Macro-F1 is the global head.

| Assignment | Seed | Known-class macro-F1 | Familiar-host AUOSCR | Known-head agreement |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 0.8161 | 0.6679 | 0.9459 |
| A | 1 | 0.7981 | 0.7111 | 0.9639 |
| A | 2 | 0.8151 | 0.6970 | 0.9559 |
| B | 0 | 0.7538 | 0.6362 | 0.9375 |
| B | 1 | 0.7549 | 0.6353 | 0.9332 |
| B | 2 | 0.7441 | 0.6181 | 0.9504 |

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

Finished outputs are `runs/A_lesion_consistency_seed0` through `runs/B_lesion_consistency_seed2`. These runs were started after the earlier decoder variants had already been evaluated on the test split.

## 5. Requirements

Same packages as `baseline/requirements.txt`:

```bash
pip install -r lesion_consistency/requirements.txt
```
