# Lesion-aware pooling

This folder uses the predicted lesion map inside classification. It is the task sheet's second example, lesion-aware feature pooling. The 7×7 decoder in `lesion_decoder` draws a map during training and discards it at test time. Here the class scores depend on that map.

The decoder matches that 7×7 head: conv 512→128, ReLU, conv 128→1, then bilinear upsample for the mask loss. Before global average pooling, the 7×7 lesion logits are turned into weights with a sigmoid. The pooled features are

```text
features + features * sigmoid(lesion logits)
```

The original features stay in the sum, and the lesion locations are emphasised again. The linear head is still dropout 0.2 and a layer from 512 to the known classes. Ground-truth masks are used only in the training loss. Validation and test read RGB images. The model predicts its own map, and the 25 class scores cannot be computed without it. Scores are computed with `baseline.metrics`.

## 1. Method

- The lesion map is predicted from the last 7×7×512 feature map. There are no skip connections.
- Classification pools `features + features * sigmoid(7×7 lesion logits)`, then uses the same dropout and linear layer as the baseline.
- Total loss = class-weighted cross-entropy + `lambda_mask` × binary cross-entropy. `lambda_mask` is 1.
- Photo and mask loading, and the loss, are imported from `lesion_decoder`.
- Backbone, optimiser, epoch count, batch size, checkpoint rule and rejection metrics match `lesion_decoder`.
- Each run keeps the checkpoint with the best known-validation macro-F1.
- The primary rejection rule remains max-softmax.

## 2. Files

| File | Purpose |
| --- | --- |
| `network.py` | ResNet-18, 7×7 lesion map, residual lesion reweighting, classification head |
| `train.py` | one training run and checkpointing |
| `evaluate.py` | validation or test evaluation from a checkpoint |
| `requirements.txt` | packages used to train and evaluate |

`lesion_decoder` must be importable. This folder uses its photo/mask loader and its classification-plus-BCE loss. Manifest loading, the validation image pipeline and the metrics live in `baseline/`.

## 3. Commands

Run from the repository root. Pass `--data-root` when PlantSeg is not in the default location.

Train one assignment and seed:

```bash
python -m lesion_aware_pool.train --assignment A --seed 0 --data-root /path/to/plantseg
```

Evaluate a saved checkpoint:

```bash
python -m lesion_aware_pool.evaluate --checkpoint runs/A_lesion_aware_pool_seed0/best.pt --split test
```

Install the packages:

```bash
pip install -r lesion_aware_pool/requirements.txt
```

A training directory looks like this:

```text
runs/A_lesion_aware_pool_seed0/
    config.json
    history.json
    summary.json
    best.pt
    test_metrics.json
```

`test_metrics.json` is written by `evaluate.py`. The 95% known-validation threshold comes from scores stored in `best.pt`. Delete a run directory before starting the same assignment and seed again. Checkpoints are not part of this folder.

## 4. Results

Six runs: assignments A and B, seeds 0, 1 and 2, 25 epochs, `lambda_mask` 1. Values below are mean ± sample standard deviation over the three seeds.

The primary outcome is test familiar-host AUOSCR with the max-softmax rule. Known-class macro-F1 is measured before rejection. The BCE-only numbers are from `lesion_decoder/README.md`, on these same manifests. Baseline numbers are from `baseline/README.md` and use different manifest hashes.

### 4.1 Main comparison

| Metric | A, lesion-aware pool | A, BCE-only decoder | A, baseline README | B, lesion-aware pool | B, BCE-only decoder | B, baseline README |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Familiar-host AUOSCR | 0.6808 ± 0.0286 | 0.7001 ± 0.0408 | 0.6874 ± 0.0061 | 0.6168 ± 0.0287 | 0.6182 ± 0.0118 | 0.6188 ± 0.0129 |
| Test known-class macro-F1 | 0.8043 ± 0.0134 | 0.8133 ± 0.0148 | 0.8129 ± 0.0115 | 0.7384 ± 0.0060 | 0.7356 ± 0.0119 | 0.7378 ± 0.0025 |

Assignment A is below the BCE-only decoder and slightly below the baseline README, and known-class macro-F1 is a little lower. Assignment B matches the baseline README. Seed 1 on B is 0.5839. On the fixed 95% threshold, familiar-host false acceptance stays about 0.78 to 0.83. Using the predicted map inside classification does not raise familiar-host rejection.

### 4.2 Per-seed test results

| Assignment | Seed | Known-class macro-F1 | Familiar-host AUOSCR |
| --- | ---: | ---: | ---: |
| A | 0 | 0.8132 | 0.6757 |
| A | 1 | 0.8109 | 0.7117 |
| A | 2 | 0.7889 | 0.6551 |
| B | 0 | 0.7369 | 0.6303 |
| B | 1 | 0.7450 | 0.5839 |
| B | 2 | 0.7334 | 0.6363 |

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

Finished outputs are `runs/A_lesion_aware_pool_seed0` through `runs/B_lesion_aware_pool_seed2`. These runs were started after the earlier decoder variants had already been evaluated on the test split.

## 5. Requirements

Same packages as `baseline/requirements.txt`:

```bash
pip install -r lesion_aware_pool/requirements.txt
```
