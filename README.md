# REID Pipeline for YOLO11/YOLO26 Pose Models

This repository trains a ReID adapter from a single non-ReID pose checkpoint and fuses it into a YOLO model with a ReID-capable pose head.

Primary entrypoint: `reid_pipeline.py`

---

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate reid
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Quick start

Run full pipeline:

```bash
python reid_pipeline.py \
  --base-model /mldata/models/v10/pt/yolo26s-v10-210226.pt \
  --run all
```

Run fast smoke test (small dataset + short training):

```bash
python reid_pipeline.py \
  --base-model /mldata/models/v10/pt/yolo26s-v10-210226.pt \
  --run all \
  --test \
  --force
```

---

## CLI reference

```bash
python reid_pipeline.py \
  --base-model <path/to/non_reid_pose.pt> \
  --run <all|build-dataset|train-adapter|fuse|sanity|eval> \
  [--config reid.yaml] \
  [--project /path/to/output_root] \
  [--name run_name] \
  [--max-processes N] \
  [--test] \
  [--force]
```

### Required

- `--base-model`: non-ReID YOLO pose checkpoint.
- `--run`: stage to run, or `all`.

### Optional

- `--config`: YAML/JSON user config (datasets, train hyperparameters, optional overrides).
- `--project`: output root override.
- `--name`: run folder name override.
- `--max-processes N`: cap the number of dataset-generation worker processes (caps `num_workers` from config).
- `--test`: smoke mode (few hundred images, short training).
- `--force`: re-run stage even if artifact already exists.

---

## Stage behavior

- `build-dataset`
  - loads configured ReID loaders
  - extracts per-detection vectors from base model
  - saves `reid_dataset.npz`
- `train-adapter`
  - trains triplet-loss adapter
  - saves `reid_adapter.pth`
- `fuse`
  - promotes base head in-place to a ReID pose head (`PoseReID` or `Pose26ReID`)
  - injects adapter weights
  - saves fused `.pt` and exports `.onnx`
- `sanity`
  - runs base and fused models on one test image
  - verifies detection parity and ReID vector shape/presence
  - writes `sanity_fused_vs_base.json`
- `eval`
  - runs evaluation and writes `eval_results.json`

`--run all` executes:

`build-dataset -> train-adapter -> fuse -> sanity -> eval`

---

## Operating model

You provide one base model:

- `--base-model /path/to/yolo11_or_yolo26_pose.pt`

The pipeline infers model-specific details from that checkpoint, including:

- head family (`Pose` vs `Pose26`)
- class count and names
- attribute metadata (`attr_nc`, attribute names when available)
- keypoint shape
- compatible ReID head variant

The only config inputs should be user choices (dataset sources, sampling, training hyperparameters, output paths, optional advanced overrides).

---

## ReID internals: adapter, training, and fusion

### Adapter model

The ReID adapter (`ReIDAdapter`) is a FiLM-modulated MLP:

- input: per-detection feature vector produced from the detector pipeline
- modulation: scale code is projected to FiLM `gamma/beta`
- embedding head: MLP + LayerNorm + L2 normalization with learnable scale
- output: fixed-length ReID embedding used for retrieval/comparison

One-line overhead example (default adapter for yolo26l feature contract `in_dim=575`, `hidden1=160`, `hidden2=192`, `emb=80`): **+147,599 params** and **~145,872 MACs (~0.292 MFLOPs) per emitted ReID vector**.

In fused models, this adapter sits in the pose head path and emits ReID vectors directly during inference.

### Training

Adapter training uses triplet-style metric learning over extracted detection features:

- dataset build stage creates feature/identity pairs from multiple ReID loaders
- train stage optimizes adapter parameters to bring same-ID vectors closer and different-ID vectors farther
- training settings are controlled by config (`train.*` fields: epochs, batch size, learning rate, patience, augmentation knobs)
- random erasing is applied at dataset-build grid time (grid-aware, per-cell) as a practical workaround while backbone training remains frozen
- adapter architecture can be adjusted via top-level config keys `hidden1`, `hidden2`, `emb` (defaults: 160, 192, 80)

The detector backbone/head remains fixed during adapter training; only adapter weights are learned.

`reid_train_triplet.py` can also be run standalone against a pre-built `.npz` dataset:

```bash
python reid_train_triplet.py --config /mldata/config/reid/reid_train.yaml
```

### Fusion

Fusion takes:

- base non-ReID pose checkpoint
- trained adapter checkpoint

and produces:

- fused ReID-enabled pose model (`.pt`)
- optional export (`.onnx`)

Fusion behavior:

- promotes the terminal pose head to a ReID variant (`PoseReID`/`Pose26ReID`)
- copies detector weights from base model
- injects adapter weights
- preserves key metadata needed for inference compatibility

The sanity stage then validates base-vs-fused detection parity and ReID output structure.

---

## Training improvements and Adapter V2

Two waves of improvements, applied in order:

### Wave 1 — Training overhaul (modern ReID recipe)

The adapter trainer was rewritten to use the combination of techniques that define the current ReID state of the art. The core change is that identity-balanced mini-batches plus batch-hard mining replace per-sample anchor/positive/negative triplets, and a classification head runs alongside the triplet loss for stronger feature separation.

Itemized:

1. **P/K batch-hard triplet sampler.** Each batch contains `P` identities × `K` samples per identity (default 64 × 4 = 256). The triplet loss picks the hardest positive and hardest negative for every anchor inside the batch (Hermans, Beyer, Leibe 2017). This is both more sample-efficient and more aggressive than random triplet sampling.
2. **Classification head alongside triplet.** A Linear/AM-Softmax/ArcFace classifier is trained jointly so the embedding space is shaped both locally (triplet, per-batch) and globally (softmax, per-identity). Only the adapter is saved — the classifier is training-only scaffolding.
3. **Cosine-space triplet mining.** Batch-hard mining uses cosine similarity rather than Euclidean distance. On the unit sphere the two are monotonically related (`d² = 2 − 2·cos`), but cosine avoids the `sqrt(0)` backward-pass NaN when the adapter collapses multiple identities onto the same point.
4. **BNNeck.** A BatchNorm layer with frozen zero bias sits between the triplet-path feature and the classifier (Luo 2019). The triplet loss sees the pre-BN feature, the classifier sees the BN'd feature — this alignment empirically improves both losses simultaneously.
5. **Margin softmax heads.** AM-Softmax (Wang 2018, cosine margin) and ArcFace (Deng 2019, angular margin) replace plain softmax when enabled. `am_softmax` is the default: `scale=30.0`, `margin=0.35`.
6. **Cross-Batch Memory (XBM).** A FIFO queue of recent batches' embeddings (default 8192) augments the negative pool for batch-hard mining (Wang 2020). The queue entries are detached — gradient only flows through the current batch — so the memory cost is tiny versus using a 8192-sized real batch. Activates after a short warm-up (epoch 3) once features stabilise.
7. **Soft-margin triplet.** `softplus(cos_an − cos_ap)` replaces the hard-margin `relu(d_ap − d_an + m)` form. Same saturating shape, no margin hyperparameter, and non-zero gradient even after the margin is satisfied.
8. **Composite checkpoint metric.** `src/reid_eval.py` now also computes d-prime (Cohen's d on same-ID vs cross-ID cosine distributions) alongside CMC Recall@K and standard-ReID mAP. Checkpoint selection can be driven by `r1`, `avg_recall`, `dprime`, or `composite = 0.5·R@1 + 0.5·min(d',3)/3`. d-prime is calibration-sensitive — two models with identical R@1 can contribute very differently to the tracker's additive cost.

Two smaller follow-ups after the main overhaul:

- **NaN guard.** A single non-finite loss or gradient permanently poisons AdamW's running moments. The loop now detects non-finite loss/grad, skips the step/enqueue/stats, and logs the count per epoch.
- **Checkpoint-metric default changed from `composite` to `r1`.** With the new losses R@1 keeps climbing for longer than d-prime does, and the `composite` schedule was picking earlier epochs while R@1 was still improving.

Default hyperparameters: `lr=0.01` (bumped from `1e-2` → `2e-2` during the overhaul; config-controlled), `pk_P=64`, `pk_K=4`, `ce_enabled=true`, `ce_mode=am_softmax`, `ce_scale=30.0`, `ce_margin=0.35`, `xbm_enabled=true`, `xbm_size=8192`, `xbm_start_epoch=3`, `soft_margin=true`.

### Wave 2 — Adapter V2 (fp16/int8-safe head)

With the wave-1 training in place, a latent architectural issue in `ReIDAdapter` surfaced: fp16 inference started producing all-NaN embeddings for some fully-trained checkpoints. This was traced to the `self.scale` parameter and fixed by a new `ReIDAdapterV2` head class, which is now the default for new training runs.

#### Why V2 was necessary: what broke in V1

V1's forward ends with `F.normalize(emb) * self.scale` where `self.scale` is a learnable scalar initialised to 10. Over training, `self.scale` collapses toward zero (observed as low as ~0.002), and the FiLM / MLP weights grow proportionally large (observed `|w|` up to ~220) to preserve the direction of the embedding. Mathematically the unit vector is unchanged, so cosine similarity is unchanged and training metrics look fine — but at fp16 inference the intermediate activations between MLP Linear layers overflow (fp16 max ≈ 65 504), the tail LayerNorm on an `inf` input returns NaN, and the output becomes NaN. Downstream, FAISS returns sentinel index `-1` for NaN queries, which numpy silently interprets as negative indexing into the gallery — giving the degenerate `Q_valid=1, R@1=1.00` signature seen on the v10r2 checkpoint.

A sequence of V1-only patches confirmed the pathology was architectural, not a hyperparameter problem:

- freeze `self.scale=10` → R@1 ≈ 0.612 (regresses from the unclamped 0.662).
- clamp `self.scale ∈ [1.0, 20.0]` → scale pins at 1.0 → R@1 ≈ 0.613.
- clamp `self.scale ∈ [0.1, 20.0]` → scale pins at 0.1 → R@1 ≈ 0.665, but fp16 inference still NaN (MLP weights still grow to compensate for the small output magnitude).

V1 training wants a small output magnitude; producing a small output magnitude requires the MLP weights to grow; and large MLP weights overflow fp16 intermediates. Any fix that only touches `self.scale` slides along the trade-off rather than breaking it.

#### What V2 does differently

V2 (`ReIDAdapterV2` in `ultralytics/nn/modules/head.py`) decouples output magnitude from weight magnitude by architecture:

- **LayerNorm after every Linear** (hidden1, hidden2, output). Each intermediate is rescaled to mean-0 / var-1 independent of upstream weight norms — large weights can no longer produce large activations.
- **GELU** instead of ReLU for smoother gradients at small widths.
- **Pre-LN FiLM input**; the FiLM projection is zero-initialised so the adapter begins as identity on the features.
- **No `self.scale`**; output is `F.normalize(emb, p=2, dim=1, eps=1e-4)`. The fp16-safe `eps` keeps the divisor within fp16's normal range (≈6e-5 min normal).
- **Default `emb=96`** (up from 80).

Empirical validation: forcing all V2 weights to 200× their default scale still produces finite, unit-norm output in both fp32 and fp16, whereas the same stress test on V1 produces NaN in fp16. The failure mode is eliminated, not worked around.

#### Selecting V1 vs V2

Training config keys (`reid_train.yaml` or pipeline config):

```yaml
adapter_version: 2   # 1 = legacy ReIDAdapter, 2 = ReIDAdapterV2 (default)
emb: 96              # default for V2 is 96; V1 defaults to 80
```

Saved `.pth` files contain enough structure that the version is auto-detected on load (`feat_ln.weight` present ⇒ V2), so `reid_pipeline.py`, `reid_dpar_test.py`, and `reid_model.py` all dispatch correctly without explicit version flags at consumption time. Fusion promotes the pose head to `PoseReID`/`Pose26ReID` (for V1 weights) or `PoseReIDV2`/`Pose26ReIDV2` (for V2 weights) as appropriate.

#### Downstream deployment note

The tracker in `ubon_cstuff/src/track/utrack/utrack.c` currently has `#define REID_VECTOR_LEN 80`. For V2 models this needs to match the V2 embedding width (96 by default); coordinate the tracker update with the first V2 model rollout.

---

## Config format

Only user intent belongs in config (datasets + training controls).

Example:

```yaml
project: /mldata/reid_runs

datasets:
  loaders: [UbonSyntheticLoader, CUHKLoader, IUSTLoader, LPWLoader]
  max_ids_per_image: 16
  # Optional: cap IDs per loader (useful for quick runs)
  # max_ids_per_loader: 200

yolo_batch_size: 16   # grids per YOLO call; lower if OOM during dataset build
num_workers: 4        # parallel dataset-build processes

train:
  epochs: 100
  batch_size: 4096    # triplet training batch size (features are in-memory)
  lr0: 0.01
  patience: 10
  augmentations: 15   # extra augmented copies per image during dataset build
  aug_rotate: 0.1
  aug_effects: 0.8    # probability to apply post-grid Albumentations effects
  albumentations_set: standard   # standard | aggressive_motion | none
  random_erasing_prob: 0.5       # probability a grid gets erasing pass
  random_erasing_per_box_prob: 0.35
  random_erasing_min_area: 0.02  # relative to each grid-box area
  random_erasing_max_area: 0.12
  random_erasing_min_aspect: 0.5
  random_erasing_max_aspect: 2.0
  random_erasing_max_regions: 1  # conservative "grid-time" mode
  random_erasing_fill_mode: mean # mean | black | constant | random
  random_erasing_fill_value: 0   # only used when fill_mode=constant
  # Optional per-loader overrides (keys match loader names, case-insensitive)
  loader_augmentations:
    UbonSyntheticLoader:
      augmentations: 20
      aug_rotate: 0.2
      aug_effects: 0.95
      albumentations_set: aggressive_motion
      random_erasing_prob: 0.65

# Optional overrides:
# reid_yaml: /path/to/manual_posereid_yaml.yaml   # explicit YAML instead of in-place head promotion
# fuse_test_image: /path/to/image.jpg             # image for fuse sanity check
# sanity_image: /path/to/image.jpg               # image for sanity stage (falls back to fuse_test_image)
# train_aug_preview_count_per_loader: 12         # number of post-augmentation preview grids to save per train loader
```

---

## Artifacts and manifest

Outputs are written under:

- `<project>/<run_name>/`

Typical files:

- `reid_dataset.npz`
- `reid_adapter.pth`
- `<base-stem>-posereid.pt`
- `<base-stem>-posereid.onnx`
- `sanity_fused_vs_base.json`
- `eval_results.json`
- `manifest.json`

`manifest.json` records:

- resolved config
- inferred base model metadata
- stage completion status
- event timeline for reruns/debugging

---

## Sanity check details

The sanity stage compares base vs fused on one image (`sanity_image` -> `fuse_test_image` -> `/mldata/image/arrest2.jpg`):

- detection count equality
- one-to-one matching by class + IoU
- strict parity metrics (IoU and confidence deltas)
- ReID output checks:
  - embeddings present
  - row count equals detection count
  - embedding dim matches fused head expectation
  - person detections have valid vectors

If sanity fails, stage raises with path to `sanity_fused_vs_base.json`.

---

## Dataset loaders

Supported loader names:

- `LastLoader`
- `CUHKLoader`
- `LPWLoader`
- `IUSTLoader`
- `UbonSyntheticLoader` (splits each 4x4 synthetic grid into 16 BGR subimages per ID)

Loaders live in `src/loaders/` and provide:

- `get_ids()`
- `get_image_paths(id)`

---

## `reid_dpar_test.py` protocol and visual outputs

`reid_dpar_test.py` uses a query/gallery-style protocol:

- one deterministic query per identity (when at least two images exist)
- remaining images become gallery
- reported metrics:
  - `Rank-1`, `Rank-5`, `Rank-10`, `Rank-20` (CMC)
  - `mAP`
  - `num_query_total`, `num_query_valid`
- detection misses reported separately as `num_missed`

Model spec supports:

- `<yolo_model>`
- `<yolo_model>,<reid_adapter.pth>`
- `<engine_or_trt_model>,<param_yaml>` (runtime parameter sidecar, not adapter)

When an adapter is provided, raw detector features are projected through the adapter before retrieval scoring.

### Query match sheets

For each dataset/model/grid run, the script writes deterministic query-vs-topK sheets:

- one query tile + top matches with ID and similarity overlays
- fixed auto-selected query inputs for visual comparability across runs

Output location:

- config key: `visual_output_dir`
- default: `/mldata/results/reid/query_gallery_visuals`
- per run: `<visual_output_dir>/<sanitized_test_name>/<dataset>/`

Example config snippet:

```yaml
results_cache_file: /mldata/results/reid/reid_test.pkl
visual_output_dir: /mldata/results/reid/query_gallery_visuals
```

---

## License

This project is dual-licensed under:

- **AGPL** for non-commercial use
- **[Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)**
