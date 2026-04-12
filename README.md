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
  # Optional per-loader overrides (keys match loader names, case-insensitive)
  loader_augmentations:
    UbonSyntheticLoader:
      augmentations: 20
      aug_rotate: 0.2
      aug_effects: 0.95
      albumentations_set: aggressive_motion

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
