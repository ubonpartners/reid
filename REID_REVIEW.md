# ReID Training Pipeline — Deep Code Review

## 1. Repository Structure

| File | Purpose |
|------|---------|
| `reid_pipeline.py` | Main orchestrator. CLI entrypoint that runs all stages end-to-end: build-dataset, train-adapter, fuse, sanity, eval. |
| `reid_dataset.py` | Embedding extraction from YOLO models. Builds image grids, runs inference, collects feature vectors, saves `.npz`. Also contains legacy standalone entrypoints. |
| `reid_train_triplet.py` | Triplet-loss training loop for the ReIDAdapter MLP. Dataset class, data loader, optimizer, LR schedule, checkpointing. |
| `reid_test.py` | Legacy debug/test script. Hardcoded model paths, calls `exit()` partway through. |
| `reid_dpar_test.py` | Full query/gallery evaluation harness with visual output sheets. Uses `stuff.inference_wrapper` for end-to-end inference. |
| `src/reid_model.py` | Model fusion: promotes Pose head to PoseReID, injects adapter weights, exports ONNX. |
| `src/reid_eval.py` | Evaluation metrics: Recall@K (sklearn pairwise + FAISS variants), and standard ReID CMC/mAP with query/gallery split. |
| `src/reid_util.py` | Loader registry, background-replacement augmentation helpers. |
| `src/loaders/common.py` | Shared split and grouping utilities for loaders. |
| `src/loaders/cuhk_loader.py` | CUHK-03 loader. |
| `src/loaders/iust_loader.py` | IUST PersonReID loader. |
| `src/loaders/last_loader.py` | LaST loader. |
| `src/loaders/lpw_loader.py` | LPW (Labeled Pedestrian in the Wild) loader. |
| `src/loaders/ubon_synthetic_loader.py` | Ubon synthetic 4×4 grid loader (returns BGR numpy arrays, not paths). |

---

## 2. Embedding Generation

### How it works

1. `get_dataset_images()` iterates over configured loaders, collects image paths (or numpy arrays for `UbonSyntheticLoader`) grouped by identity.
2. `make_reid_feats()` replicates images `num_aug+1` times, shuffles them, then groups them into random grid layouts (M×N cells on a single canvas).
3. `get_feats()` packs these grid descriptions into batches, dispatches to workers (single-process or multiprocess via `stuff.mp_workqueue_run`).
4. Each worker calls `stuff.create_image_grid()` to render images onto a canvas with optional rotation/effects augmentation, then runs YOLO inference on the canvas.
5. `get_feat_process_batch()` matches grid cell bounding boxes to YOLO detections using IOMA (Intersection Over Minimum Area) scoring, and extracts the feature vector from the best-matching detection.
6. Results are collected into numpy arrays and saved to `.npz`.

### Findings

**Well done:**
- IOMA matching (`_ioma_score_matrix`) is fully vectorised and handles grid-to-detection association cleanly.
- The grid approach is clever: it batches many small person crops into a single inference pass, avoiding the overhead of running YOLO on thousands of tiny images individually.
- Multiprocess support with per-worker model instantiation is properly structured.

**Bug: multi-scale image size only applied to final remainder batch** (`reid_dataset.py`, lines 283–287)

The main loop always uses `img_size=640` (set at line 261). A random size is only sampled for the final partial remainder batch. If multi-scale augmentation during embedding extraction was intended, the size should be sampled per-batch inside the main loop, not only for the leftover.

**Bug: `__getitem__` uses last loop variable `w` for `img_size` of entire batch** (`reid_dataset.py`, line 214)

Inside `make_feat_process_batch_work`, `w["img_size"]` is taken from the last item in the batch to pass as the YOLO `imgsz`. If items within a batch could have differing target sizes (which happens for the remainder batch), the inference runs at the wrong resolution for all-but-the-last item. In practice batches are homogeneous, but this is an implicit assumption that could silently break.

**Style: `is None` comparisons written as `== None`** (`reid_dataset.py`, lines 84, 195)

`if images==None` and `if best_feats[j]!=None` should be `is None` / `is not None` per PEP 8.

**Dead code: `debug_feature_stats` block** (`reid_dataset.py`, lines 324–336)

Gated behind `debug_feature_stats=False`, never set to `True` in the pipeline. Should be removed.

**Divergent `on_predict_start` callbacks** (`reid_dataset.py` lines 133–157 vs `reid_test.py` lines 8–18)

The callback in `reid_dataset.py` uses `t.detach()` and handles end-to-end models. The copy in `reid_test.py` uses `t.clone()` and registers a `post_hook` that is never invoked. These have drifted; the `reid_test.py` version is the stale one.

---

## 3. Data Loading / Dataset Classes

### `TripletReIDDataset` (`reid_train_triplet.py`, lines 17–55)

**Well done:**
- Filters singleton identities so triplet mining always has a valid positive pair.
- Online triplet mining: anchor/positive/negative sampled fresh each epoch.

**Performance: negative sampling rebuilds list on every `__getitem__`** (line 48)

```python
random.choice([l for l in self.label_to_indices if l != anchor_label])
```

This allocates a new list of all labels (minus the anchor) on every single sample. For thousands of identities this is noticeable. Fix: precompute a list of all labels once in `__init__` and use index-based rejection sampling:

```python
# __init__
self.all_labels = list(self.label_to_indices.keys())

# __getitem__
while True:
    neg_label = random.choice(self.all_labels)
    if neg_label != anchor_label:
        break
```

**No hard/batch-hard negative mining**

Negatives are sampled uniformly at random; the training loop only applies a semi-hard filter post-hoc. Batch-hard mining (choosing the hardest negative in each batch) converges faster and produces tighter embeddings, and is straightforward to add.

### Loaders

**Missing extension filtering in IUST and CUHK loaders**

- `src/loaders/iust_loader.py`, line 17: `os.listdir()` passed directly with no extension filter. Non-image files (`.txt`, `.DS_Store`) will be treated as image paths and fail during YOLO inference.
- `src/loaders/cuhk_loader.py`, line 12: Same problem.

Fix: `[f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]`

**LaST loader does not filter non-directory entries** (`src/loaders/last_loader.py`, line 15)

`os.listdir()` on the gallery/train directory may include files, which will be iterated as identity subdirectories and fail.

**LPW loader uses hardcoded scene/view names** (`src/loaders/lpw_loader.py`, lines 14–15)

`scen1/scen2/scen3` and `view1/view2/view3` are hardcoded. Additional scenes/views are silently skipped.

**Typo: `cukh_path`** (`src/loaders/cuhk_loader.py`, line 8)

Parameter and attribute are named `cukh_path`; should be `cuhk_path`.

---

## 4. Model Architecture

### ReIDAdapter

A FiLM-modulated MLP (defined in the ultralytics fork):
- Input: concatenated per-detection feature vector (512 feat + nc class logits + attr_nc attribute logits + 3 scale codes = `in_dim`)
- FiLM conditioning on scale codes (8 dims projected to gamma/beta)
- MLP layers: `in_dim → 160 → 192 → 80`
- L2 normalisation + learnable scale at output

### Head promotion (`src/reid_model.py`)

**Well done:** `_promote_head_to_reid_inplace()` (line 89) changes the class of the existing head object in-place, preserving all learned detection weights exactly and avoiding a full model rebuild.

**Fragile: `n.endswith(name)` for weight key matching** (lines 34–50)

The adapter weight merge matches state-dict keys by suffix. If two keys in the base model share a suffix (unlikely but possible with deeply nested modules), the wrong weights could be silently overwritten. A stricter match using exact suffix after the last `.` would be safer.

**No check that all adapter keys were consumed** (lines 34–50)

If an adapter key matches multiple base keys, `replaced` is set `True` but multiple base keys are overwritten. The function warns only if a key was not replaced at all, not if it matched too many.

**Shadows builtin `dict`** (`src/reid_model.py`, line 14)

```python
dict, metadata = make_reid_feats(...)  # 'dict' shadows builtin
```

Rename to `feats_dict` or `data`.

---

## 5. Training Loop (`reid_train_triplet.py`)

**Well done:**
- Warmup + cosine annealing LR schedule via `LambdaLR` is clean.
- Margin annealing from `margin_start` to `margin_end` stabilises triplet training.
- Early stopping, saving best by average recall, and reloading best checkpoint at end.
- Semi-hard triplet filter (line 114) prevents collapse from trivially easy triplets.

**Bug: early stopping patience is off-by-one** (line 142)

```python
if epoch > best_epoch + patience:   # triggers after patience+1 epochs
```

Should be `>=` to respect the configured patience exactly.

**Logic: reported average loss is deflated when easy triplets are skipped** (line 125)

`total_loss` accumulates `loss.item() * anchor.size(0)` but when `mask.sum() == 0` the batch is skipped with `continue`. The denominator still counts those skipped samples, making the reported loss an underestimate when many triplets are trivially easy (common early in training with a pretrained model).

**Performance: `num_workers=8` hardcoded for in-memory dataset** (line 65)

The dataset is entirely in-memory numpy arrays. `num_workers=8` adds IPC serialisation overhead with no benefit. Set to `0` or `2`.

**Hardcoded adapter dimensions** (line 68)

```python
ReIDAdapter(in_dim=in_dim, hidden1=160, hidden2=192, emb=80)
```

Hidden and embedding dimensions are not configurable from the pipeline config. Add these to `DEFAULT_CONFIG` and thread through `train_cfg`.

**Missing gradient clipping**

No `clip_grad_norm_` is applied. Triplet loss with margin annealing can produce gradient spikes early in training, especially when fine-tuning from a pretrained backbone. Adding a moderate clip (e.g., `max_norm=1.0`) adds robustness at negligible cost.

---

## 6. Evaluation (`src/reid_eval.py`)

**Well done:** `evaluate_standard_reid` implements proper CMC/mAP with camera-aware junk filtering — the standard protocol for person ReID benchmarks.

**Bug: `evaluate_recall_at_k` crashes when `model is None`** (line 8)

```python
def evaluate_recall_at_k(feats_tensor, labels, model, k=5):
    model.eval()          # line 8 — crashes if model is None
    ...
    if model is None:     # line 11 — too late
```

The null check comes after the `.eval()` call. This function is not called from the pipeline (the FAISS variant is used), so it is effectively dead code with a latent crash.

**Bug: FAISS `index.add()` may receive a PyTorch tensor** (`src/reid_eval.py`, lines 39–44)

When `model is None`:
```python
embeddings = feats_tensor.cpu()   # returns Tensor, not ndarray
```
When `model is not None`:
```python
embeddings = model(feats_tensor).cpu().numpy()   # returns ndarray
```
`faiss.IndexFlatIP.add()` requires a contiguous float32 numpy array. The tensor branch works via implicit conversion in some FAISS builds but is not guaranteed. Add `.numpy()` in the `model is None` branch:
```python
embeddings = feats_tensor.cpu().numpy()
```

---

## 7. Utility / Helper Code (`src/reid_util.py`)

**Dead code: `replace_backgrounds` and all helpers are never called** (lines 28–130)

`replace_backgrounds`, `expand_canvas`, `crop_random_background`, and `paste_on_background` implement person segmentation + background replacement using YOLOv11-seg. This function is never called anywhere in the pipeline. The actual augmentation is done by `stuff.create_image_grid` with `aug_rotate`/`aug_effects`. Delete or move to a separate experiments file.

---

## 8. Config / CLI (`reid_pipeline.py`)

**Well done:**
- Stage-based pipeline with manifest tracking and skip-if-complete logic.
- `deep_merge` for config layering is simple and correct.
- `apply_test_mode` provides a proper smoke-test configuration.
- `inspect_base_model` extracts model metadata for manifest recording.

**Config default mismatches** (lines 111–124)

The `.get()` fallback defaults in `build_resolved_config` disagree with `DEFAULT_CONFIG`:

| Setting | `DEFAULT_CONFIG` | `.get()` fallback |
|---------|-----------------|-------------------|
| `yolo_batch_size` | 16 | 64 |
| `epochs` | 100 | 80 |
| `batch_size` | 4096 | 128 |
| `augmentations` | 15 | 3 |

These mismatches are harmless when `DEFAULT_CONFIG` is always applied first (as it is via `load_config`), but the fallbacks are misleading and would produce wrong values if `build_resolved_config` were called independently. Consolidate: use a single source of defaults.

---

## 9. Dead / Legacy Code

### `reid_test.py` — entire file is a debug artifact

Hardcoded model paths (`/mldata/models/v8/...`), hardcoded `ReIDAdapter(in_dim=575, ...)` dimensions, calls `exit()` on line 38, leaving all code from line 40 onwards unreachable. Should be deleted or archived.

### `reid_dataset.py` legacy entrypoints (lines 410–646)

`make_reid_dataset`, `fuse_model`, `test_reid`, `eval_reid`, and the `__main__` block all predate `reid_pipeline.py`. They duplicate pipeline functionality and have their own issues:

- `test_reid` (line 537): `dict, metadata = make_reid_feats(...)` — shadows builtin `dict`.
- `test_reid` (line 590): loops `for _ in range(len(img_list))` showing CV2 windows — hangs on large datasets.
- `eval_reid` (line 607): hardcodes `raw=True` instead of reading from metadata.
- `fuse_model` (line 499): duplicates the sanity check from `run_sanity` in the pipeline.

These should all be deleted. The interactive visualisation (`show_top_matches`) can be kept in a separate `scripts/` file if still useful.

---

## 10. Design / Complexity

### `reid_dataset.py` is doing too many things

This single file handles: dataset image collection, grid-based feature extraction with multiprocessing, feature saving, interactive visualisation, and five legacy standalone entrypoints. At minimum, split into:
- `reid_features.py` — grid construction, YOLO inference, IOMA matching, npz output
- `reid_viz.py` — interactive visualisation tools

### Augmentation parameter flow is convoluted

Settings travel through six levels with name changes:
```
DEFAULT_CONFIG ("augmentations")
  → load_config
  → build_resolved_config ("train_augmentations")
  → make_reid_feats ("num_aug")
  → get_feats
  → make_feat_process_batch_work
  → stuff.create_image_grid
```
Standardise on a single name (`num_aug`) throughout and pass a typed config dataclass instead of arbitrary dicts.

### Grid-based embedding extraction failure modes

The grid approach is good for throughput but has silent failure modes:
- Small persons in large grids may not be detected → those embeddings are silently missing.
- IOMA matching fails when detections span multiple cells or a cell produces no detection.
- The random grid sizing adds non-determinism that is hard to reproduce exactly.

Consider logging a warning or count when the IOMA match fails to find a detection above threshold, so dataset coverage can be monitored.

---

## 11. Summary Table

| Severity | File | Line(s) | Description |
|----------|------|---------|-------------|
| **Bug** | `src/reid_eval.py` | 8 | `evaluate_recall_at_k` calls `model.eval()` before null check — crashes if `model is None` |
| **Bug** | `src/reid_eval.py` | 40 | `evaluate_recall_faiss` passes PyTorch tensor to FAISS when `model is None` — should call `.numpy()` |
| **Bug** | `reid_dataset.py` | 283–287 | Multi-scale image size only applied to final remainder batch, not all batches |
| **Logic** | `reid_train_triplet.py` | 142 | Early stopping fires after `patience+1` epochs instead of `patience` (`>` vs `>=`) |
| **Logic** | `reid_train_triplet.py` | 125 | Average loss is under-reported when easy triplets are skipped |
| **Perf** | `reid_train_triplet.py` | 48 | Negative label list rebuilt on every `__getitem__` call |
| **Perf** | `reid_train_triplet.py` | 65 | `num_workers=8` hardcoded for in-memory dataset — IPC overhead for no gain |
| **Style** | `reid_dataset.py` | 84, 195 | `== None` / `!= None` instead of `is None` / `is not None` |
| **Fragile** | `src/reid_model.py` | 37 | `n.endswith(name)` for weight key matching may match unintended keys |
| **Fragile** | `src/reid_model.py` | 34–50 | No check that all adapter keys were consumed exactly once |
| **Style** | `src/reid_model.py` | 14 | Shadows builtin `dict` |
| **Style** | `src/loaders/cuhk_loader.py` | 8 | Typo: `cukh_path` should be `cuhk_path` |
| **Config** | `reid_pipeline.py` | 111–124 | `.get()` fallbacks disagree with `DEFAULT_CONFIG` values |
| **Missing** | `src/loaders/iust_loader.py` | 17 | No file extension filter on `os.listdir()` |
| **Missing** | `src/loaders/cuhk_loader.py` | 12 | No file extension filter on `os.listdir()` |
| **Missing** | `src/loaders/last_loader.py` | 15 | No directory entry filter on `os.listdir()` |
| **Missing** | `reid_train_triplet.py` | — | Adapter dimensions not configurable; no gradient clipping; no batch-hard mining |
| **Dead** | `reid_test.py` | entire | Legacy debug script; hardcoded paths; code after `exit()` is unreachable |
| **Dead** | `src/reid_eval.py` | 6–33 | `evaluate_recall_at_k` is never called by the pipeline |
| **Dead** | `src/reid_util.py` | 28–130 | `replace_backgrounds` and helpers are never called |
| **Dead** | `reid_dataset.py` | 324–336 | `debug_feature_stats` block never activated |
| **Dead** | `reid_dataset.py` | 410–646 | Legacy standalone entrypoints duplicate pipeline functionality |

---

## 12. What the Pipeline Does Well

1. **Stage-based pipeline with manifest tracking** — stages can be skipped if artifacts exist or force-rerun with `--force`. Well-engineered for iterative development.
2. **In-place head promotion** — `_promote_head_to_reid_inplace()` is elegant: changes the class of the head object in-place, preserving exact learned weights without model reconstruction.
3. **Sanity checking** — the sanity stage compares base vs. fused model predictions on the same image, catching fusion errors early.
4. **Evaluation quality** — `evaluate_standard_reid` implements proper CMC/mAP with camera-aware junk filtering, the standard protocol for person ReID benchmarks.
5. **Grid-based batch inference** — packing many person crops into grid images for batch YOLO inference is an effective throughput optimisation for embedding extraction.
6. **Deterministic splits** — all loaders use `numpy_split_list` with a fixed seed for reproducible train/val splits.
