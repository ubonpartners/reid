import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import stuff
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import build_reid_adapter_from_state_dict

import reid_dataset
import reid_train_triplet
import src.reid_eval as reid_eval
import src.reid_model as reid_model


DEFAULT_CONFIG = {
    "project": str(Path(__file__).resolve().parent / "runs"),
    "datasets": {
        "loaders": ["ubonsyntheticloader","CUHKLoader", "IUSTLoader", "LPWLoader"],
        "max_ids_per_image": 16,
    },
    "yolo_batch_size": 16,
    "num_workers": 4,
    "train": {
        "epochs": 100,
        "batch_size": 4096,
        "lr0": 0.01,
        "patience": 10,
        "augmentations": 15,
        "aug_rotate": 0.1,
        "aug_effects": 0.8,
        "albumentations_set": "standard",
        # Grid-aware random erasing (conservative defaults).
        "random_erasing_prob": 0.50,
        "random_erasing_per_box_prob": 0.35,
        "random_erasing_min_area": 0.02,
        "random_erasing_max_area": 0.12,
        "random_erasing_min_aspect": 0.5,
        "random_erasing_max_aspect": 2.0,
        "random_erasing_max_regions": 1,
        "random_erasing_max_attempts": 10,
        "random_erasing_min_side": 8,
        "random_erasing_fill_mode": "mean",
        "random_erasing_fill_value": 0,
    },
}


def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path):
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        user_cfg = stuff.load_dictionary(config_path)
        deep_merge(cfg, user_cfg)
    return cfg


def sanitize_name(name):
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in name)


def inspect_base_model(base_model_path):
    y = YOLO(base_model_path, verbose=False)
    head = y.model.model[-1]
    names = y.model.names
    attr_names = getattr(y.model, "attr_names", None)
    attr_ncs = getattr(y.model, "attr_ncs", None)
    info = {
        "task": y.task,
        "head_type": type(head).__name__,
        "nc": int(getattr(head, "nc", len(names))),
        "num_names": len(names),
        "attr_nc": int(getattr(head, "attr_nc", 0) or 0),
        "kpt_shape": list(getattr(head, "kpt_shape", getattr(y.model, "kpt_shape", []))),
        "num_scales": int(getattr(head, "nl", 0)),
        "feat_width": int(getattr(head, "FEAT_WIDTH", 512)),
    }
    if attr_names is not None:
        info["attr_names"] = list(attr_names)
    if attr_ncs is not None:
        info["attr_ncs"] = list(attr_ncs)
    return info


def resolve_run_paths(cfg, base_model_path, run_name=None):
    stem = Path(base_model_path).stem
    run_name = run_name or sanitize_name(stem)
    run_dir = Path(cfg["project"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "dataset": str(run_dir / "reid_dataset.npz"),
        "adapter": str(run_dir / "reid_adapter.pth"),
        "fused_pt": str(run_dir / f"{stem}-posereid.pt"),
        "fused_onnx": str(run_dir / f"{stem}-posereid.onnx"),
        "sanity": str(run_dir / "sanity_fused_vs_base.json"),
        "eval": str(run_dir / "eval_results.json"),
    }
    return paths


def build_resolved_config(cfg, base_model_path, paths):
    train_cfg = cfg.get("train", {})
    return {
        "project": cfg["project"],
        "datasets": cfg["datasets"],
        "yolo_batch_size": cfg.get("yolo_batch_size", 32),
        "num_workers": cfg.get("num_workers", 6),
        "yolo_model": base_model_path,
        "reid_dataset": paths["dataset"],
        "reid_model": paths["adapter"],
        "reid_yolo_model": paths["fused_pt"],
        "reid_onnx_model": paths["fused_onnx"],
        "train_epochs": train_cfg.get("epochs", 100),
        "train_batch_size": train_cfg.get("batch_size", 4096),
        "train_lr0": train_cfg.get("lr0", 0.01),
        "train_patience": train_cfg.get("patience", 10),
        "train_augmentations": train_cfg.get("augmentations", 15),
        "train_aug_rotate": train_cfg.get("aug_rotate", 0),
        "train_aug_effects": train_cfg.get("aug_effects", 0),
        "train_albumentations_set": train_cfg.get("albumentations_set", "standard"),
        "train_random_erasing_prob": train_cfg.get("random_erasing_prob", 0.5),
        "train_random_erasing_per_box_prob": train_cfg.get("random_erasing_per_box_prob", 0.35),
        "train_random_erasing_min_area": train_cfg.get("random_erasing_min_area", 0.02),
        "train_random_erasing_max_area": train_cfg.get("random_erasing_max_area", 0.12),
        "train_random_erasing_min_aspect": train_cfg.get("random_erasing_min_aspect", 0.5),
        "train_random_erasing_max_aspect": train_cfg.get("random_erasing_max_aspect", 2.0),
        "train_random_erasing_max_regions": train_cfg.get("random_erasing_max_regions", 1),
        "train_random_erasing_max_attempts": train_cfg.get("random_erasing_max_attempts", 10),
        "train_random_erasing_min_side": train_cfg.get("random_erasing_min_side", 8),
        "train_random_erasing_fill_mode": train_cfg.get("random_erasing_fill_mode", "mean"),
        "train_random_erasing_fill_value": train_cfg.get("random_erasing_fill_value", 0),
        "train_loader_augmentations": train_cfg.get("loader_augmentations", {}),
        "train_aug_preview_dir": str(Path(paths["run_dir"]) / "aug_preview"),
        "train_aug_preview_count_per_loader": int(cfg.get("train_aug_preview_count_per_loader", 12)),
        # Optional manual override remains supported.
        "reid_yaml": cfg.get("reid_yaml"),
        "fuse_test_image": cfg.get("fuse_test_image"),
        "sanity_image": cfg.get("sanity_image"),
    }


def apply_test_mode(cfg):
    cfg = deepcopy(cfg)
    cfg["datasets"] = cfg.get("datasets", {})
    cfg["datasets"]["loaders"] = cfg["datasets"].get("loaders", ["LastLoader"])[:1]
    cfg["datasets"]["max_ids_per_image"] = min(int(cfg["datasets"].get("max_ids_per_image", 3)), 3)
    cfg["datasets"]["max_ids_per_loader"] = int(cfg["datasets"].get("max_ids_per_loader", 80))
    cfg["yolo_batch_size"] = min(int(cfg.get("yolo_batch_size", 16)), 16)
    cfg["num_workers"] = min(int(cfg.get("num_workers", 2)), 2)
    cfg["train"] = cfg.get("train", {})
    cfg["train"]["epochs"] = min(int(cfg["train"].get("epochs", 10)), 10)
    cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", 64)), 64)
    cfg["train"]["patience"] = min(int(cfg["train"].get("patience", 3)), 3)
    cfg["train"]["augmentations"] = 0
    cfg["train"]["aug_rotate"] = 0
    cfg["train"]["aug_effects"] = 0
    cfg["train"]["albumentations_set"] = "none"
    cfg["train"]["random_erasing_prob"] = 0
    cfg["train"]["random_erasing_per_box_prob"] = 0
    cfg["train_aug_preview_count_per_loader"] = min(int(cfg.get("train_aug_preview_count_per_loader", 4)), 4)
    return cfg


def read_manifest(path):
    if not os.path.exists(path):
        return {"created_at": now_utc(), "stages": {}, "events": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def log_event(manifest, message, **extra):
    e = {"time": now_utc(), "message": message}
    e.update(extra)
    manifest["events"].append(e)


def stage_complete(manifest, stage, artifact_path):
    return manifest.get("stages", {}).get(stage, {}).get("status") == "ok" and os.path.exists(artifact_path)


def run_build_dataset(resolved, manifest):
    log_event(manifest, "build-dataset:start")
    datasets = reid_dataset.get_dataset_images(resolved)
    # Pick a deterministic sanity image for fuse-check when not provided.
    if not resolved.get("fuse_test_image"):
        for key in ("val", "train"):
            if key not in datasets:
                continue
            for identity_imgs in datasets[key]:
                for candidate in identity_imgs:
                    # Some loaders (e.g. ubon synthetic) return in-memory ndarrays.
                    # fuse_test_image must be a filesystem path for later YOLO sanity step.
                    if isinstance(candidate, str):
                        resolved["fuse_test_image"] = candidate
                        break
                if resolved.get("fuse_test_image"):
                    break
            if resolved.get("fuse_test_image"):
                break
    _, metadata = reid_dataset.make_reid_feats(config=resolved, datasets=datasets, save=True)
    manifest["stages"]["build-dataset"] = {
        "status": "ok",
        "finished_at": now_utc(),
        "artifact": resolved["reid_dataset"],
        "metadata": metadata,
        "fuse_test_image": resolved.get("fuse_test_image"),
    }
    log_event(manifest, "build-dataset:done", artifact=resolved["reid_dataset"])


def run_train_adapter(resolved, manifest):
    log_event(manifest, "train-adapter:start")
    data = np.load(resolved["reid_dataset"])
    feats = data["train_feats"]
    labels = data["train_labels"]
    val_feats = data["val_feats"]
    val_labels = data["val_labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_train_triplet.train_triplet_model(
        feats,
        labels,
        val_feats,
        val_labels,
        device=device,
        epochs=int(resolved.get("train_epochs", 20)),
        patience=int(resolved.get("train_patience", 10)),
        batch_size=int(resolved.get("train_batch_size", 128)),
        lr=float(resolved.get("train_lr0", 0.01)),
        output=resolved["reid_model"],
        emb=int(resolved.get("emb", 96)),
        adapter_version=int(resolved.get("adapter_version", 2)),
    )
    manifest["stages"]["train-adapter"] = {
        "status": "ok",
        "finished_at": now_utc(),
        "artifact": resolved["reid_model"],
    }
    log_event(manifest, "train-adapter:done", artifact=resolved["reid_model"])


def run_fuse(resolved, manifest):
    log_event(manifest, "fuse:start")
    reid_model.make_reid_model(resolved)
    manifest["stages"]["fuse"] = {
        "status": "ok",
        "finished_at": now_utc(),
        "artifact_pt": resolved["reid_yolo_model"],
        "artifact_onnx": resolved["reid_onnx_model"],
    }
    log_event(
        manifest,
        "fuse:done",
        artifact_pt=resolved["reid_yolo_model"],
        artifact_onnx=resolved["reid_onnx_model"],
    )


def _box_iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter + 1e-12
    return inter / union


def _run_predict(model_path, image_path, device):
    model = YOLO(model_path, verbose=False)
    result = model(
        image_path,
        conf=0.05,
        max_det=500,
        half=(device.startswith("cuda")),
        iou=0.45,
        verbose=False,
        device=device,
    )[0]
    boxes = result.boxes.xyxyn.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    reid = getattr(result, "reid_embeddings", None)
    if reid is not None:
        reid = reid.detach().cpu().numpy()
    names = model.names if hasattr(model, "names") else {}
    return {"boxes": boxes, "conf": conf, "cls": cls, "reid": reid, "names": names}


def _resolve_person_class_index(names):
    if isinstance(names, dict):
        for idx, name in names.items():
            if isinstance(name, str) and name.strip().lower() == "person":
                return int(idx)
    elif isinstance(names, list):
        for idx, name in enumerate(names):
            if isinstance(name, str) and name.strip().lower() == "person":
                return int(idx)
    return 0


def run_sanity(resolved, paths, manifest):
    log_event(manifest, "sanity:start")
    image_path = resolved.get("sanity_image") or resolved.get("fuse_test_image") or "/mldata/image/arrest2.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Sanity image not found: {image_path}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = _run_predict(resolved["yolo_model"], image_path, device)
    fused = _run_predict(resolved["reid_yolo_model"], image_path, device)

    used_fused = set()
    matches = []
    for i in range(len(base["boxes"])):
        best_j = -1
        best_iou = -1.0
        for j in range(len(fused["boxes"])):
            if j in used_fused:
                continue
            if int(base["cls"][i]) != int(fused["cls"][j]):
                continue
            iou = _box_iou_xyxy(base["boxes"][i], fused["boxes"][j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_fused.add(best_j)
            matches.append((i, best_j, float(best_iou)))

    count_equal = len(base["boxes"]) == len(fused["boxes"])
    matched_all = len(matches) == len(base["boxes"]) == len(fused["boxes"])
    min_iou = min([m[2] for m in matches], default=0.0)
    conf_diffs = [abs(float(base["conf"][i]) - float(fused["conf"][j])) for i, j, _ in matches]
    max_conf_diff = max(conf_diffs) if conf_diffs else None
    mean_conf_diff = (sum(conf_diffs) / len(conf_diffs)) if conf_diffs else None

    fused_model = YOLO(resolved["reid_yolo_model"], verbose=False)
    head = fused_model.model.model[-1]
    expected_reid_dim = int(getattr(getattr(head, "reid", None), "emb", 0))
    reid = fused["reid"]
    reid_present = reid is not None
    reid_rows = int(reid.shape[0]) if reid_present else 0
    reid_dim = int(reid.shape[1]) if reid_present and reid.ndim == 2 else 0
    person_class_idx = _resolve_person_class_index(fused.get("names", {}))
    person_idx = np.where(fused["cls"] == person_class_idx)[0]
    num_person_dets = int(len(person_idx))
    person_reid_ok = reid_present and reid_rows == len(fused["cls"]) and num_person_dets > 0 and reid_dim == expected_reid_dim
    det_compare_ok = count_equal and matched_all and min_iou >= 0.99
    sanity_ok = det_compare_ok and person_reid_ok

    report = {
        "status": "ok" if sanity_ok else "failed",
        "image": image_path,
        "device": device,
        "base_model": resolved["yolo_model"],
        "fused_model": resolved["reid_yolo_model"],
        "detections": {
            "base_count": int(len(base["boxes"])),
            "fused_count": int(len(fused["boxes"])),
            "count_equal": bool(count_equal),
            "matched_all": bool(matched_all),
            "min_iou": float(min_iou),
            "max_conf_diff": max_conf_diff,
            "mean_conf_diff": mean_conf_diff,
        },
        "reid": {
            "present": bool(reid_present),
            "rows": reid_rows,
            "dim": reid_dim,
            "expected_dim": expected_reid_dim,
            "person_class_idx": person_class_idx,
            "person_detections": num_person_dets,
            "person_vectors_ok": bool(person_reid_ok),
        },
    }
    with open(paths["sanity"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if not sanity_ok:
        raise RuntimeError(f"Sanity check failed. See {paths['sanity']}")

    manifest["stages"]["sanity"] = {
        "status": "ok",
        "finished_at": now_utc(),
        "artifact": paths["sanity"],
        "image": image_path,
    }
    log_event(manifest, "sanity:done", artifact=paths["sanity"], image=image_path)


def run_eval(resolved, paths, manifest):
    log_event(manifest, "eval:start")
    data = np.load(resolved["reid_dataset"])
    state_dict = torch.load(resolved["reid_model"], map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_reid_adapter_from_state_dict(state_dict, device=device)

    out = {}
    for v in data.files:
        if v.startswith("val") and v.endswith("_labels"):
            name = v[:-len("_labels")]
            labels = data[v]
            feats = data[name + "_feats"]
            out[name] = reid_eval.evaluate_recall_faiss(model, feats, labels, device=device, name=name, do_print=False)

    with open(paths["eval"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    manifest["stages"]["eval"] = {
        "status": "ok",
        "finished_at": now_utc(),
        "artifact": paths["eval"],
        "val_sets": sorted(out.keys()),
    }
    log_event(manifest, "eval:done", artifact=paths["eval"])


def main():
    parser = argparse.ArgumentParser(prog="reid_pipeline.py")
    parser.add_argument("--logging", type=str, default="info", help="Logging config: level[:console|file]")
    parser.add_argument("--base-model", type=str, required=True, help="Input non-ReID YOLO11/26 pose model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file")
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "build-dataset", "train-adapter", "fuse", "sanity", "eval"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--test", action="store_true", help="Run fast smoke mode (small dataset, short training)")
    parser.add_argument("--project", type=str, default=None, help="Override output project directory")
    parser.add_argument("--name", type=str, default=None, help="Override run name")
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Optional cap for dataset-generation worker processes (applies to config num_workers).",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun stages even if artifacts exist")
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)

    cfg = load_config(opt.config)
    if opt.project:
        cfg["project"] = opt.project
    if opt.test:
        cfg = apply_test_mode(cfg)
    if opt.max_processes is not None:
        if opt.max_processes < 1:
            raise ValueError("--max-processes must be >= 1")
        cfg["num_workers"] = min(int(cfg.get("num_workers", 1)), int(opt.max_processes))

    base_model = opt.base_model
    if not os.path.exists(base_model):
        raise FileNotFoundError(f"Base model not found: {base_model}")

    if opt.test and not opt.name:
        run_name = sanitize_name(Path(base_model).stem + "-smoke")
    else:
        run_name = opt.name

    paths = resolve_run_paths(cfg, base_model, run_name=run_name)
    resolved = build_resolved_config(cfg, base_model, paths)
    manifest = read_manifest(paths["manifest"])
    manifest["resolved_config"] = resolved
    manifest["base_model_info"] = inspect_base_model(base_model)
    manifest["mode"] = {
        "run": opt.run,
        "test": bool(opt.test),
        "max_processes": opt.max_processes,
        "force": bool(opt.force),
    }
    write_manifest(paths["manifest"], manifest)

    stage_order = ["build-dataset", "train-adapter", "fuse", "sanity", "eval"] if opt.run == "all" else [opt.run]
    artifact_for_stage = {
        "build-dataset": resolved["reid_dataset"],
        "train-adapter": resolved["reid_model"],
        "fuse": resolved["reid_yolo_model"],
        "sanity": paths["sanity"],
        "eval": paths["eval"],
    }

    for stage in stage_order:
        if not opt.force and stage_complete(manifest, stage, artifact_for_stage[stage]):
            log_event(manifest, f"{stage}:skip", reason="already complete")
            write_manifest(paths["manifest"], manifest)
            continue
        if stage == "build-dataset":
            run_build_dataset(resolved, manifest)
        elif stage == "train-adapter":
            run_train_adapter(resolved, manifest)
        elif stage == "fuse":
            run_fuse(resolved, manifest)
        elif stage == "sanity":
            run_sanity(resolved, paths, manifest)
        elif stage == "eval":
            run_eval(resolved, paths, manifest)
        write_manifest(paths["manifest"], manifest)

    print(f"Pipeline completed. Manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
