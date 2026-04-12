import argparse
import gc
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import stuff
import stuff.coord as coord
import src.reid_eval as reid_eval
import src.reid_util as ru
from ultralytics.nn.modules.head import ReIDAdapter

def _build_reid_adapter_from_state_dict(state_dict, device):
    """
    Reconstruct ReIDAdapter dimensions from saved weights.
    This avoids hardcoding legacy dims (e.g. 575/160/192/80).
    """
    film_w = state_dict["film.0.weight"]  # [2*feat_dim, 8]
    feat_dim = int(film_w.shape[0] // 2)
    in_dim = feat_dim + 8
    hidden1 = int(state_dict["mlp.0.weight"].shape[0])
    hidden2 = int(state_dict["mlp.3.weight"].shape[0])
    emb = int(state_dict["mlp.6.weight"].shape[0])
    model = ReIDAdapter(in_dim=in_dim, hidden1=hidden1, hidden2=hidden2, emb=emb).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _select_equally_spaced(indices, n):
    if n <= 0:
        return []
    if len(indices) <= n:
        return list(indices)
    picks = np.linspace(0, len(indices) - 1, n, dtype=int)
    return [indices[i] for i in sorted(set(picks.tolist()))]


def _to_numpy1d(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().astype(np.float32)
    return np.asarray(v, dtype=np.float32)


def _image_to_bgr(image_or_path):
    """
    Accept either an image path or an in-memory numpy image and return BGR uint8.
    """
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    if isinstance(image_or_path, (str, os.PathLike)):
        return cv2.imread(str(image_or_path))
    return None


def _build_compact_test_label(model_path, reid_model_pt, grid_cols, grid_rows):
    model_label = os.path.basename(model_path)
    parts = [model_label]
    if reid_model_pt:
        parts.append(f"reid:{os.path.basename(reid_model_pt)}")
    return " | ".join(parts)


def _parse_regenerate_datasets(config_value, available_datasets):
    """
    Parse config value for dataset cache invalidation.
    Supported forms:
      - list/tuple/set: ["dataset_a", "dataset_b"]
      - string: "dataset_a" or "all"/"*"
      - dict: {"datasets": [...]} or {"include": [...]}
    """
    if config_value is None:
        return []

    raw = config_value
    if isinstance(config_value, dict):
        if "datasets" in config_value:
            raw = config_value["datasets"]
        elif "include" in config_value:
            raw = config_value["include"]
        else:
            raw = []

    if isinstance(raw, str):
        raw = [raw]
    elif isinstance(raw, (tuple, set)):
        raw = list(raw)
    elif not isinstance(raw, list):
        raw = []

    wanted = []
    for item in raw:
        ds = str(item).strip()
        if not ds:
            continue
        if ds in ("*", "all", "ALL"):
            return list(available_datasets)
        wanted.append(ds)

    # keep only configured datasets and preserve original config order
    wanted_set = set(wanted)
    return [ds for ds in available_datasets if ds in wanted_set]


def _parse_model_spec(model_spec):
    """
    Parse model spec into:
    - runtime_model_spec: what inference_wrapper should receive
    - base_model_path: primary model path for naming/cache identity
    - reid_model_pt: optional adapter path when spec is model,adapter.pth
    """
    runtime_model_spec = model_spec
    base_model_path = model_spec
    reid_model_pt = None

    if "," in model_spec:
        x, y = model_spec.split(",", 1)
        y = y.strip()
        base_model_path = x
        if y.endswith(".pth") or y.endswith(".pt"):
            # Adapter form: model_path,adapter.pth
            runtime_model_spec = x
            reid_model_pt = y
        else:
            # Non-adapter extra params (e.g. trt,param_yaml) stay in runtime spec.
            runtime_model_spec = model_spec
    return runtime_model_spec, base_model_path, reid_model_pt


def _infer_camid(image_path):
    name = os.path.basename(str(image_path)).lower()
    patterns = [
        r"(?:^|[_-])c(\d+)(?:[_-]|$)",
        r"(?:^|[_-])cam(\d+)(?:[_-]|$)",
        r"(?:^|[_-])v(\d+)(?:[_-]|$)",
    ]
    for p in patterns:
        m = re.search(p, name)
        if m:
            return int(m.group(1))
    return -1


def _collect_embeddings_for_dataset(loader_name, inf, reid_model, min_conf=0.05, max_ids_per_image=16):
    loader_class = ru.get_dataset_loader(loader_name)
    loader = loader_class(task="val")
    ids = loader.get_ids()

    all_images = []
    all_ids = []
    for identity in ids:
        images = loader.get_image_paths(identity)
        images = images[:max_ids_per_image]
        all_images.extend(images)
        all_ids.extend([identity] * len(images))

    all_dets = stuff.infer_grid(inf, all_images, grid_cols=inf.grid_cols, grid_rows=inf.grid_rows, width=640, height=640)
    assert len(all_dets) == len(all_ids)

    entries = []
    for i, dets in enumerate(all_dets):
        best_vec = None
        best_score = 0.0
        for d in dets:
            if d["class"] == 0 and d["confidence"] > min_conf:
                score = coord.box_a(d["box"]) * d["confidence"]
                if score > best_score:
                    if reid_model is not None and d.get("feats") is not None:
                        best_vec = d["feats"]
                    elif d.get("reid_vector") is not None:
                        best_vec = d["reid_vector"]
                    elif d.get("feats") is not None:
                        best_vec = d["feats"]
                    else:
                        best_vec = None
                    best_score = score
        if best_vec is None:
            continue
        entries.append(
            {
                "image": all_images[i],
                "id": all_ids[i],
                "camid": _infer_camid(all_images[i]),
                "vec": _to_numpy1d(best_vec),
            }
        )
    return ids, all_images, entries


def _choose_fixed_visual_inputs(loader_name, max_ids_per_image=16, n=6):
    loader_class = ru.get_dataset_loader(loader_name)
    loader = loader_class(task="val")
    ids = loader.get_ids()
    candidates = []
    for identity in ids:
        images = loader.get_image_paths(identity)[:max_ids_per_image]
        if len(images) >= 2:
            # Only pin fixed visual queries for path-backed datasets.
            # Some loaders return in-memory numpy arrays, which are not stable/path-like.
            if isinstance(images[0], (str, os.PathLike)):
                candidates.append(str(images[0]))
    return _select_equally_spaced(candidates, n)


def _apply_adapter_if_needed(reid_model, vectors):
    if len(vectors) == 0:
        return np.empty((0, 0), dtype=np.float32)
    x = np.stack(vectors).astype(np.float32)
    if reid_model is None:
        return x
    device = next(reid_model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        emb = reid_model(xt).detach().cpu().numpy().astype(np.float32)
    return emb


def _build_query_gallery_split(entries, max_queries_per_id=1):
    by_id = {}
    for i, e in enumerate(entries):
        by_id.setdefault(e["id"], []).append(i)

    query_indices = []
    for identity in sorted(by_id.keys(), key=lambda x: str(x)):
        idxs = sorted(by_id[identity], key=lambda k: str(entries[k]["image"]))
        if len(idxs) < 2:
            continue
        q = _select_equally_spaced(idxs, max_queries_per_id)
        query_indices.extend(q)

    query_set = set(query_indices)
    gallery_indices = [i for i in range(len(entries)) if i not in query_set]
    return np.array(query_indices, dtype=np.int64), np.array(gallery_indices, dtype=np.int64)


def _add_grid_mean_rows(rows):
    """
    Add display-only rows with grid='mean' that average metrics across grid variants
    for the same dataset + compact test label.
    """
    base_rows = [r for r in rows if str(r.get("grid", "")).lower() != "mean"]
    grouped = {}
    for row in base_rows:
        key = (row.get("dataset", ""), row.get("test_short", ""))
        grouped.setdefault(key, []).append(row)

    mean_rows = []
    for (dataset_name, test_short), group_rows in grouped.items():
        if len(group_rows) < 2:
            continue
        numeric_keys = sorted(
            {
                k
                for row in group_rows
                for k, v in row.items()
                if isinstance(v, float)
            }
        )
        mean_row = {
            "dataset": dataset_name,
            "test_short": test_short,
            "test": test_short,
            "grid": "mean",
        }
        for key in numeric_keys:
            vals = [row[key] for row in group_rows if isinstance(row.get(key), float)]
            mean_row[key] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        mean_row["vis_dir"] = ""
        mean_rows.append(mean_row)
    return rows + mean_rows


def _save_query_visuals(
    out_dir,
    dataset_name,
    test_name,
    entries,
    embeddings,
    query_indices,
    gallery_indices,
    labels,
    topk=10,
    num_visual_queries=6,
    fixed_query_images=None,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(query_indices) == 0 or len(gallery_indices) == 0:
        return []

    emb = embeddings.astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    q_emb = emb[query_indices]
    g_emb = emb[gallery_indices]
    sims = q_emb @ g_emb.T

    chosen_q_pos = []
    if fixed_query_images:
        q_by_img = {str(entries[int(qi)]["image"]): pos for pos, qi in enumerate(query_indices)}
        for imgp in fixed_query_images:
            pos = q_by_img.get(str(imgp))
            if pos is not None:
                chosen_q_pos.append(pos)
    if len(chosen_q_pos) == 0:
        chosen_q_pos = _select_equally_spaced(list(range(len(query_indices))), num_visual_queries)
    if len(chosen_q_pos) > num_visual_queries:
        chosen_q_pos = chosen_q_pos[:num_visual_queries]
    saved = []
    cell_w, cell_h = 224, 224
    cols = 4
    max_cells = 1 + topk
    rows = int(np.ceil(max_cells / cols))

    for qpos in chosen_q_pos:
        q_global = int(query_indices[qpos])
        q_entry = entries[q_global]
        order = np.argsort(-sims[qpos])
        best_g = [int(gallery_indices[j]) for j in order[:topk]]

        tiles = []
        q_img = _image_to_bgr(q_entry["image"])
        if q_img is None:
            q_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        q_img = cv2.resize(q_img, (cell_w, cell_h))
        cv2.putText(q_img, f"QUERY id:{q_entry['id']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(q_img)

        for rank, gidx in enumerate(best_g, start=1):
            g = entries[gidx]
            img = _image_to_bgr(g["image"])
            if img is None:
                img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            img = cv2.resize(img, (cell_w, cell_h))
            tp = labels[gidx] == labels[q_global]
            clr = (0, 255, 0) if tp else (255, 255, 255)
            sim_val = float(sims[qpos, np.where(gallery_indices == gidx)[0][0]])
            cv2.putText(img, f"R{rank} id:{g['id']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1, cv2.LINE_AA)
            cv2.putText(img, f"sim:{sim_val:.3f}", (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(img)

        while len(tiles) < rows * cols:
            tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row = np.hstack(tiles[r * cols : (r + 1) * cols])
            grid_rows.append(row)
        sheet = np.vstack(grid_rows)

        stem = _sanitize_name(f"{dataset_name}_{test_name}_q{q_global}")
        out_path = os.path.join(out_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, sheet)
        saved.append(out_path)
    return saved


def test(model_spec, grid_cols=5, grid_rows=4, min_conf=0.05, rc=None, datasets=None, visual_root=None):
    runtime_model_spec, base_model_path, reid_model_pt = _parse_model_spec(model_spec)

    reid_model = None
    inf = None
    try:
        model_name = os.path.basename(base_model_path)
        grid_label = f"{grid_cols}x{grid_rows}"
        test_name = f"{model_name} std-qg-v1 grid:{grid_cols}x{grid_rows}"
        if reid_model_pt is not None:
            test_name += f" {os.path.basename(reid_model_pt)}"
        test_short = _build_compact_test_label(base_model_path, reid_model_pt, grid_cols, grid_rows)

        def ensure_models_loaded():
            nonlocal reid_model, inf
            if inf is not None:
                return
            if reid_model_pt is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                state_dict = torch.load(reid_model_pt, map_location="cpu")
                reid_model = _build_reid_adapter_from_state_dict(state_dict, device)
            inf = stuff.inference_wrapper(runtime_model_spec, thr=min_conf, get_feats=True, fold_attributes=True)
            inf.grid_cols = grid_cols
            inf.grid_rows = grid_rows

        results = []
        for loader_name in datasets:
            result = rc.get({"dataset": loader_name, "test": test_name})
            if result is None:
                ensure_models_loaded()
                fixed_visual_inputs = _choose_fixed_visual_inputs(loader_name, max_ids_per_image=16, n=6)
                ids, all_images, entries = _collect_embeddings_for_dataset(
                    loader_name=loader_name,
                    inf=inf,
                    reid_model=reid_model,
                    min_conf=min_conf,
                    max_ids_per_image=16,
                )

                result = {
                    "dataset": loader_name,
                    "test": test_name,
                    "test_short": test_short,
                    "grid": grid_label,
                    "num_ids": float(len(ids)),
                    "num_img": float(len(all_images)),
                    "num_missed": float(len(all_images) - len(entries)),
                    "num_emb": float(len(entries)),
                    "num_query_total": 0.0,
                    "num_query_valid": 0.0,
                    "Rank-1": 0.0,
                    "Rank-5": 0.0,
                    "Rank-10": 0.0,
                    "Rank-20": 0.0,
                    "mAP": 0.0,
                }

                if len(entries) > 0:
                    vectors = [e["vec"] for e in entries]
                    labels = np.array([e["id"] for e in entries])
                    camids = np.array([e["camid"] for e in entries], dtype=np.int64)
                    embeddings = _apply_adapter_if_needed(reid_model, vectors)

                    query_idx, gallery_idx = _build_query_gallery_split(entries, max_queries_per_id=1)
                    if np.any(camids < 0):
                        q_cam = None
                        g_cam = None
                    else:
                        q_cam = camids[query_idx] if len(query_idx) > 0 else None
                        g_cam = camids[gallery_idx] if len(gallery_idx) > 0 else None

                    metrics = reid_eval.evaluate_standard_reid(
                        embeddings=embeddings,
                        labels=labels,
                        query_indices=query_idx,
                        gallery_indices=gallery_idx,
                        query_camids=q_cam,
                        gallery_camids=g_cam,
                        ks=(1, 5, 10, 20),
                    )
                    result.update(metrics)

                    vis_saved = []
                    if visual_root:
                        vis_dir = os.path.join(visual_root, _sanitize_name(test_name), loader_name)
                        vis_saved = _save_query_visuals(
                            out_dir=vis_dir,
                            dataset_name=loader_name,
                            test_name=test_name,
                            entries=entries,
                            embeddings=embeddings,
                            query_indices=query_idx,
                            gallery_indices=gallery_idx,
                            labels=labels,
                            topk=10,
                            num_visual_queries=6,
                            fixed_query_images=fixed_visual_inputs,
                        )
                        result["vis_dir"] = vis_dir
                        result["vis_count"] = float(len(vis_saved))
                        result["vis_inputs"] = ",".join([os.path.basename(p) for p in fixed_visual_inputs[:6]])
                    else:
                        result["vis_dir"] = ""
                        result["vis_count"] = 0.0
                        result["vis_inputs"] = ",".join([os.path.basename(p) for p in fixed_visual_inputs[:6]])
                else:
                    result["vis_dir"] = ""
                    result["vis_count"] = 0.0
                    result["vis_inputs"] = ",".join([os.path.basename(p) for p in fixed_visual_inputs[:6]])

                rc.add(result)
            # Keep display-only fields in sync even for older cached metric rows.
            result["test_short"] = test_short
            result["grid"] = grid_label
            results.append(result)

        if len(results) == 0:
            return results

        mean_row = {"dataset": "Mean", "test": test_name, "test_short": test_short, "grid": grid_label}
        numeric_keys = [k for k, v in results[0].items() if isinstance(v, float)]
        for k in numeric_keys:
            vals = [r[k] for r in results]
            mean_row[k] = float(np.mean(vals)) if len(vals) > 0 else 0.0
        mean_row["vis_dir"] = ""
        results.append(mean_row)
        return results
    finally:
        # Explicitly release per-model resources to avoid cumulative GPU memory growth across model loops.
        del inf
        del reid_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='reid_dpar_test.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default='/mldata/config/reid/reid_test.yaml', help='config file to use (json or yml)')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)

    config=stuff.load_dictionary(opt.config)
    models=config["models"]
    datasets=config["datasets"]
    results_cache_file=config["results_cache_file"]
    visual_root = config.get("visual_output_dir", "/mldata/results/reid/query_gallery_visuals")
    regenerate_datasets = _parse_regenerate_datasets(config.get("regenerate_datasets"), datasets)

    rc=stuff.ResultCache(results_cache_file)
    if len(regenerate_datasets) > 0:
        total_deleted = 0
        for dataset_name in regenerate_datasets:
            total_deleted += rc.delete({"dataset": dataset_name})
        print(
            f"Regenerated datasets requested: {', '.join(regenerate_datasets)} "
            f"(removed {total_deleted} cached rows)"
        )

    grids=[tuple(x) for x in config["grids"]]
    dataset_order = {name: idx for idx, name in enumerate(datasets)}

    results=[]

    def sort_fn(r):
        ds = r.get("dataset", "")
        if ds == "Mean":
            # show_data sorts with reverse=True, so very small rank pushes Mean to bottom
            ds_rank = -1_000_000
        else:
            # Keep configured dataset order (first in config appears first in table)
            ds_rank = 1_000_000 - dataset_order.get(ds, 1_000_000)
        return ds_rank + float(r.get("mAP", 0.0))

    for grid_w,grid_h in grids:
        for model in models:
            results += test(model, grid_w, grid_h, rc=rc, datasets=datasets, visual_root=visual_root)
            display_results = _add_grid_mean_rows(results)

            stuff.show_data(display_results,
                            columns=["dataset","test_short","grid","num_ids","num_img","num_missed","num_emb","num_query_total","num_query_valid","Rank-1","Rank-5","Rank-10","Rank-20","mAP","vis_count"],
                            column_text=["dataset","test","grid","Num id","Num Img","Missed","Emb","Q total","Q valid","Rank-1","Rank-5","Rank-10","Rank-20","mAP","Vis"],
                            sort_fn=sort_fn)