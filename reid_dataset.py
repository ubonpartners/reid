import argparse
import stuff
import src.reid_util as ru
import src.reid_model as reid_model
import src.reid_eval as reid_eval
import numpy as np
from functools import partial
from ultralytics import YOLO
from ultralytics.nn.modules.head import build_reid_adapter_from_state_dict
from tqdm import tqdm
import torch
import json
import random
import cv2
import os
from pathlib import Path

try:
    import albumentations as A
except Exception:
    A = None


_AUG_TRANSFORM_CACHE = {}


def _build_albumentations_transform(set_name):
    key = (set_name or "none").strip().lower()
    if key in _AUG_TRANSFORM_CACHE:
        return _AUG_TRANSFORM_CACHE[key]
    if A is None:
        _AUG_TRANSFORM_CACHE[key] = None
        return None

    if key in ("none", "off", "disabled"):
        transforms = []
    elif key == "aggressive_motion":
        transforms = [
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.25),

        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.15,
            p=0.2
        ),
        A.RandomGamma(gamma_limit=(90, 115), p=0.1),
        A.HueSaturationValue(
            hue_shift_limit=6,
            sat_shift_limit=10,
            val_shift_limit=8,
            p=0.1
        ),
        A.ISONoise(
            color_shift=(0.01, 0.02),
            intensity=(0.03, 0.08),
            p=0.05
        ),
        A.ImageCompression(quality_range=(50, 95), p=0.15),
        A.RandomRain(p=0.02),
        A.ToGray(p=0.02),
    ]
    else:
        transforms = [
            A.Blur(p=0.05),
            A.MedianBlur(p=0.05),
            A.MotionBlur(p=0.08),
            A.ToGray(p=0.02),
            A.CLAHE(p=0.03),
            A.RandomBrightnessContrast(p=0.08),
            A.RandomGamma(p=0.05),
            A.ISONoise(p=0.1),
            A.ChromaticAberration(p=0.02),
            A.ImageCompression(quality_range=(30, 100), p=0.1),
        ]
    _AUG_TRANSFORM_CACHE[key] = A.Compose(transforms) if transforms else None
    return _AUG_TRANSFORM_CACHE[key]


def _apply_post_aug_image(img, effect_prob, albumentations_set):
    if effect_prob <= 0 or random.random() >= effect_prob:
        return img
    transform = _build_albumentations_transform(albumentations_set)
    if transform is None:
        return img
    img_np = np.asarray(img)
    return transform(image=img_np)["image"]


def _coerce_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _apply_grid_random_erasing(img, grid_boxes, aug_cfg):
    prob = _clamp(_coerce_float(aug_cfg.get("random_erasing_prob", 0.0), 0.0), 0.0, 1.0)
    if prob <= 0.0 or random.random() >= prob:
        return img
    if grid_boxes is None or len(grid_boxes) == 0:
        return img

    per_box_prob = _clamp(_coerce_float(aug_cfg.get("random_erasing_per_box_prob", 0.35), 0.35), 0.0, 1.0)
    min_area = _clamp(_coerce_float(aug_cfg.get("random_erasing_min_area", 0.02), 0.02), 0.0001, 1.0)
    max_area = _clamp(_coerce_float(aug_cfg.get("random_erasing_max_area", 0.12), 0.12), min_area, 1.0)
    min_aspect = _clamp(_coerce_float(aug_cfg.get("random_erasing_min_aspect", 0.5), 0.5), 0.1, 10.0)
    max_aspect = _clamp(_coerce_float(aug_cfg.get("random_erasing_max_aspect", 2.0), 2.0), min_aspect, 10.0)
    max_regions = max(1, _coerce_int(aug_cfg.get("random_erasing_max_regions", 1), 1))
    max_attempts = max(1, _coerce_int(aug_cfg.get("random_erasing_max_attempts", 10), 10))
    min_side = max(2, _coerce_int(aug_cfg.get("random_erasing_min_side", 8), 8))
    fill_mode = str(aug_cfg.get("random_erasing_fill_mode", "mean")).strip().lower()
    if fill_mode not in {"mean", "random", "constant", "black"}:
        fill_mode = "mean"
    fill_value = _clamp(_coerce_float(aug_cfg.get("random_erasing_fill_value", 0), 0), 0.0, 255.0)

    img_np = np.asarray(img).copy()
    if img_np.ndim < 2:
        return img

    height, width = img_np.shape[:2]
    if height < min_side or width < min_side:
        return img

    def _draw_patch(er_h, er_w):
        if fill_mode == "mean":
            return None
        if fill_mode == "constant":
            if img_np.ndim == 2:
                return np.array(fill_value, dtype=img_np.dtype)
            return np.full((er_h, er_w, img_np.shape[2]), fill_value, dtype=img_np.dtype)
        if fill_mode == "black":
            if img_np.ndim == 2:
                return np.array(0, dtype=img_np.dtype)
            return np.zeros((er_h, er_w, img_np.shape[2]), dtype=img_np.dtype)
        if img_np.ndim == 2:
            patch = np.random.randint(0, 256, size=(er_h, er_w), dtype=np.uint8)
        else:
            patch = np.random.randint(0, 256, size=(er_h, er_w, img_np.shape[2]), dtype=np.uint8)
        if img_np.dtype != np.uint8:
            patch = patch.astype(img_np.dtype)
        return patch

    applied = False
    for box in grid_boxes:
        if box is None or len(box) != 4:
            continue
        if random.random() >= per_box_prob:
            continue

        x1 = _clamp(int(round(float(box[0]) * width)), 0, width - 1)
        y1 = _clamp(int(round(float(box[1]) * height)), 0, height - 1)
        x2 = _clamp(int(round(float(box[2]) * width)), 0, width)
        y2 = _clamp(int(round(float(box[3]) * height)), 0, height)
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_side or bh < min_side:
            continue

        box_area = float(bw * bh)
        for _ in range(max_regions):
            placed = False
            for _ in range(max_attempts):
                target_area = random.uniform(min_area, max_area) * box_area
                aspect = random.uniform(min_aspect, max_aspect)
                er_w = int(round(np.sqrt(target_area * aspect)))
                er_h = int(round(np.sqrt(target_area / max(aspect, 1e-6))))
                if er_w < min_side or er_h < min_side:
                    continue
                if er_w >= bw or er_h >= bh:
                    continue

                ex1 = random.randint(x1, x2 - er_w)
                ey1 = random.randint(y1, y2 - er_h)
                ex2 = ex1 + er_w
                ey2 = ey1 + er_h

                fill_patch = _draw_patch(er_h, er_w)
                if fill_patch is None:
                    source = img_np[y1:y2, x1:x2]
                    if source.size == 0:
                        continue
                    if img_np.ndim == 2:
                        fill_patch = np.array(source.mean(), dtype=img_np.dtype)
                    else:
                        fill_patch = source.reshape(-1, source.shape[2]).mean(axis=0).astype(img_np.dtype)

                img_np[ey1:ey2, ex1:ex2] = fill_patch
                applied = True
                placed = True
                break
            if not placed:
                break

    if not applied:
        return img
    return img_np


def _normalize_aug_policy(policy):
    if not isinstance(policy, dict):
        return {}
    out = dict(policy)
    if "augmentations" in out and "num_aug" not in out:
        out["num_aug"] = out["augmentations"]
    if "aug_rotate" in out and "rotate_prob" not in out:
        out["rotate_prob"] = out["aug_rotate"]
    if "aug_effects" in out and "effect_prob" not in out:
        out["effect_prob"] = out["aug_effects"]
    if "aug_random_erasing" in out and "random_erasing_prob" not in out:
        out["random_erasing_prob"] = out["aug_random_erasing"]
    if "erase_prob" in out and "random_erasing_prob" not in out:
        out["random_erasing_prob"] = out["erase_prob"]
    if "random_erasing" in out:
        re_cfg = out["random_erasing"]
        if isinstance(re_cfg, dict):
            re_aliases = {
                "prob": "random_erasing_prob",
                "per_box_prob": "random_erasing_per_box_prob",
                "min_area": "random_erasing_min_area",
                "max_area": "random_erasing_max_area",
                "min_aspect": "random_erasing_min_aspect",
                "max_aspect": "random_erasing_max_aspect",
                "max_regions": "random_erasing_max_regions",
                "max_attempts": "random_erasing_max_attempts",
                "min_side": "random_erasing_min_side",
                "fill_mode": "random_erasing_fill_mode",
                "fill_value": "random_erasing_fill_value",
            }
            for src_key, dst_key in re_aliases.items():
                if src_key in re_cfg and dst_key not in out:
                    out[dst_key] = re_cfg[src_key]
        elif "random_erasing_prob" not in out:
            out["random_erasing_prob"] = re_cfg
    area_pair = out.get("random_erasing_area")
    if isinstance(area_pair, (list, tuple)) and len(area_pair) == 2:
        if "random_erasing_min_area" not in out:
            out["random_erasing_min_area"] = area_pair[0]
        if "random_erasing_max_area" not in out:
            out["random_erasing_max_area"] = area_pair[1]
    return out


def _canonical_loader_token(name):
    s = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if s.endswith("loader"):
        s = s[:-len("loader")]
    return s

def _ioma_score_matrix(grid_boxes, det_boxes, det_conf):
    """
    Vectorized IOMA * confidence scores.
    grid_boxes: [G,4] xyxy
    det_boxes:  [D,4] xyxy
    det_conf:   [D]
    returns:    [G,D]
    """
    if grid_boxes.size == 0 or det_boxes.size == 0:
        return np.empty((grid_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)

    gx1, gy1, gx2, gy2 = grid_boxes[:, 0:1], grid_boxes[:, 1:2], grid_boxes[:, 2:3], grid_boxes[:, 3:4]
    dx1, dy1, dx2, dy2 = det_boxes[None, :, 0], det_boxes[None, :, 1], det_boxes[None, :, 2], det_boxes[None, :, 3]

    iw = np.maximum(0.0, np.minimum(gx2, dx2) - np.maximum(gx1, dx1))
    ih = np.maximum(0.0, np.minimum(gy2, dy2) - np.maximum(gy1, dy1))
    inter = iw * ih

    g_area = np.maximum(0.0, (gx2 - gx1) * (gy2 - gy1))
    d_area = np.maximum(0.0, (dx2 - dx1) * (dy2 - dy1))
    min_area = np.minimum(g_area, d_area) + 1e-7
    ioma = inter / min_area
    return ioma * det_conf[None, :]

def select_equally_spaced(lst, N):
    if N <= 0:
        return []
    if N >= len(lst):
        return lst.copy()

    # Generate N equally spaced indices
    indices = np.linspace(0, len(lst) - 1, N, dtype=int)
    unique_indices = sorted(set(indices))  # enforce uniqueness and sorted order
    return [lst[i] for i in unique_indices]

def loader_get_images(loader_class, task, max_ids_per_image, max_ids_per_loader=None):
    loader=loader_class(task=task)
    ids=loader.get_ids()
    if max_ids_per_loader is not None and max_ids_per_loader > 0:
        ids = select_equally_spaced(ids, max_ids_per_loader)
    ret=[]
    total_images=0
    for id in ids:
        images=loader.get_image_paths(id)
        if images==None or len(images)==0:
            continue
        images=select_equally_spaced(images, max_ids_per_image)
        ret.append(images)
        total_images+=len(images)
    return ret, loader.get_name(), total_images

def get_dataset_images(config, tasks=["train","val"]):
    datasets_config=config["datasets"]

    max_ids_per_image=8
    if "max_ids_per_image" in datasets_config:
        max_ids_per_image=datasets_config["max_ids_per_image"]
    max_ids_per_loader = datasets_config.get("max_ids_per_loader")

    datasets={}

    train=[]
    val=[]
    total_train_images=0
    for d in datasets_config["loaders"]:
        loader_class=ru.get_dataset_loader(d)

        if "train" in tasks:
            this_train, train_name, this_train_num=loader_get_images(
                loader_class, "train", max_ids_per_image, max_ids_per_loader=max_ids_per_loader
            )
            train+=this_train
            total_train_images+=this_train_num
            datasets["train_"+train_name]=this_train

        if "val" in tasks:
            valset, name, _=loader_get_images(
                loader_class, "val", max_ids_per_image, max_ids_per_loader=max_ids_per_loader
            )
            valset_partial=select_equally_spaced(valset, max(100, len(valset)//4))
            val+=valset_partial
            datasets["val_"+name]=valset

    if "train" in tasks:
        datasets["train"]=train
    if "val" in tasks:
        datasets["val"]=val

    print("Import results-")
    for d in datasets:
        print(f"    Dataset {d:30s} {len(datasets[d]):5d} IDs {sum([len(x) for x in datasets[d]]):6d} images")

    return datasets

def on_predict_start(predictor: object, persist: bool = False) -> None:
    # Expanded feature extraction relies on kept-anchor indices from NMS; this is
    # unavailable/incompatible in end-to-end decode paths.
    is_end2end = bool(getattr(getattr(predictor, "model", None), "end2end", False))
    if is_end2end:
        predictor.save_feats = False
        predictor.expanded_feats = False
        predictor._feats = None
        return

    predictor.save_feats = True
    predictor.expanded_feats = True
    predictor._feats = None

    # Install hooks only once per predictor instance.
    if getattr(predictor, "_feat_hooks_installed", False):
        return

    # Register hooks to extract input and output of Detect layer
    def pre_hook(module, input):
        # Keep detached references; cloning these large feature maps is expensive.
        predictor._feats = [t.detach() for t in input[0]]

    predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)
    predictor._feat_hooks_installed = True

def get_feat_process_batch(batch, model, img_size, person_class_index=0):
    images=[b["img"] for b in batch]
    results=model.predict(images, verbose=False, half=True,
                          conf=0.1, classes=[person_class_index],
                          imgsz=[img_size,img_size], rect=True,
                          end2end=False)
    ret_feats=[]
    ret_ids=[]
    ret_idx=[]
    for i, result in enumerate(results):
        grid_boxes=batch[i]["grid_boxes"]
        id_list=batch[i]["id_list"]
        idx_list=batch[i]["idx_list"]
        num=batch[i]["num"]
        det_boxes = result.boxes.xyxyn.detach().cpu().numpy()  # [D,4] normalized xyxy
        det_confidences = result.boxes.conf.detach().cpu().numpy()  # [D]
        if model.task=="posereid":
            det_feats=result.reid_embeddings
        else:
            det_feats=result.feats
        if det_feats is None or len(det_feats) == 0 or det_boxes.shape[0] == 0:
            continue

        gboxes = np.asarray(grid_boxes, dtype=np.float32)
        scores = _ioma_score_matrix(gboxes, det_boxes.astype(np.float32), det_confidences.astype(np.float32))
        if scores.size == 0:
            continue

        best_det_idx = scores.argmax(axis=1)  # [num]
        best_scores = scores[np.arange(scores.shape[0]), best_det_idx]

        best_feats=[None]*num
        for gn in range(min(num, scores.shape[0])):
            if best_scores[gn] > 0.1:
                best_feats[gn]=det_feats[int(best_det_idx[gn])]
        for j in range(num):
            if best_feats[j]!=None:
                ret_feats.append(best_feats[j])
                ret_ids.append(id_list[j])
                ret_idx.append(idx_list[j])
    return ret_feats, ret_ids, ret_idx

def make_feat_process_batch_work(model, batch_work):
    image_batch=[]
    for w in batch_work:
        img, grid_boxes=stuff.create_image_grid(w["imgs"], w["M"], w["N"],
                                             w["img_size"], w["img_size"],
                                             max_random_shrink=0.5,
                                             aug_rotate=w["aug_rotate"],
                                             aug_effects=0)
        img = _apply_grid_random_erasing(img, grid_boxes, w)
        img = _apply_post_aug_image(
            img,
            effect_prob=float(w.get("aug_effect_prob", 0)),
            albumentations_set=w.get("albumentations_set", "standard"),
        )
        image_batch.append({"img":img,
                            "grid_boxes":grid_boxes,
                            "id_list":w["ids"],
                            "idx_list":w["idxs"],
                            "num":len(w["ids"])})
    feats, ids, idxs=get_feat_process_batch(image_batch, model, w["img_size"])
    return {"feats":feats, "ids":ids, "idxs": idxs}

def work_queue_fn(work_item, mpwq_context, mpwq_progress_fn=None):
    model=mpwq_context["process_setup_results"]
    return make_feat_process_batch_work(model, work_item)

def make_model(args):
    # each multiprocess worker needs to create it's own yolo model instance
    model=YOLO(args,  verbose=False)
    if model.task!="posereid":
        model.add_callback("on_predict_start", partial(on_predict_start, persist=False))
    return model

def get_feats(img_list, id_list_in, index_list_in, model_name,
              name, batch_size=64, num_aug=0,
              aug_rotate=False, aug_effect_prob=0, albumentations_set="standard",
              random_erasing_prob=0.0,
              random_erasing_per_box_prob=0.35,
              random_erasing_min_area=0.02,
              random_erasing_max_area=0.12,
              random_erasing_min_aspect=0.5,
              random_erasing_max_aspect=2.0,
              random_erasing_max_regions=1,
              random_erasing_max_attempts=10,
              random_erasing_min_side=8,
              random_erasing_fill_mode="mean",
              random_erasing_fill_value=0,
              num_workers=1, num_classes=40,
              debug_feature_stats=False,
              mp_min_work_items=6,
              force_multiprocess=True):

    # step 1: replicate the lists so that each item appears multiple times
    # each instance will get different random scaling/augmentation

    img_list_replicated = [img for img in img_list for _ in range(num_aug+1)]
    id_list_replicated = [id_ for id_ in id_list_in for _ in range(num_aug+1)]
    index_list_replicated = [idx_ for idx_ in index_list_in for _ in range(num_aug+1)]
    if len(img_list_replicated) == 0:
        return [], [], [], "Scales [0, 0, 0]"
    combined = list(zip(img_list_replicated, id_list_replicated, index_list_replicated))
    random.shuffle(combined)
    img_list_shuffled, id_list_shuffled, index_list_shuffled = zip(*combined)
    img_list_shuffled = list(img_list_shuffled)
    id_list_shuffled = list(id_list_shuffled)
    index_list_shuffled = list(index_list_shuffled)

    # step 2: run through all the reid images and split them in to groups to
    # be put on random "grid images", and batch these description up into
    # suitable batch sizes. We don't actually do any real work yet

    feat_list = []
    id_list=[]
    idx_list=[]
    through=0
    pbar = tqdm(total=len(img_list_shuffled), leave=False)
    img_size=640
    work_items=[]
    batch_work=[]

    while through<len(img_list_shuffled):
        items=  [(2,1),(3,1),(3,2),(4,2),(4,3),(5,3),(6,3),(5,4),(7,4),(8,5)]
        weights=[   8,    4,    3,    2,    2,    2,    2,    2,    1,    1 ]
        N,M=random.choices(items, weights=weights, k=1)[0]
        imgs=img_list_shuffled[through:through+N*M]
        ids=id_list_shuffled[through:through+N*M]
        idxs=index_list_shuffled[through:through+N*M]
        batch_work_item={"imgs":imgs, "ids":ids, "idxs": idxs, "N":N, "M":M,
                         "img_size":img_size,
                         "aug_rotate":aug_rotate,
                         "aug_effect_prob":aug_effect_prob,
                         "albumentations_set":albumentations_set,
                         "random_erasing_prob":random_erasing_prob,
                         "random_erasing_per_box_prob":random_erasing_per_box_prob,
                         "random_erasing_min_area":random_erasing_min_area,
                         "random_erasing_max_area":random_erasing_max_area,
                         "random_erasing_min_aspect":random_erasing_min_aspect,
                         "random_erasing_max_aspect":random_erasing_max_aspect,
                         "random_erasing_max_regions":random_erasing_max_regions,
                         "random_erasing_max_attempts":random_erasing_max_attempts,
                         "random_erasing_min_side":random_erasing_min_side,
                         "random_erasing_fill_mode":random_erasing_fill_mode,
                         "random_erasing_fill_value":random_erasing_fill_value}
        batch_work.append(batch_work_item)
        if len(batch_work)>=batch_size:
            work_items.append(batch_work)
            batch_work=[]

        pbar.update(len(imgs))
        through+=len(imgs)
    # Append any remainder work (otherwise the last partial batch is dropped).
    if len(batch_work) > 0:
        img_size=random.choice([512, 640, 704, 768, 832, 864, 960])
        for item in batch_work:
            item["img_size"] = img_size
        work_items.append(batch_work)
        batch_work=[]

    pbar.close()

    results=[]

    make_model_args=model_name

    # step 3: do the actual work, each 'work item' is a batch
    # of grid images. The work involves loading the reid images,
    # rendering them to the grids, then running the yolo model
    # and collecting the resulting embeddings.
    # - run multiprocess-parallel if lots of work to do
    # it takes ~2 hours to generate 1 million images on my 3090
    # but only ~20 mins with 8 processes

    if num_workers > 1 and (force_multiprocess or len(work_items) >= mp_min_work_items):
        results=stuff.mp_workqueue_run(work_items, work_queue_fn,
                               desc=f"multiprocess:{name}",
                               num_workers=num_workers,
                               process_setup_fn=make_model,
                               process_setup_args=make_model_args,
                               show_pbars=False)
    else:
        model=make_model(make_model_args)
        for i,batch_work in tqdm(enumerate(work_items), total=len(work_items), leave=False):
            results.append(make_feat_process_batch_work(model, batch_work))

    # finally combine the results together

    for r in results:
        feat_list+=r["feats"]
        id_list+=r["ids"]
        idx_list+=r["idxs"]

    debug = "Scales [disabled]"
    if debug_feature_stats:
        # Optional expensive debug path.
        sizes=[0,0,0]
        nz=[0,0,0]
        for f in feat_list:
            l=f.cpu().to(torch.int).tolist()
            for i in range(3):
                if l[512+num_classes+i]!=0:
                    nz[i]+=l.count(0)
                sizes[i]+=l[512+num_classes+i]
        for i in range(3):
            nz[i]/=(sizes[i]+0.00001)
        debug=f"Scales {sizes}"

    return feat_list, id_list, idx_list, debug


def _resolve_train_loader_policies(config):
    datasets_cfg = config.get("datasets", {})
    loader_names = datasets_cfg.get("loaders", [])

    global_defaults = {
        "num_aug": int(config.get("train_augmentations", 3)),
        "rotate_prob": float(config.get("train_aug_rotate", 0)),
        "effect_prob": float(config.get("train_aug_effects", 0)),
        "albumentations_set": str(config.get("train_albumentations_set", "standard")),
        "random_erasing_prob": float(config.get("train_random_erasing_prob", 0.0)),
        "random_erasing_per_box_prob": float(config.get("train_random_erasing_per_box_prob", 0.35)),
        "random_erasing_min_area": float(config.get("train_random_erasing_min_area", 0.02)),
        "random_erasing_max_area": float(config.get("train_random_erasing_max_area", 0.12)),
        "random_erasing_min_aspect": float(config.get("train_random_erasing_min_aspect", 0.5)),
        "random_erasing_max_aspect": float(config.get("train_random_erasing_max_aspect", 2.0)),
        "random_erasing_max_regions": int(config.get("train_random_erasing_max_regions", 1)),
        "random_erasing_max_attempts": int(config.get("train_random_erasing_max_attempts", 10)),
        "random_erasing_min_side": int(config.get("train_random_erasing_min_side", 8)),
        "random_erasing_fill_mode": str(config.get("train_random_erasing_fill_mode", "mean")),
        "random_erasing_fill_value": float(config.get("train_random_erasing_fill_value", 0)),
    }

    policies = {}
    for loader_name in loader_names:
        loader_class = ru.get_dataset_loader(loader_name)
        defaults = dict(global_defaults)
        if hasattr(loader_class, "get_train_augmentation_defaults"):
            loader_defaults = loader_class.get_train_augmentation_defaults()
            if isinstance(loader_defaults, dict):
                defaults.update(_normalize_aug_policy(loader_defaults))
        policies[_canonical_loader_token(loader_name)] = defaults

    override_map = config.get("train_loader_augmentations")
    if not isinstance(override_map, dict):
        override_map = datasets_cfg.get("loader_augmentations", {})
    if isinstance(override_map, dict):
        for name, overrides in override_map.items():
            if not isinstance(overrides, dict):
                continue
            key = _canonical_loader_token(name)
            if key not in policies:
                policies[key] = dict(global_defaults)
            policies[key].update(_normalize_aug_policy(overrides))
    return policies, global_defaults


def _extract_loader_key_from_dataset_name(dataset_name):
    if dataset_name.startswith("train_"):
        return _canonical_loader_token(dataset_name[len("train_"):])
    return None


def _generate_aug_preview(datasets, dataset_name, aug_cfg, preview_dir, sample_count=12):
    if not preview_dir or sample_count <= 0:
        return
    if dataset_name not in datasets:
        return
    identities = datasets[dataset_name]
    if len(identities) == 0:
        return

    os.makedirs(preview_dir, exist_ok=True)
    items = [(2, 1), (3, 1), (3, 2), (4, 2), (4, 3), (5, 3), (6, 3), (5, 4), (7, 4), (8, 5)]
    saved = 0
    attempts = 0
    max_attempts = max(sample_count * 10, 20)
    while saved < sample_count and attempts < max_attempts:
        attempts += 1
        N, M = random.choice(items)
        need = N * M
        selected_ids = random.choices(identities, k=need)
        imgs = []
        for identity_imgs in selected_ids:
            if not identity_imgs:
                continue
            imgs.append(random.choice(identity_imgs))
        if len(imgs) == 0:
            continue
        img_size = random.choice([512, 640, 704, 768, 832, 864, 960])
        canvas, grid_boxes = stuff.create_image_grid(
            imgs,
            M,
            N,
            img_size,
            img_size,
            max_random_shrink=0.5,
            aug_rotate=float(aug_cfg.get("rotate_prob", 0)),
            aug_effects=0,
        )
        canvas = _apply_grid_random_erasing(canvas, grid_boxes, aug_cfg)
        canvas = _apply_post_aug_image(
            canvas,
            effect_prob=float(aug_cfg.get("effect_prob", 0)),
            albumentations_set=aug_cfg.get("albumentations_set", "standard"),
        )
        out = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        out_name = f"{dataset_name}_{saved:03d}.jpg"
        cv2.imwrite(str(Path(preview_dir) / out_name), out)
        saved += 1


def _run_dataset_feature_extract(dataset_name, dataset_groups, config, num_classes,
                                 batch_size, debug_feature_stats, mp_min_work_items,
                                 force_multiprocess, aug_cfg):
    id_list = []
    image_list = []
    index = 0
    for identity_imgs in dataset_groups:
        id_list += [index] * len(identity_imgs)
        image_list += identity_imgs
        index += 1

    num_workers = int(config.get("num_workers", 4))
    num_original_images = len(image_list)
    img_indices = list(range(len(image_list)))
    feat_list, id_list, img_indices, dbg = get_feats(
        image_list,
        id_list,
        img_indices,
        config["yolo_model"],
        name=dataset_name,
        batch_size=batch_size,
        num_aug=int(aug_cfg.get("num_aug", 0)),
        aug_rotate=float(aug_cfg.get("rotate_prob", 0)),
        aug_effect_prob=float(aug_cfg.get("effect_prob", 0)),
        albumentations_set=aug_cfg.get("albumentations_set", "standard"),
        random_erasing_prob=float(aug_cfg.get("random_erasing_prob", 0.0)),
        random_erasing_per_box_prob=float(aug_cfg.get("random_erasing_per_box_prob", 0.35)),
        random_erasing_min_area=float(aug_cfg.get("random_erasing_min_area", 0.02)),
        random_erasing_max_area=float(aug_cfg.get("random_erasing_max_area", 0.12)),
        random_erasing_min_aspect=float(aug_cfg.get("random_erasing_min_aspect", 0.5)),
        random_erasing_max_aspect=float(aug_cfg.get("random_erasing_max_aspect", 2.0)),
        random_erasing_max_regions=int(aug_cfg.get("random_erasing_max_regions", 1)),
        random_erasing_max_attempts=int(aug_cfg.get("random_erasing_max_attempts", 10)),
        random_erasing_min_side=int(aug_cfg.get("random_erasing_min_side", 8)),
        random_erasing_fill_mode=aug_cfg.get("random_erasing_fill_mode", "mean"),
        random_erasing_fill_value=float(aug_cfg.get("random_erasing_fill_value", 0)),
        num_workers=num_workers,
        num_classes=num_classes,
        debug_feature_stats=debug_feature_stats,
        mp_min_work_items=mp_min_work_items,
        force_multiprocess=force_multiprocess,
    )
    assert len(feat_list) == len(id_list)
    filtered = [(v, id_, idx) for v, id_, idx in zip(feat_list, id_list, img_indices) if v is not None]
    if len(filtered) == 0:
        print(f"    Dataset {dataset_name:30s} {0:5d} IDs {0:6d} images (from {num_original_images:6d} input) dbg:{dbg} [EMPTY]")
        return (
            np.empty((0, 0), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    filtered_vectors, filtered_ids, filtered_indices = zip(*filtered)
    features_np = torch.stack([v.cpu() for v in filtered_vectors]).numpy().astype(np.float32)
    labels_np = np.array(filtered_ids, dtype=np.int64)
    indices_np = np.array(filtered_indices, dtype=np.int64)
    num_ids = len(np.unique(filtered_ids))
    print(f"    Dataset {dataset_name:30s} {num_ids:5d} IDs {len(filtered_vectors):6d} images (from {num_original_images:6d} input) dbg:{dbg}")
    return features_np, labels_np, indices_np

def make_reid_feats(config=None, datasets=None, save=True):
    batch_size=config.get("yolo_batch_size", 64)
    debug_feature_stats = bool(config.get("debug_feature_stats", False))
    mp_min_work_items = int(config.get("mp_min_work_items", 6))
    force_multiprocess = bool(config.get("force_multiprocess", False))
    train_loader_policies, global_train_aug = _resolve_train_loader_policies(config)
    preview_dir = config.get("train_aug_preview_dir")
    preview_per_loader = int(config.get("train_aug_preview_count_per_loader", 12))

    model=YOLO(config["yolo_model"],  verbose=False)
    num_classes = len(model.names)
    raw = model.task != "posereid"
    print("Number of classes:", num_classes)
    del model

    print("Generated features-")
    output_dict={}

    # Build each training loader separately to allow loader-specific augmentation policy.
    train_loader_keys = [k for k in datasets.keys() if k.startswith("train_")]
    if len(train_loader_keys) == 0 and "train" in datasets:
        train_loader_keys = ["train"]

    train_parts = []
    label_offset = 0
    for d in sorted(train_loader_keys):
        dataset_groups = datasets[d]
        loader_key = _extract_loader_key_from_dataset_name(d)
        aug_cfg = dict(global_train_aug)
        if loader_key is not None:
            aug_cfg.update(train_loader_policies.get(loader_key, {}))
        if d == "train":
            aug_cfg = dict(global_train_aug)

        if preview_dir:
            _generate_aug_preview(
                datasets,
                d,
                aug_cfg,
                preview_dir=preview_dir,
                sample_count=preview_per_loader,
            )

        features_np, labels_np, indices_np = _run_dataset_feature_extract(
            dataset_name=d,
            dataset_groups=dataset_groups,
            config=config,
            num_classes=num_classes,
            batch_size=batch_size,
            debug_feature_stats=debug_feature_stats,
            mp_min_work_items=mp_min_work_items,
            force_multiprocess=force_multiprocess,
            aug_cfg=aug_cfg,
        )
        output_dict[f"{d}_feats"] = features_np
        output_dict[f"{d}_labels"] = labels_np
        output_dict[f"{d}_indices"] = indices_np

        if labels_np.size > 0:
            labels_shifted = labels_np + label_offset
            train_parts.append((features_np, labels_shifted, indices_np))
            label_offset += int(labels_np.max()) + 1

    if train_parts:
        output_dict["train_feats"] = np.concatenate([x[0] for x in train_parts], axis=0)
        output_dict["train_labels"] = np.concatenate([x[1] for x in train_parts], axis=0)
        output_dict["train_indices"] = np.concatenate([x[2] for x in train_parts], axis=0)
    elif "train" in datasets or len(train_loader_keys) > 0:
        output_dict["train_feats"] = np.empty((0, 0), dtype=np.float32)
        output_dict["train_labels"] = np.array([], dtype=np.int64)
        output_dict["train_indices"] = np.array([], dtype=np.int64)

    # Non-train datasets use default (non-augmented) extraction.
    for d in datasets:
        if d == "train" or d.startswith("train_"):
            continue
        features_np, labels_np, indices_np = _run_dataset_feature_extract(
            dataset_name=d,
            dataset_groups=datasets[d],
            config=config,
            num_classes=num_classes,
            batch_size=batch_size,
            debug_feature_stats=debug_feature_stats,
            mp_min_work_items=mp_min_work_items,
            force_multiprocess=force_multiprocess,
            aug_cfg={
                "num_aug": 0,
                "rotate_prob": 0.0,
                "effect_prob": 0.0,
                "albumentations_set": "none",
            },
        )
        output_dict[f"{d}_feats"]=features_np
        output_dict[f"{d}_labels"]=labels_np
        output_dict[f"{d}_indices"]=indices_np

    # Step 4: Save to .npy files
    metadata={"config":config, "raw":raw, "train_loader_policies": train_loader_policies}
    if save:
        output=config["reid_dataset"]
        np.savez(output, **output_dict, metadata=json.dumps(metadata))
        print(f"Saved dataset as {output}")
    return output_dict, metadata

def make_reid_dataset(config_yaml):
    config=stuff.load_dictionary(config_yaml)

    datasets=get_dataset_images(config)
    make_reid_feats(config=config, datasets=datasets)

def show_top_matches(K, img_list, feat_list, id_list, img_indices, cosine_fn):
    """
    Shows the query image and top 8 image matches by cosine similarity.

    Args:
        K (int): Index of the query image.
        img_list (List[np.ndarray]): List of OpenCV images.
        feat_list (List[torch.Tensor]): List of 1D torch feature vectors.
        id_list (List[str]): List of IDs corresponding to each image.
        cosine_fn (Callable): Function taking two tensors and returning cosine similarity.
    """
    N = len(feat_list)
    if not (len(feat_list) == len(id_list)):# == N):
        raise ValueError("All input lists must be of the same length")

    query_feat = feat_list[K]
    similarities = []

    w=192
    h=256

    for i in range(N):
        if i == K:
            continue
        sim = cosine_fn(query_feat, feat_list[i])
        similarities.append((sim, i))

    # Sort and get top 8 matches
    top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:15]

    # Start with the query image
    display_images = []

    raw = cv2.imread(img_list[img_indices[K]])
    if raw is None:
        raw = np.zeros((h, w, 3), dtype=np.uint8)
    query_img = cv2.resize(raw, (w, h))
    query_img = cv2.putText(query_img.copy(), f"QUERY ID:{id_list[K]}", (5, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    display_images.append(query_img)

    for sim, idx in top_matches:
        raw = cv2.imread(img_list[img_indices[idx]])
        if raw is None:
            continue
        img = cv2.resize(raw, (w, h))
        label = f"ID: {id_list[idx]}"
        score = f"Sim: {sim:.2f}"
        if id_list[idx]==id_list[K]:
            clr=(0, 255, 0)
        else:
            clr=(255,255,255)
        img = cv2.putText(img.copy(), label, (5, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
        img = cv2.putText(img, score, (5, 125),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        display_images.append(img)

    # Grid layout: 4x4 (1 query + up to 15 matches) => 16 total.
    while len(display_images) < 16:
        display_images.append(np.zeros((h, w, 3), dtype=np.uint8))  # pad if needed

    row1 = np.hstack(display_images[0:4])
    row2 = np.hstack(display_images[4:8])
    row3 = np.hstack(display_images[8:12])
    row4 = np.hstack(display_images[12:16])
    grid = np.vstack([row1, row2, row3, row4])

    window_name = f"Query and Top 8 Matches for ID: {id_list[K]}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 1024)
    cv2.imshow(window_name, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fuse_model(config_yaml):
    config=stuff.load_dictionary(config_yaml)
    reid_model.make_reid_model(config_yaml)
    # Optional sanity check: run base vs fused model on one image.
    jpg = config.get("fuse_test_image") or config.get("test_image") or "/mldata/image/arrest2.jpg"
    if not os.path.exists(jpg):
        raise FileNotFoundError(
            f"fuse_model sanity-check image not found: {jpg}. "
            "Set config key 'fuse_test_image' (or 'test_image') to a valid path."
        )
    yolo1=YOLO(config["yolo_model"])
    results1=yolo1(jpg, conf=0.05, max_det=500, half=True, iou=0.45, verbose=False, device="cuda:0")[0]

    yolo2=YOLO(config["reid_yolo_model"])
    results2=yolo2(jpg, conf=0.05, max_det=500, half=True, iou=0.45, verbose=False, device="cuda:0")[0]

    det_boxes1 = results1.boxes.xyxyn.tolist() # center
    det_classes1 = results1.boxes.cls.tolist()
    det_confidences1 = results1.boxes.conf.tolist()

    det_boxes2 = results2.boxes.xyxyn.tolist() # center
    det_classes2 = results2.boxes.cls.tolist()
    det_confidences2 = results2.boxes.conf.tolist()

    n=min(len(det_boxes1), len(det_boxes2))
    for i in range(n):
        print(f" {i:2d} CONF {det_confidences1[i]:0.3f}|{det_confidences2[i]:0.3f} CLASS {det_classes1[i]}:{det_classes2[i]}")

    print(results2.reid_embeddings)

def test_reid(config_yaml, val_set=None):
    config=stuff.load_dictionary(config_yaml)

    datasets=get_dataset_images(config, tasks=["val"])
    if val_set is not None and val_set!="all":
        s=datasets[val_set]
        datasets={}
        datasets[val_set]=s
    dict, metadata=make_reid_feats(config=config, datasets=datasets, save=False)
    raw=metadata["raw"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(config["reid_model"], map_location="cpu")

    feat_list=dict[val_set+"_feats"]
    id_list=dict[val_set+"_labels"]
    img_indices=dict[val_set+"_indices"]
    img_list=[]

    in_dim=feat_list[0].shape[0]

    model=None
    if raw:
        print(f"Using reid adapter model {config['reid_model']}")
        model = build_reid_adapter_from_state_dict(state_dict, device=device)
    else:
        print("No reid adapter model, using raw embeddings")

    assert len(feat_list)==len(id_list) and len(feat_list)==len(img_indices)

    for i in datasets[val_set]:
        for j in i:
            img_list.append(j)

    reid_eval.evaluate_recall_faiss(model, feat_list, id_list, device=device, name=val_set)

    # --- Transform a sample vector ---
    feat_list2=[]
    if raw:
        with torch.no_grad():
            for i, f in enumerate(feat_list):
                if f is None:
                    feat_list2.append(None)
                else:
                    f=torch.from_numpy(f).to(device)
                    vec_256 = f.unsqueeze(0)
                    f64 = model(vec_256)
                    feat_list2.append(f64.squeeze(0))
    else:
        for i, f in enumerate(feat_list):
            feat_list2.append(torch.from_numpy(f).to(device))

    assert len(feat_list)==len(feat_list2)

    if len(feat_list2) == 0:
        print("No usable features extracted; skipping interactive match visualization.")
        return

    for _ in range(len(img_list)):
        idx = random.randint(0, len(feat_list2) - 1)
        show_top_matches(idx, img_list, feat_list2, id_list, img_indices, stuff.cosine_similarity)

def eval_reid(config_yaml):
    config=stuff.load_dictionary(config_yaml)

    data = np.load(config["reid_dataset"])
    arrays=data.files

    val_names=[]
    for v in arrays:
        if v.startswith("val") and v.endswith("_labels"):
            name=v[:-len("_labels")]
            val_names.append(name)

    raw=True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(config["reid_model"], map_location="cpu")

    in_dim=data[val_names[0]+"_feats"][0].shape[0]

    model=None
    if raw:
        print(f"Using reid adapter model {config['reid_model']} features {in_dim}")
        model = build_reid_adapter_from_state_dict(state_dict, device=device)
    else:
        print("No reid adapter model, using raw embeddings")

    for n in val_names:
        features=data[n+"_feats"]
        labels=data[n+"_labels"]
        reid_eval.evaluate_recall_faiss(model, features, labels, device=device, name=n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='reid_dataset.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default="/mldata/config/reid/reid_train.yaml", help="config")
    parser.add_argument('--test', type=str, default=None, help='test val set')
    parser.add_argument('--fuse-model', action='store_true', help='make fused model')
    parser.add_argument('--eval', action='store_true', help='evaluate top K')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)
    if opt.test is not None:
        test_reid(opt.config, val_set=opt.test)
        exit()
    if opt.fuse_model:
        fuse_model(opt.config)
        exit()
    if opt.eval:
        eval_reid(opt.config)
        exit()
    make_reid_dataset(opt.config)