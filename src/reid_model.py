import torch
import stuff
from ultralytics import YOLO
from copy import deepcopy
import yaml
from ultralytics.nn.modules.head import (
    Pose,
    Pose26,
    Pose26ReID,
    Pose26ReIDV2,
    PoseReID,
    PoseReIDV2,
    ReIDAdapter,
    ReIDAdapterV2,
)


def merge_weights_from(model_base, merge_from):
    """Copy weights from a YOLO checkpoint into model_base by exact key + shape match."""
    model = YOLO(merge_from)

    state_base = model_base.model.state_dict()
    state_src = model.model.state_dict()

    for name, param in state_src.items():
        if name in state_base:
            if param.shape == state_base[name].shape:
                print(f"can replace {name}")
                state_base[name] = param
            else:
                print(f"can't replace {name} - different shapes")
    model_base.model.load_state_dict(state_base)

    diff_keys = {k: state_base[k] for k in state_base if k not in state_src}
    print(diff_keys.keys())


def merge_weights_from_pth(model_base, merge_from, debug=False):
    """
    Merge a raw .pth state dict into model_base.

    Keys are matched by exact name or by suffix (e.g. "reid.fc.weight" matches
    "model.22.reid.fc.weight"). Raises on shape mismatch or ambiguous suffix match.
    """
    state_dict = torch.load(merge_from, map_location='cpu')
    state_base = model_base.model.state_dict()
    err = False
    for name, param in state_dict.items():
        matches = [n for n in state_base if n == name or n.endswith("." + name)]
        if len(matches) > 1:
            raise RuntimeError(
                f"Adapter merge ambiguity: source key '{name}' matches multiple "
                f"base keys: {matches}. Use fully-qualified key names."
            )
        replaced = False
        if matches:
            n = matches[0]
            if param.shape == state_base[n].shape:
                if debug:
                    print(f"can replace {n} with {name}")
                state_base[n] = param
                replaced = True
            else:
                if debug:
                    print(f"can't replace {n} with {name} - bad shape")
                err = True
        if not replaced and not matches:
            if debug:
                print(f"{name} not replaced!")
    if err:
        raise RuntimeError("Adapter merge failed due to shape/key mismatch.")
    model_base.model.load_state_dict(state_base)


def _load_config(config_or_yaml):
    if isinstance(config_or_yaml, dict):
        return config_or_yaml
    return stuff.load_dictionary(config_or_yaml)


def _sniff_adapter_version_from_pth(reid_weights_path):
    """Return (adapter_cls, emb) by inspecting keys+shapes in a saved state_dict."""
    sd = torch.load(reid_weights_path, map_location="cpu")
    if "feat_ln.weight" in sd:
        emb = int(sd["mlp.8.weight"].shape[0])
        return ReIDAdapterV2, emb
    emb = int(sd["mlp.6.weight"].shape[0])
    return ReIDAdapter, emb


def _promote_head_to_reid_inplace(model, adapter_cls=ReIDAdapter, emb=None):
    """
    Promote the terminal pose head to its ReID variant by in-place class swap,
    then attach a freshly initialised adapter (v1 or v2 depending on adapter_cls).

    This is the preferred fusion path: it preserves all detector weights exactly
    as they came from the base checkpoint and avoids any YAML rebuild.
    """
    head = model.model.model[-1]
    is_v2 = adapter_cls is ReIDAdapterV2
    if isinstance(head, Pose26) and not isinstance(head, Pose26ReID):
        head.__class__ = Pose26ReIDV2 if is_v2 else Pose26ReID
    elif isinstance(head, Pose) and not isinstance(head, PoseReID):
        head.__class__ = PoseReIDV2 if is_v2 else PoseReID
    elif isinstance(head, (PoseReID, Pose26ReID)):
        pass
    else:
        raise ValueError(f"Unsupported head type for ReID promotion: {type(head).__name__}")

    in_dim = int(head.nc + getattr(head, "attr_nc", 0) + head.FEAT_PLUS_CODE)
    kwargs = {"in_dim": in_dim}
    if emb is not None:
        kwargs["emb"] = int(emb)
    head.reid = adapter_cls(**kwargs)
    return model


def _copy_ckpt_metadata(base_ckpt_path, fused_ckpt_path):
    """Copy task/names/attr metadata from the base checkpoint into the fused checkpoint."""
    print(f"Loading {base_ckpt_path}")
    ckpt_base = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_fused = torch.load(fused_ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt_fused.get("model")
    ema = ckpt_fused.get("ema")
    if model is None:
        raise ValueError("Checkpoint does not contain a 'model' key")

    train_args = ckpt_base.get("train_args", {}) or {}
    base_model = ckpt_base.get("model")
    names     = getattr(base_model, "names",      None)
    kpt_shape = getattr(base_model, "kpt_shape",  None)
    attr_names = getattr(base_model, "attr_names", None)
    attr_ncs  = getattr(base_model, "attr_ncs",   None)
    attr_nc   = getattr(base_model, "attr_nc",    None)

    def _apply_metadata(m):
        if m is None:
            return
        if not hasattr(m, "args"):
            m.args = {}
        m.args["task"] = "posereid"
        m.task = "posereid"
        if names is not None:
            m.names = names
        if kpt_shape is not None:
            m.kpt_shape = kpt_shape
        if attr_names is not None:
            m.attr_names = attr_names
        if attr_ncs is not None:
            m.attr_ncs = attr_ncs
        if attr_nc is not None:
            m.attr_nc = attr_nc
        if hasattr(m, "yaml") and isinstance(m.yaml, dict) and attr_nc is not None:
            m.yaml["attr_nc"] = int(attr_nc)

    # Important: Ultralytics load_checkpoint() prefers ckpt['ema'] over ckpt['model'].
    # Patch both so task routing always resolves to PoseReIDPredictor.
    _apply_metadata(model)
    _apply_metadata(ema)

    train_args["task"] = "posereid"
    ckpt_fused["train_args"] = train_args
    torch.save(ckpt_fused, fused_ckpt_path)


def make_reid_model(config_or_yaml):
    """
    Build a fused ReID-capable YOLO model from a base pose checkpoint and trained adapter.

    Normal path (no reid_yaml in config):
      - loads base model
      - promotes head in-place to PoseReID / Pose26ReID
      - injects adapter weights
      - saves .pt and exports .onnx

    Manual YAML path (reid_yaml set in config):
      - builds model from explicit YAML (for experiments)
      - copies weights from base checkpoint by name/shape match
    """
    config = _load_config(config_or_yaml)

    yolo_weights = config["yolo_model"]
    reid_yaml    = config.get("reid_yaml")
    reid_weights = config["reid_model"]
    dest_model   = config["reid_yolo_model"]
    dest_onnx    = config["reid_onnx_model"]

    if reid_yaml is None:
        adapter_cls, emb = _sniff_adapter_version_from_pth(reid_weights)
        print(f"Fusing {adapter_cls.__name__} (emb={emb}) from {reid_weights}")
        model = YOLO(yolo_weights, verbose=False)
        model = _promote_head_to_reid_inplace(model, adapter_cls=adapter_cls, emb=emb)
    else:
        model = YOLO(reid_yaml, task="pose")
        model.save(dest_model)
        model = YOLO(dest_model)
        merge_weights_from(model, yolo_weights)

    merge_weights_from_pth(model, reid_weights, debug=True)
    model.save(dest_model)

    _copy_ckpt_metadata(yolo_weights, dest_model)
    print(f"Saved finished output model {dest_model}")

    model = YOLO(dest_model)
    model.export(format="onnx", imgsz=(640, 640), dynamic=True, simplify=False, optimize=False)
    onnx_file = dest_model.replace(".pt", ".onnx")
    stuff.rename(onnx_file, dest_onnx)
