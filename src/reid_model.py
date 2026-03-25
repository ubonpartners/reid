import torch
import stuff
from ultralytics import YOLO
from copy import deepcopy
import yaml
from ultralytics.nn.modules.head import Pose, Pose26, PoseReID, Pose26ReID, ReIDAdapter

def merge_weights_from(model_base, merge_from):
    # replace weights in 'model_base' with those from 'merge_from'
    # only replace weights with same name and shape
    model =YOLO(merge_from)

    dict_base=model_base.model.state_dict()
    dict=model.model.state_dict()

    for name,param in dict.items():
        if name in dict_base:
            if param.shape==dict_base[name].shape:
                print(f"can replace {name}")
                dict_base[name]=param
            else:
                print(f"can't replace {name} - different shapes")
    model_base.model.load_state_dict(dict_base)

    diff_keys = {k: dict_base[k] for k in dict_base if k not in dict}
    print(diff_keys.keys())

def merge_weights_from_pth(model_base, merge_from, debug=False):
    # replace weights in 'model_base' with those from 'merge_from'
    # only replace weights with same name and shape
    state_dict =torch.load(merge_from, map_location='cpu')
    dict_base=model_base.model.state_dict()
    err=False
    for name,param in state_dict.items():
        replaced=False
        for n in dict_base:
            if n.endswith(name):
                if param.shape==dict_base[n].shape:
                    if debug:
                        print(f"can replace {n} with {name}")
                    dict_base[n]=param
                    replaced=True
                else:
                    if debug:
                        print(f"can't replace {n} with {name} - bad shape")
                    err=True
        if not replaced:
            if debug:
                print(f"{name} not replaced!")
    if err:
        raise RuntimeError("Adapter merge failed due to shape/key mismatch.")
    model_base.model.load_state_dict(dict_base)

def _load_config(config_or_yaml):
    if isinstance(config_or_yaml, dict):
        return config_or_yaml
    return stuff.load_dictionary(config_or_yaml)


def _infer_posereid_cfg_from_base(yolo_weights):
    base = YOLO(yolo_weights, verbose=False)
    cfg = deepcopy(base.model.yaml)
    model_head = base.model.model[-1]
    head_type = type(model_head).__name__
    head = cfg.get("head")
    if not isinstance(head, list) or len(head) == 0:
        raise ValueError("Cannot infer PoseReID config: missing yaml head list in base model.")
    # Replace final head module with matching ReID variant while preserving structure.
    head[-1][2] = "Pose26ReID" if head_type == "Pose26" else "PoseReID"
    cfg["head"] = head

    # Keep attr head width explicit when available.
    attr_nc = int(getattr(model_head, "attr_nc", 0) or 0)
    if attr_nc > 0:
        cfg["attr_nc"] = attr_nc

    # Keep class/keypoint settings synced with loaded base model.
    cfg["nc"] = int(getattr(model_head, "nc", cfg.get("nc", 0)))
    if hasattr(model_head, "kpt_shape"):
        cfg["kpt_shape"] = list(model_head.kpt_shape)
    return cfg


def _write_yaml(cfg, output_yaml):
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def _promote_head_to_reid_inplace(model):
    head = model.model.model[-1]
    if isinstance(head, Pose26) and not isinstance(head, Pose26ReID):
        head.__class__ = Pose26ReID
    elif isinstance(head, Pose) and not isinstance(head, PoseReID):
        head.__class__ = PoseReID
    elif isinstance(head, (PoseReID, Pose26ReID)):
        pass
    else:
        raise ValueError(f"Unsupported head type for ReID promotion: {type(head).__name__}")

    in_dim = int(head.nc + getattr(head, "attr_nc", 0) + head.FEAT_PLUS_CODE)
    head.reid = ReIDAdapter(in_dim=in_dim)
    return model


def _copy_ckpt_metadata(base_ckpt_path, fused_ckpt_path):
    print(f"Loading {base_ckpt_path}")
    ckpt_base = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_fused = torch.load(fused_ckpt_path, map_location="cpu", weights_only=False)
    model = ckpt_fused.get("model")
    if model is None:
        raise ValueError("Checkpoint does not contain a 'model' key")

    train_args = ckpt_base.get("train_args", {}) or {}
    names = getattr(ckpt_base.get("model"), "names", None)
    kpt_shape = getattr(ckpt_base.get("model"), "kpt_shape", None)
    attr_names = getattr(ckpt_base.get("model"), "attr_names", None)
    attr_ncs = getattr(ckpt_base.get("model"), "attr_ncs", None)
    attr_nc = getattr(ckpt_base.get("model"), "attr_nc", None)

    if not hasattr(model, "args"):
        model.args = {}
    model.args["task"] = "posereid"
    model.task = "posereid"
    if names is not None:
        model.names = names
    if kpt_shape is not None:
        model.kpt_shape = kpt_shape
    if attr_names is not None:
        model.attr_names = attr_names
    if attr_ncs is not None:
        model.attr_ncs = attr_ncs
    if attr_nc is not None:
        model.attr_nc = attr_nc
    if hasattr(model, "yaml") and isinstance(model.yaml, dict) and attr_nc is not None:
        model.yaml["attr_nc"] = int(attr_nc)

    train_args["task"] = "posereid"
    ckpt_fused["train_args"] = train_args
    torch.save(ckpt_fused, fused_ckpt_path)


def make_reid_model(config_or_yaml):
    config = _load_config(config_or_yaml)

    yolo_weights = config["yolo_model"]
    reid_yaml = config.get("reid_yaml")
    reid_weights = config["reid_model"]
    dest_model = config["reid_yolo_model"]
    dest_onnx = config["reid_onnx_model"]

    # Preferred path: preserve base detector behavior exactly by promoting the base head in-place.
    # Fallback path keeps manual YAML mode for explicit experiments.
    if reid_yaml is None:
        model = YOLO(yolo_weights, verbose=False)
        model = _promote_head_to_reid_inplace(model)
    else:
        model = YOLO(reid_yaml, task="pose")
        model.save(dest_model)
        model = YOLO(dest_model)
        merge_weights_from(model, yolo_weights)

    merge_weights_from_pth(model, reid_weights, True)
    model.save(dest_model)

    # copy across metadata and task settings required by runtime
    _copy_ckpt_metadata(yolo_weights, dest_model)
    print(f"Saved finished output model {dest_model}")

    model = YOLO(dest_model)
    model.export(format="onnx", imgsz=(640, 640), dynamic=True, simplify=False, optimize=False)
    onnx_file=dest_model.replace(".pt", ".onnx")
    stuff.rename(onnx_file, dest_onnx)