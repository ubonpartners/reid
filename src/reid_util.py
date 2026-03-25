import cv2
import numpy as np
import random
from ultralytics import YOLO
import src.loaders.last_loader as last_loader
import src.loaders.cuhk_loader as cuhk_loader
import src.loaders.lpw_loader as lpw_loader
import src.loaders.iust_loader as iust_loader

_LOADER_REGISTRY = {
    "lastloader": last_loader.LastLoader,
    "cuhkloader": cuhk_loader.CUHKLoader,
    "lpwloader": lpw_loader.LPWLoader,
    "iustloader": iust_loader.IUSTLoader,
}


def get_dataset_loader(loader_name):
    """Resolve a dataset loader class by name (case-insensitive)."""
    loader_name = loader_name.lower()
    loader = _LOADER_REGISTRY.get(loader_name)
    if loader is None:
        raise ValueError(f"Unknown REID loader {loader_name}. Known: {sorted(_LOADER_REGISTRY)}")
    return loader

def expand_canvas(image, scale=1.2):
    """Expand the canvas size by the given scale and center the image."""
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    dx, dy = (new_w - w) // 2, (new_h - h) // 2
    expanded = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    expanded[dy:dy+h, dx:dx+w] = image
    return expanded

def crop_random_background(bg_image, target_size):
    """Randomly crop a region from bg_image of size target_size (width, height)."""
    th, tw = target_size[1], target_size[0]
    h, w = bg_image.shape[:2]
    if h < th or w < tw:
        bg_image=cv2.resize(bg_image, (tw, th), interpolation=cv2.INTER_LINEAR)
        w=tw
        h=th
    y = random.randint(0, h - th)
    x = random.randint(0, w - tw)
    return bg_image[y:y+th, x:x+tw]

def paste_on_background(fg_rgba, bg_bgr):
    """Composite fg (with alpha) over a background crop."""
    alpha = fg_rgba[:, :, 3] / 255.0
    fg_rgb = fg_rgba[:, :, :3]
    composite = (alpha[..., None] * fg_rgb + (1 - alpha[..., None]) * bg_bgr).astype(np.uint8)
    return composite

def replace_backgrounds(person_paths, background_paths, model=None):

    if model is None:
        model = YOLO('yolov11x-seg.pt', verbose=False)

    # Load all person images
    person_imgs = [cv2.imread(p) for p in person_paths]

    # Batched inference
    #print(f"Infer {len(person_imgs)}")
    results = model(person_imgs, imgsz=[256,128], rect=True, half=True, verbose=False)

    ret=[]
    for i, (person_img, result) in enumerate(zip(person_imgs, results)):
        found = False
        h, w = person_img.shape[:2]
        if result.masks is None:
            ret.append(None)
            continue
        for mask, cls, box in zip(result.masks.data, result.boxes.cls, result.boxes.xyxy):
            if int(cls) == 0:  # class 0 is 'person'
                found = True

                imgh, imgw = result.masks.orig_shape
                mask = mask.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                maskh, maskw=mask.shape

                # horrible - "mask" is not at the size of the input image
                # its not even the same shape as the input image has been
                # scaled/padded to fit the inference size at the initial
                # aspect ratio
                h=((maskh*imgw)//maskw)
                if (h>=imgh):
                    mask=cv2.resize(mask, (imgw, h))
                    pad=(h-imgh)//2
                    mask_resized=mask[pad:imgh+pad, 0:imgw]
                else:
                    w=((maskw*imgh)//maskh)
                    assert w>=imgw
                    mask=cv2.resize(mask, (w, imgh))
                    pad=(w-imgw)//2
                    mask_resized=mask[0:imgh, pad:imgw+pad]
                break
        if not found:
            print(f"No person found in {person_paths[i]}")
            ret.append(None)
            continue

        # Create 4-channel image with alpha mask
        fg = cv2.cvtColor(person_img, cv2.COLOR_BGR2BGRA)
        fg[:, :, 3] = mask_resized

        # Expand canvas
        fg_expanded = expand_canvas(fg, scale=1.2)

        # Random background crop
        if background_paths:
            bg = cv2.imread(random.choice(background_paths))
            if bg is None:
                bg = np.full((640, 640, 3), 128, dtype=np.uint8)
        else:
            bg = np.full((640, 640, 3), 128, dtype=np.uint8)
        assert bg is not None
        try:
            bg_crop = crop_random_background(bg, (fg_expanded.shape[1], fg_expanded.shape[0]))
        except ValueError:
            print(f"Background too small for {person_paths[i]}, skipping.")
            continue

        # Composite
        composite = paste_on_background(fg_expanded, bg_crop)

        ret.append(composite)
    return ret
