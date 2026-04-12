import logging
import os

import cv2
import numpy as np

from src.loaders.common import numpy_split_list


class UbonSyntheticLoader:
    """
    Loader for Ubon synthetic ReID grids where each source image is a 4x4 collage
    of the same identity in different views/scenes.

    For each source image (one identity), this loader returns 16 BGR numpy arrays
    (one crop per grid cell) with the same identity id.
    """

    @classmethod
    def get_train_augmentation_defaults(cls):
        # Synthetic composites are visually clean, so use stronger train augmentations.
        return {
            "num_aug": 20,
            "rotate_prob": 0.2,
            "effect_prob": 0.95,
            "albumentations_set": "aggressive_motion",
        }

    def __init__(
        self,
        task="val",
        synthetic_path="/mldata/downloaded_datasets/reid/ubon-synthetic",
        grid_rows=4,
        grid_cols=4,
    ):
        self.synthetic_path = synthetic_path
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)

        all_files = []
        for name in sorted(os.listdir(self.synthetic_path)):
            lower = name.lower()
            if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
                all_files.append(os.path.join(self.synthetic_path, name))

        # Deterministic train/val split by source-image identity.
        self.train_ids, self.val_ids = numpy_split_list(all_files)
        self.ids = self.val_ids if task == "val" else self.train_ids

        # Cache split crops so repeated access is cheap.
        self._img_cache = {}
        logging.info(
            "UBON synthetic %s found %d ids (source grids) from %s",
            task,
            len(self.ids),
            self.synthetic_path,
        )

    def get_info(self):
        return "Ubon synthetic 4x4 grid dataset"

    def get_name(self):
        return "ubon_synthetic"

    def get_ids(self):
        # Keep ids as source file paths, similar to LAST-style string ids.
        return self.ids

    def _split_grid_to_tiles(self, img_bgr):
        h, w = img_bgr.shape[:2]
        y_edges = np.linspace(0, h, self.grid_rows + 1, dtype=np.int32)
        x_edges = np.linspace(0, w, self.grid_cols + 1, dtype=np.int32)

        tiles = []
        for r in range(self.grid_rows):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            for c in range(self.grid_cols):
                x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
                if y1 <= y0 or x1 <= x0:
                    continue
                tile = img_bgr[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                # Return independent BGR arrays.
                tiles.append(tile.copy())
        return tiles

    def get_image_paths(self, identity):
        # Interface name is historical; this loader returns BGR numpy arrays.
        if identity in self._img_cache:
            return self._img_cache[identity]

        img = cv2.imread(identity)
        if img is None:
            logging.warning("UBON synthetic: failed to read image %s", identity)
            self._img_cache[identity] = []
            return []

        tiles = self._split_grid_to_tiles(img)
        if len(tiles) != self.grid_rows * self.grid_cols:
            logging.warning(
                "UBON synthetic: expected %d tiles but got %d for %s",
                self.grid_rows * self.grid_cols,
                len(tiles),
                identity,
            )

        self._img_cache[identity] = tiles
        return tiles
