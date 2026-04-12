import os
import logging
from src.loaders.common import group_by_prefix, numpy_split_list

class CUHKLoader:
    @classmethod
    def get_train_augmentation_defaults(cls):
        return {
            "num_aug": 15,
            "rotate_prob": 0.1,
            "effect_prob": 0.8,
            "albumentations_set": "standard",
        }

    def __init__(self,
                 task="val",
                 cukh_path="/mldata/downloaded_datasets/reid/cuhk03"):

        self.cukh_path=cukh_path+"/archive/images_detected"

        ids=group_by_prefix(os.listdir(self.cukh_path), 5)
        self.train_ids, self.val_ids=numpy_split_list(ids)

        if task=="val":
            self.ids=self.val_ids
        else:
            self.ids=self.train_ids

        logging.info(f"CUHK {task} found {len(self.ids)} ids")

    def get_info(self):
        return "CUHK dataset"

    def get_name(self):
        return "cuhk"

    def get_ids(self):
        return list(range(len(self.ids)))

    def get_image_paths(self, id):
        return [self.cukh_path+"/"+i for i in self.ids[id]]