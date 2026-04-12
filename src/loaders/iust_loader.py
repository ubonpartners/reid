import os
import logging
from src.loaders.common import group_by_prefix

class IUSTLoader:
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
                 iust_path="/mldata/downloaded_datasets/reid/IUSTPersonReID"):

        self.iust_path=iust_path

        if task=="val":
            self.iust_path=self.iust_path+"/bounding_box_test"
        else:
            self.iust_path=self.iust_path+"/bounding_box_train"

        ids=group_by_prefix(os.listdir(self.iust_path), 4)
        logging.info(f"IUST {task} found {len(ids)} ids")
        self.ids=ids

    def get_info(self):
        return "IUST dataset"

    def get_name(self):
        return "iust"

    def get_ids(self):
        return list(range(len(self.ids)))

    def get_image_paths(self, id):
        return [self.iust_path+"/"+i for i in self.ids[id]]