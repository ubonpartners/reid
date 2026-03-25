import os
import logging
from src.loaders.common import numpy_split_list

class LPWLoader:
    def __init__(self,
                 task="val",
                 lpw_path="/mldata/downloaded_datasets/reid/LPW"):

        self.lpw_path=lpw_path

        ret={}

        for scene in ["scen1","scen2","scen3"]:
            path=self.lpw_path+"/"+scene
            for view in ["view1","view2","view3"]:
                view_path=path+"/"+view
                files=os.listdir(view_path)
                for f in files:
                    id=scene+"."+f
                    if id not in ret:
                        ret[id]=[]
                    images=os.listdir(view_path+"/"+f)
                    for i in images:
                        ret[id].append(view_path+"/"+f+"/"+i)

        self.img_dict=ret
        self.ids=ret.keys()
        self.train_ids, self.val_ids=numpy_split_list(self.ids)

        if task=="val":
            self.ids=self.val_ids
        else:
            self.ids=self.train_ids

        logging.info(f"LPW {task} found {len(self.ids)} ids")

    def get_info(self):
        return "LPW dataset"

    def get_name(self):
        return "lpw"

    def get_ids(self):
        return self.ids

    def get_image_paths(self, id):
        return self.img_dict[id]