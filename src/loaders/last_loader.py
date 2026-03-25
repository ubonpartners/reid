
import os
import logging

class LastLoader:
    def __init__(self,
                 task="val",
                 last_path="/mldata/downloaded_datasets/reid/last"):

        self.last_path=last_path
        if task=="val":
            self.id_path=last_path+"/val/gallery"
        else:
            self.id_path=last_path+"/train"
        self.ids=os.listdir(self.id_path)
        self.ids=[id for id in self.ids if id!="000000"]
        logging.info(f"LAST {task} found {len(self.ids)} ids")

    def get_info(self):
        return "last dataset"

    def get_name(self):
        return "last"

    def get_ids(self):
        return self.ids

    def get_image_paths(self, id):
        path=self.id_path+"/"+id
        images=os.listdir(path)
        return [self.id_path+"/"+id+"/"+i for i in images]