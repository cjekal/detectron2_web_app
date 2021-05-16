from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
import json
import glob

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = "./model.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    def get_dataset_catalog():
        return []

    DatasetCatalog.register("my_dataset", get_dataset_catalog)
    MetadataCatalog.get("my_dataset").thing_classes = ["Edge", "Weld"]

    predictor = BatchPredictor(cfg)
    images = [cv2.imread(file) for file in
              glob.glob("data/inference/**/*.jpg", recursive=True) +
              glob.glob("data/inference/**/*.JPG", recursive=True) +
              glob.glob("data/inference/**/*.png", recursive=True) +
              glob.glob("data/inference/**/*.PNG", recursive=True)
    ]
    outputs = predictor.predict(images)
    with open("data/predictions.json", "w") as f:
        json.dump(outputs, f, indent=2)
