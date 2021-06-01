from batch_predictor import BatchPredictor
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

    thing_classes = ["Edge", "Weld"]
    DatasetCatalog.register("my_dataset", get_dataset_catalog)
    MetadataCatalog.get("my_dataset").thing_classes = thing_classes

    predictor = BatchPredictor(cfg)
    images = [cv2.imread(file) for file in
              glob.glob("data/inference/**/*.jpg", recursive=True) +
              glob.glob("data/inference/**/*.JPG", recursive=True) +
              glob.glob("data/inference/**/*.png", recursive=True) +
              glob.glob("data/inference/**/*.PNG", recursive=True)
    ]
    outputs = predictor.predict(images)
    
    predictions = []
    for image, output in zip(images, outputs):
        instances = output['instances'].to("cpu")
        num_instances = len(instances)
        image_size = instances.image_size
        json_instances = []
        for i in range(num_instances):
            instance = instances[i]
            pred_box_area = instance.pred_boxes.area().data.tolist()
            pred_box_center = instance.pred_boxes.get_centers().data.tolist()
            pred_box = instance.pred_boxes.data.tolist()
            score = instance.scores.data.tolist()
            pred_class = instance.pred_classes.data.tolist()
            pred_mask = instance.pred_masks.data.tolist()
            json_instances.append({
                "pred_box": pred_box,
                "pred_box_area": pred_box_area,
                "pred_box_center": pred_box_center,
                "score": score,
                "pred_class": thing_classes[pred_class],
                "pred_mask": pred_mask
            })
        predictions.append({
            "image": image,
            "image_size": image_size,
            "num_instances": num_instances,
            "instances":json_instances
        })
    
    print("about to serialize:")
    print(json.dumps(predictions, indent=2))

    with open("data/predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
