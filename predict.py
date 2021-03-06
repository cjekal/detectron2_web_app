from batch_predictor import BatchPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
import glob
import os
import joblib
import sys

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_predictions_filename(image):
    global metadata
    filename_without_ext = os.path.splitext(os.path.basename(image))[0]
    return f"data/predictions/{metadata['name']}/{filename_without_ext}.pkl"

def image_is_unprocessed(image):
    predictions_file = get_predictions_filename(image)
    return not os.path.exists(predictions_file)

def process_chunk(chunk, predictor):
    print(f"processing chunk of {len(chunk)} images")
    outputs = predictor.predict([cv2.imread(filename) for filename in chunk])
    
    for image, output in zip(chunk, outputs):
        instances = output['instances'].to("cpu")
        os.makedirs(os.path.dirname(get_predictions_filename(image)), exist_ok=True)
        with open(get_predictions_filename(image), "wb") as f:
            joblib.dump(instances, f)
    print("done processing chunk")

if __name__ == "__main__":
    global metadata
    metadata = {
        "corrosion": {
            "name": "corrosion",
            "thing_classes": ["Corrosion", "Light Corrosion"],
            "model_weights_path": "./model-corrosion.pth"
        },
        "edges-and-welds": {
            "name": "edges-and-welds",
            "thing_classes": ["Edge", "Weld"],
            "model_weights_path": "./model-edges-and-welds.pth"
        }
    }[sys.argv[1]]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = metadata["model_weights_path"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    def get_dataset_catalog():
        return []

    thing_classes = metadata["thing_classes"]
    DatasetCatalog.register("my_dataset", get_dataset_catalog)
    MetadataCatalog.get("my_dataset").thing_classes = thing_classes

    predictor = BatchPredictor(cfg)
    image_filenames = glob.glob("data/inference/*.jpg", recursive=True) + glob.glob("data/inference/*.JPG", recursive=True) + glob.glob("data/inference/*.jpeg", recursive=True) + glob.glob("data/inference/*.png", recursive=True) + glob.glob("data/inference/*.PNG", recursive=True)
    image_filenames = [i for i in image_filenames if image_is_unprocessed(i)]
    for chunk in chunks(image_filenames, 24):
        process_chunk(chunk, predictor)
    print("done processing!")
