import cv2 as cv
import os
import joblib
import glob
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_raw_image(prediction_filename):
    basename = os.path.basename(prediction_filename)
    basename_without_ext, basename_ext = os.path.splitext(basename)
    raw_image_filename = glob.glob(f"data/inference/{basename_without_ext}.*")[0]
    return cv.imread(raw_image_filename), basename_without_ext, basename_ext

if __name__ == "__main__":
    os.makedirs("data/predictions/visualized", exist_ok=True)
    
    def get_dataset_catalog():
        return []
    
    DatasetCatalog.register("my_dataset", get_dataset_catalog)
    MetadataCatalog.get("my_dataset").thing_classes = ["Corrosion", "Light Corrosion", "Edge", "Weld"]
    metadata = MetadataCatalog.get("my_dataset")
    
    filenames = glob.glob("data/predictions/merged/*.pkl")
    for filename in filenames:
        predictions = joblib.load(filename)
        im, basename_without_ext, basename_ext = get_raw_image(filename)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        v = v.draw_instance_predictions(predictions["instances"])
        cv.imwrite(f"data/predictions/visualized/{basename_without_ext}{basename_ext}", v.get_image())
