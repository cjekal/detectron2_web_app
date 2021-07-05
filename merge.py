import os
import joblib
import glob
import json
from detectron2.structures.instances import Instances

def zip_by_pattern(corrosion_prediction_filenames, edges_welds_predictions_filenames):
    func = lambda x: os.path.basename(x)
    root_filenames = set(list(map(func, corrosion_prediction_filenames)) + list(map(func, edges_welds_predictions_filenames)))
    return [
        {
            "corrosion_file": next(x for x in corrosion_prediction_filenames if func(x) == f),
            "edges_and_welds_file": next(x for x in edges_welds_predictions_filenames if func(x) == f),
            "base_file": f
        }
        for f in root_filenames
    ]

def merge(prediction_pair):
    corrosion_prediction = joblib.load(prediction_pair["corrosion_file"])
    edges_and_welds_prediction = joblib.load(prediction_pair["edges_and_welds_file"])
    for i, pred_class in enumerate(edges_and_welds_prediction.pred_classes):
        edges_and_welds_prediction.pred_classes[i] = pred_class + 2
    return {"prediction_pair": prediction_pair, "instances": Instances.cat([corrosion_prediction, edges_and_welds_prediction])}

def get_merged_filename(merged_prediction):
    return os.path.join(f"data/predictions/merged/{merged_prediction['prediction_pair']['base_file']}")

def print_bbox_json(merged_prediction):
    json_prediction = {"prediction_pair": merged_prediction["prediction_pair"]}
    json_prediction["instances"] = [get_dict_for_instance(instance) for instance in merged_prediction["instances"]]
    print(json.dumps(json_prediction, indent=2))
    print("---")

def get_dict_for_instance(instance):
    if len(instance.pred_boxes) > 1:
        raise Exception("more than 1 pred boxes")
    classes = ["Coating Corrosion", "Edge", "Weld"]
    pred_class = instance.pred_classes[0].item()
    bbox = instance.pred_boxes.tensor[0].tolist()
    center = instance.pred_boxes.get_centers()[0].tolist()
    return {
        "pred_class": pred_class,
        "bbox": bbox,
        "bbox_center": center
    }

if __name__ == "__main__":
    os.makedirs("data/predictions/merged", exist_ok=True)

    predictions = zip_by_pattern(glob.glob("data/predictions/corrosion/*.pkl"), glob.glob("data/predictions/edges-and-welds/*.pkl"))
    for prediction_pair in predictions:
        merged_prediction = merge(prediction_pair)
        with open(get_merged_filename(merged_prediction), "wb") as f:
            joblib.dump(merged_prediction, f)
        print_bbox_json(merged_prediction)
