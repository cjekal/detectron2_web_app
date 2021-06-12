import os
import joblib
import glob
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
    return {"prediction_pair": prediction_pair, "instances": Instances.cat[corrosion_prediction, edges_and_welds_prediction]}

def get_merged_filename(merged_prediction):
    return os.path.join(f"data/predictions/merged/{merged_prediction['prediction_pair']['base_file']}")

if __name__ == "__main__":
    os.makedirs("data/predictions/merged", exist_ok=True)

    predictions = zip_by_pattern(glob.glob("data/predictions/corrosion/*.pkl"), glob.glob("data/predictions/edges-and-welds/*.pkl"))
    for prediction_pair in predictions:
        merged_prediction = merge(prediction_pair)
        with open(get_merged_filename(merged_prediction), "wb") as f:
            joblib.dump(merged_prediction, f)
