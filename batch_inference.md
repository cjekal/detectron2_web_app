# Batch Inference

Batch inference is where you take your trained models (or pre-trained models from the model zoo) and perform predictions using groups of images at a time for faster speed.

## Steps

Here's the steps for performing batch inference.

1. SSH into the `ml-inferencer` machine
1. Copy all the images you want to run inference on to `/mnt/disk1/ml-apps/detectron2_web_app/data/inference/` directory.
    1. Only *.JPG, *.PNG, *.jpg and *.png files are picked up (case-sensitive globbing)
1. Copy the Corrosion model weights to `/mnt/disk1/ml-apps/detectron2_web_app/model-corrosion.pth`
1. Copy the Edges-and-Welds model weights to `/mnt/disk1/ml-apps/detectron2_web_app/model-edges-and-welds.pth`
1. Kick-off the pipeline by running the following scripts in order:
    1. `python predict.py corrosion`
    1. `python predict.py edges-and-welds`
    1. `python merge.py`
    1. `python visualize.py`
1. Copy the predictions (`/mnt/disk1/ml-apps/detectron2_web_app/data/predictions/corrosion/*.pkl` and `/mnt/disk1/ml-apps/detectron2_web_app/data/predictions/edges-and-welds/*.pkl`) to your long-term storage of choice
1. Copy the visualizations (`/mnt/disk1/ml-apps/detectron2_web_app/data/predictions/visualized`) to your long-term storage of choice
