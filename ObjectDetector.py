import cv2 as cv
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image

class Detector:

	def __init__(self):

		# set model and test set
		self.model = 'mask_rcnn_R_101_FPN_3x.yaml'

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		# self.cfg.merge_from_file("test.yaml")
		self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")) 

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# set number of classes to detect
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

		# get weights 
		self.cfg.MODEL.WEIGHTS = "./model.pth"
		# self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  

		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

		def get_dataset_catalog():
			return []
		
		DatasetCatalog.register("my_dataset", get_dataset_catalog)
		MetadataCatalog.get("my_dataset").thing_classes = ["Edge", "Weld"]

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'

	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	def inference(self, file):
		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)

		# with open(self.curr_dir+'/data.txt', 'w') as fp:
		# 	json.dump(outputs['instances'], fp)
		# 	# json.dump(cfg.dump(), fp)

		# get metadata
		metadata = MetadataCatalog.get("my_dataset")

		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

		# get image 
		img = v.get_image()
		img = Image.fromarray(img)
		# write to jpg
		# cv.imwrite('img.jpg',v.get_image())

		return img
