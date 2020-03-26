
import cv2
import numpy as np
import os

from time_series.utils.bounding_box import BoundingBox
from time_series.utils.frame import Frame
from typing import List
from yolo.exceptions.exceptions import ReadFrameException
from yolo.utils.camera import Camera

# ----------HOW TO----------
# cameras = [Camera(0)]		use the default camera, you can use the absolute path to a mp4 video
# yolo = Yolo(cameras)
# yolo.run()


class Yolo:
	labels = None
	yolo_network = None
	LABELS_TO_DETECT = [2, 3, 5, 6, 7]  # [Car, Motorbike, Bus, Train, Truck]
	CONFIDENCE = 0.8

	def __init__(self, cameras: List[Camera], yolo_path=os.path.join(os.getcwd(),'yolo', 'resources', 'yolo-coco'), debug=False):
		self.debug = debug
		if self.debug:
			print('[DEBUG] Yolo init - Camera {}, Yolo {}'.format(cameras, yolo_path))
		self.cameras = cameras
		self.yolo_path = yolo_path
		self.start_yolo()

	def start_yolo(self):
		if self.debug:
			print('[DEBUG] Yolo Start')
		labels_file = os.path.sep.join([self.yolo_path, 'coco.names'])
		self.labels = open(labels_file).read().strip().split('\n')
		config_file = os.path.sep.join([self.yolo_path, 'yolov3.cfg'])
		print(config_file)
		weights_file = os.path.sep.join([self.yolo_path, 'yolov3.weights'])
		print(weights_file)
		self.yolo_network = cv2.dnn.readNetFromDarknet(config_file, weights_file)

	def image_capture(self) -> np.ndarray:
		if self.debug:
			print('[DEBUG] Image Capture')
		frames = []
		for camera in self.cameras:
			try:
				image_from_camera = camera.try_to_capture_image()
			except ReadFrameException as e:
				raise ReadFrameException(e.message)
			frames.append(image_from_camera)
		frames = np.array(frames)
		return frames

	def get_cameras_images(self):
		layer_names = self.yolo_network.getLayerNames()
		layer_names = [layer_names[i[0] - 1] for i in self.yolo_network.getUnconnectedOutLayers()]
		images = self.image_capture()
		frames = []
		for image in images:
			(H, W) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			self.yolo_network.setInput(blob)
			layer_outputs = self.yolo_network.forward(layer_names)

			b_boxes = []

			# will be returned 3 layers, but just the first is necessary (thinking about this...)
			for output in layer_outputs[:-2]:
				for detection in output:
					scores = detection[5:]
					if scores.any() < self.CONFIDENCE and np.argmax(scores) not in self.LABELS_TO_DETECT:
						continue
					(x, y, w, h) = detection[0:4] * np.array([W, H, W, H])
					class_id = int(np.argmax(scores))
					score = scores[class_id]
					b_boxes.append(BoundingBox(x, y, w, h, score, self.labels[class_id]))
			frames.append(Frame(1, b_boxes))

		return frames
