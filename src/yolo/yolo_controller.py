import math

import cv2
import numpy as np
import os

from typing import List
from src.yolo.exceptions.exceptions import ReadFrameException
from src.yolo.utils.bounding_box import BoundingBox
from src.yolo.utils.camera import Camera

# ----------HOW TO----------
# cameras = [Camera(0)]		use the default camera, you can use the absolute path to a mp4 video
# yolo = Yolo(cameras)
# yolo.run()
from src.yolo.utils.constants import Constants
from src.yolo.utils.frame import Frame
from src.yolo.utils.img_car import ImgCar
from src.yolo.utils.point import Point
from src.yolo.utils.size import Size


class ControladorYolo:
	labels = None
	yolo_network = None
	LABELS_TO_DETECT = [2, 3, 5, 6, 7]  # [Car, Motorbike, Bus, Train, Truck]
	CONFIDENCE = 0.8
	frames: List[Frame] = list()

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

	def get_cameras_images(self):
		layer_names = self.yolo_network.getLayerNames()
		layer_names = [layer_names[i[0] - 1] for i in self.yolo_network.getUnconnectedOutLayers()]
		cars_list: List[ImgCar] = list()

		for camera in self.cameras:
			image = camera.try_to_capture_image()
			(H, W) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			self.yolo_network.setInput(blob)
			layer_outputs = self.yolo_network.forward(layer_names)

			# will be returned 3 layers, but just the first is necessary (thinking about this...)
			for output in layer_outputs[:]:
				for detection in output:
					scores = detection[5:]
					if scores.any() < self.CONFIDENCE and np.argmax(scores) not in self.LABELS_TO_DETECT:
						continue
					(x, y, w, h) = detection[0:4] * np.array([W, H, W, H])
					class_id = int(np.argmax(scores))
					score = scores[class_id]

					b_box = BoundingBox(Point(x, y), Size(w, h), self.extract_b_box_as_image(image, x, y, w, h), score)
					distance = self.extract_distance_car_and_camera(H, h)
					position = self.extract_car_position(distance, camera.angle, x, W)
					cars_list.append(ImgCar(b_box, position))
		self.frames.append(Frame(cars_list))

	def extract_distance_car_and_camera(self, H, h):
		rad = math.radians(Constants.CAMERA_APERTURE_ANGLE / (H / h))
		return Constants.CAR_SIZE / rad

	def extract_car_position(self, distance, camera_angle, x, W):
		angle = x - (W / 2) / (W / 2) * Constants.HALF_CAMERA_APERTURE_ANGLE
		adjacent_side = math.cos(angle) * distance
		opposite_side = math.sin(angle) * distance
		if math.cos(camera_angle) != 0:
			x = opposite_side * math.cos(camera_angle)
			y = adjacent_side * math.cos(camera_angle)
			return Point(x, y)
		else:
			y = opposite_side * math.sin(camera_angle)
			x = adjacent_side * math.sin(camera_angle) * -1
			return Point(x, y)

	def extract_b_box_as_image(self, image, x, y, w, h):
		return image[y:y + h, x:x + w]