import math

import cv2
import numpy as np
import os

from typing import List

from logger.log import Log
from yolo.helpers.bounding_box import BoundingBox
from yolo.helpers.camera import Camera

# ----------HOW TO----------
# cameras = [Camera(0)]		use the default camera, you can use the absolute path to a mp4 video
# yolo = Yolo(cameras)
# yolo.run()
from yolo.helpers.yoloconstants import YOLOConstants
from yolo.helpers.frame import Frame
from utils.img_car import ImgCar
from utils.point import Point
from yolo.helpers.size import Size


class YoloController:
	TAG = 'YOLO'
	log = Log()
	labels = None
	yolo_network = None
	LABELS_TO_DETECT = [2] #, 3, 5, 6, 7]  # [Car, Motorbike, Bus, Train, Truck]
	CONFIDENCE = 0.9
	frames: List[Frame] = list()

	def __init__(self, cameras: List[Camera], yolo_path=os.path.join(os.getcwd(), 'yolo', 'resources', 'yolo-coco'), debug=False):
		self.debug = debug
		self.log.i(self.TAG, f'Yolo init')
		if self.debug:
			self.log.d(self.TAG, f'Camera {cameras}, Yolo {yolo_path}')
		self.cameras = cameras
		self.yolo_path = yolo_path
		self.start_yolo()

	def start_yolo(self):
		self.log.i(self.TAG, f'Yolo Start')
		labels_file = os.path.sep.join([self.yolo_path, 'coco.names'])
		self.labels = open(labels_file).read().strip().split('\n')
		config_file = os.path.sep.join([self.yolo_path, 'yolov3.cfg'])
		weights_file = os.path.sep.join([self.yolo_path, 'yolov3.weights'])
		self.yolo_network = cv2.dnn.readNetFromDarknet(config_file, weights_file)
		if self.debug:
			self.log.i(self.TAG, f'Labels: {labels_file}, Config: {config_file}, Weights: {weights_file}')

	def get_cameras_images(self):
		self.log.i(self.TAG, f'Getting images')
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
			for output in layer_outputs[2:]:
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
					img_car = ImgCar(b_box, position)
					if self.debug:
						self.log.d(self.TAG, img_car)
					cars_list.append(img_car)
			image = None
		self.frames.append(Frame(cars_list))
		return cars_list

	@staticmethod
	def extract_distance_car_and_camera(H, h):
		rad = math.radians(YOLOConstants.CAMERA_APERTURE_ANGLE / (H / h))
		return YOLOConstants.CAR_SIZE / rad

	def extract_car_position(self, distance, camera_angle, x, W):
		angle = (x - (W / 2)) / (W / 2) * YOLOConstants.HALF_CAMERA_APERTURE_ANGLE
		angle = math.radians(angle)
		adjacent_side = math.cos(angle) * distance
		opposite_side = math.sin(angle) * distance
		if self.debug:
			self.log.d(self.TAG, f'X= {x}, W= {W}, Distance= {distance}')
			self.log.d(self.TAG, f'angle= {angle}, camera_angle= {int(math.cos(math.radians(camera_angle)))}, '
								f'adj= {adjacent_side}, ops= {opposite_side}')
			self.log.d(self.TAG, f'angle: cos= {math.cos(angle)}, sin= {math.sin(angle)}')
		if int(math.cos(math.radians(camera_angle))) != 0:
			pos_x = opposite_side * math.cos(math.radians(camera_angle))
			pos_y = adjacent_side * math.cos(math.radians(camera_angle))
			self.log.i(self.TAG, f'position: front/rear - X= {pos_x}, Y= {pos_y}')
			return Point(pos_x, pos_y)
		else:
			pos_y = opposite_side * int(math.sin(math.radians(camera_angle)))
			pos_x = adjacent_side * int(math.sin(math.radians(camera_angle))) * -1
			self.log.i(self.TAG, f'position: side - X= {pos_x}, Y= {pos_y}')
			return Point(pos_x, pos_y)

	@staticmethod
	def extract_b_box_as_image(image, x, y, w, h):
		return image[int(y - (h * 0.4)):int(y + (h * 0.4)), int(x - (w * 0.4)):int(x + (w * 0.4))]
