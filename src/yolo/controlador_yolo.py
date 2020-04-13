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
from src.yolo.utils.constantes import Constantes
from src.yolo.utils.frame import Frame
from src.yolo.utils.img_carro import ImgCarro
from src.yolo.utils.ponto import Ponto
from src.yolo.utils.tamanho import Tamanho


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
		carros: List[ImgCarro] = list()

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

					box = BoundingBox(Ponto(x, y), Tamanho(w, h), self.extrair_subimagem(image, x, y, w, h), score)
					distancia = self.extrair_distancia(H, h)
					posicao = self.extrair_posicao(distancia, camera.graus, x, W)
					carros.append(ImgCarro(box, posicao))
		self.frames.append(Frame(carros))

	def extrair_distancia(self, H, h):
		ocupacao_radianos = math.radians(Constantes.ANGULO_ABERTURA_CAMERA / (H / h))
		return Constantes.TAMANHO_CARRO / ocupacao_radianos

	def extrair_posicao(self, distancia, angulo_camera, x, W):
		angulo = x - (W / 2) / (W / 2) * Constantes.METADE_ANGULO_ABERTURA_CAMERA
		cateto_adjascente = math.cos(angulo) * distancia
		cateto_oposto = math.sin(angulo) * distancia
		if math.cos(angulo_camera) != 0:
			x = cateto_oposto * math.cos(angulo_camera)
			y = cateto_adjascente * math.cos(angulo_camera)
			return Ponto(x, y)
		else:
			y = cateto_oposto * math.sin(angulo_camera)
			x = cateto_adjascente * math.sin(angulo_camera) * -1
			return Ponto(x, y)

	def extrair_subimagem(self, imagem, x, y, w, h):
		return imagem[y:y + h, x:x + w]
