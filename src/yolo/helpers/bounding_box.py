from pathlib import Path

import cv2
import numpy


class BoundingBox:
    def __init__(self, center, size, image, score):
        self.center = center
        self.size = size
        self.image = image
        self.score = score

    def __str__(self):
        cv2.imwrite(str(Path.joinpath(Path.cwd(), 'saved_images', f'{str(self.center)}.jpg')), self.image)
        return f'center: {self.center}, size: {self.size}, score: {self.score}'
