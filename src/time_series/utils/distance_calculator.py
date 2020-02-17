from math import sqrt
from .bounding_box import BoundingBox


class DistanceCalculator:

    @classmethod
    def euclidean_distance(cls, bb_1: BoundingBox, bb_2: BoundingBox) -> float:
        return sqrt(((bb_1.pos_x - bb_2.pos_x) ** 2) + ((bb_1.pos_y - bb_2.pos_y) ** 2))
