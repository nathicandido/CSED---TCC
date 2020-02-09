from .bounding_box import BoundingBox
from typing import List


class Frame:

    def __init__(self, frame_number: int, bounding_box_list: List[BoundingBox]=None):
        if bounding_box_list is None:
            bounding_box_list = list()

        self.frame_number = frame_number
        self.bounding_box_list = bounding_box_list

    def __str__(self):
        return f'<Frame: {self.frame_number}, {self.bounding_box_list}>'
