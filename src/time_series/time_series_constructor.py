from typing import List
from .utils.frame import Frame
from .utils.bounding_box import BoundingBox
from .utils.distance_calculator import DistanceCalculator


class TSConstructor:

    @classmethod
    def calculate_distance(cls, bounding_box: BoundingBox, bounding_box_list: List[BoundingBox]):
        distance_list = [(bb.label,
                          DistanceCalculator.euclidean_distance(bb, bounding_box))
                         for bb in bounding_box_list]

        return distance_list

    @classmethod
    def build_time_series(cls, frame_list: List[Frame]) -> None:
        """
        this method will build a set of time series

        :param frame_list: list of frames containing the bounding-boxes
        :return:
        """

        # SETTING THE FIRST FRAME LABELS
        [setattr(bounding_box, 'label', str(index))
         for index, bounding_box
         in enumerate(frame_list[0].bounding_box_list)]

        labeled_list = [frame_list[0].bounding_box_list]
        del frame_list[0]

        for index, frame in enumerate(frame_list):
            labeled_list.append()

