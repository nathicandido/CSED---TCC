from operator import itemgetter
from typing import List, Dict, Tuple

from .utils.bounding_box import BoundingBox
from .utils.distance_calculator import DistanceCalculator
from .utils.frame import Frame


class TSConstructor:

    @classmethod
    def __set_initial_labels(cls, frame_list: List[Frame]) -> None:
        """

        @param frame_list:
        @return:
        """
        [setattr(bounding_box, 'label', str(index))
         for index, bounding_box
         in enumerate(frame_list[0].bounding_box_list)]

    @classmethod
    def __map_initial_labels(cls, frame_list: List[Frame]) -> Dict[str, Dict[str, List]]:
        """

        @param frame_list:
        @return:
        """
        label_mapping = dict()
        [label_mapping.update({str(bounding_box.label): {'pos_x': [], 'pos_y': []}})
         for bounding_box in frame_list[0].bounding_box_list]

        return label_mapping

    @classmethod
    def __calculate_distance(cls, bounding_box: BoundingBox, bounding_box_list: List[BoundingBox]) -> List[
        Tuple[str, float]]:
        """

        @param bounding_box:
        @param bounding_box_list:
        @return:
        """
        distance_list = [(bb.label,
                          DistanceCalculator.euclidean_distance(bb, bounding_box))
                         for bb in bounding_box_list]

        return distance_list

    @classmethod
    def build_time_series(cls, frame_list: List[Frame]) -> Dict[str, Dict[str, list]]:
        """

        @param frame_list:
        @return:
        """

        cls.__set_initial_labels(frame_list)

        for index, frame in enumerate(frame_list):
            if index > 0:
                for bounding_box in frame.bounding_box_list:
                    distance_list = cls.__calculate_distance(bounding_box, frame_list[index - 1].bounding_box_list)
                    distance_list.sort(key=itemgetter(1))  # SORTING BY DISTANCE
                    setattr(bounding_box, 'label', distance_list[0][0])

        ts_mapping = cls.__map_initial_labels(frame_list)

        for frame in frame_list:
            for bounding_box in frame.bounding_box_list:
                ts_mapping[bounding_box.label]['pos_x'].append(bounding_box.pos_x)
                ts_mapping[bounding_box.label]['pos_y'].append(bounding_box.pos_y)

        return ts_mapping
