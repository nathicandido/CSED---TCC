from typing import List
from .utils.frame import Frame


class TSConstructor:

    @classmethod
    def build_time_series(cls, frame_list: List[Frame]) -> None:
        """
        this method will build a set of time series

        :param frame_list: list of frames containing the bounding-boxes
        :return:
        """
        for frame in frame_list:
            print(*frame.bounding_box_list)
