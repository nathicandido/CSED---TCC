import cv2

from constants.general_parameters import GeneralParameters
from yolo.exceptions.exceptions import ReadFrameException


class Camera:

    def __init__(self, camera_id, angle):
        self.camera_id = camera_id
        self.video_capture = cv2.VideoCapture(camera_id)
        self.angle = angle
        prop = cv2.CAP_PROP_FRAME_COUNT
        self.frame_count = int(self.video_capture.get(prop))
        self.pointer = 0

    def try_to_capture_image(self):
        status, image = self.video_capture.read()
        if not status:
            raise ReadFrameException('Problems to read {}'.format(self.camera_id))
        self.pointer += GeneralParameters.NUMBER_OF_FRAMES
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.pointer)
        return image

    def __repr__(self):
        return f"Camera<ID: {self.camera_id}, ANGLE: {self.angle}, FRAME_COUNT: {self.frame_count}>"
