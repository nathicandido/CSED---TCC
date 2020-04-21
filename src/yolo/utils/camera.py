import cv2

from src.yolo.exceptions.exceptions import ReadFrameException


class Camera:

    def __init__(self, id, angle):
        self.id = id
        self.video_capture = cv2.VideoCapture(id)
        self.angle = angle

    def try_to_capture_image(self):
        # type: (None) -> numpy
        status, image = self.video_capture.read()
        if not status:
            raise ReadFrameException('Problems to read {}'.format(self.id))
        return image
