import cv2

from yolo.exceptions.exceptions import ReadFrameException


class Camera:

    def __init__(self, id):
        self.id = id
        self.video_capture = cv2.VideoCapture(id)
        self.image = self.video_capture.read()

    def try_to_capture_image(self):
        # type: (None) -> numpy
        status, self.image = self.video_capture.read()
        if not status:
            raise ReadFrameException('Problems to read {}'.format(self.id))
        return self.image
