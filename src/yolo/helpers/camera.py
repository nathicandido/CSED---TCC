import cv2

from yolo.exceptions.exceptions import ReadFrameException


class Camera:

    def __init__(self, camera_id, angle):
        self.camera_id = camera_id
        self.video_capture = cv2.VideoCapture(camera_id)
        self.angle = angle

    def try_to_capture_image(self):
        status, image = self.video_capture.read()
        if not status:
            raise ReadFrameException('Problems to read {}'.format(self.camera_id))
        return image
