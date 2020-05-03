import cv2

from yolo.exceptions.exceptions import ReadFrameException
import imutils

class Camera:

    def __init__(self, camera_id, angle):
        self.camera_id = camera_id
        self.video_capture = cv2.VideoCapture(camera_id)
        self.angle = angle
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        print(int(self.video_capture.get(prop)))

    def try_to_capture_image(self):
        status, image = self.video_capture.read()
        if not status:
            raise ReadFrameException('Problems to read {}'.format(self.camera_id))
        return image
