import cv2

class VideoStreamHandle:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        assert self.cam.isOpened()

    def capture_frame(self):
        ret, image = self.cam.read()
        assert ret

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
