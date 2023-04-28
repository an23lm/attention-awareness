import cv2


class CameraInput:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def get_frames(self):
        _, frame = self.camera.read()

        self.frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        return self.frame, rgb_frame

    def get_frame_dim(self):
        frame_h, frame_w, _ = self.frame.shape

        return frame_h, frame_w
