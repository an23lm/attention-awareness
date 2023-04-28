import cv2


class CameraOutput:
    def __init__(self, enable):
        self.enable = enable

    def set_frame(self, frame, dim):
        self.frame = frame
        self.frame_w = dim[1]
        self.frame_h = dim[0]

    def add_points(self, points, rgb_color=(0, 255, 0)):
        for id, landmark in enumerate(points):
            x = int(landmark.x * self.frame_w)
            y = int(landmark.y * self.frame_h)
            cv2.circle(self.frame, (x, y), 1, rgb_color, thickness=1)

    def add_text(self, text, rgb_color=(0, 255, 0)):
        cv2.putText(self.frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb_color, 2)

    def show_output(self):
        cv2.imshow('Debug Output', self.frame)
        cv2.waitKey(1)
