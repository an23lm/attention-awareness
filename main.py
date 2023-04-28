from cam_input import CameraInput
from cam_output import CameraOutput
from face_feature_detection import FaceDetection
from throttled_aggregate_trigger import ThrottledAggregateTrigger
from microphone_control import MicrophoneControl
import logging

logging.basicConfig(level=logging.DEBUG)

input = CameraInput()
output = CameraOutput(enable=True)

face = FaceDetection()


def perform_action(in_attention=False):
    if (in_attention):
        MicrophoneControl.unmute()
    else:
        MicrophoneControl.mute()


throttler = ThrottledAggregateTrigger(.25, perform_action)

try:
    while True:
        frame, rgb_frame = input.get_frames()
        frame_dim = input.get_frame_dim()
        output.set_frame(frame, frame_dim)

        face.process(rgb_frame, frame_dim)

        if face.is_face_detected():
            # right iris
            output.add_points(face.right_iris(), rgb_color=(0, 255, 0))

            # left iris
            output.add_points(face.left_iris(), rgb_color=(0, 255, 0))

            # left eye
            output.add_points(face.left_eye(), rgb_color=(0, 0, 255))

            # right eye
            output.add_points(face.right_eye(), rgb_color=(0, 0, 255))

            # face top
            output.add_points(face.silhouette_top(), rgb_color=(0, 0, 255))

            # face bottom
            output.add_points(face.silhouette_bottom(), rgb_color=(0, 0, 255))

            in_attention, reason = face.is_in_attention_range()

            if in_attention:
                output.add_text("paying attention", (0, 255, 255))
            else:
                output.add_text(f"no attention", (255, 255, 0))

            throttler.add(in_attention)

        output.show_output()
finally:
    throttler.kill()
