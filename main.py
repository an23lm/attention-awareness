from cam_input import CameraInput
from cam_output import CameraOutput
from face_feature_detection import FaceDetection

input = CameraInput()
output = CameraOutput(enable=True)

face = FaceDetection()

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

        if face.is_in_attention_range():
            output.add_text("paying attention", (0, 255, 255))
        else:
            output.add_text("not paying attention", (255, 255, 0))

        # print(f'h {face.horizontal_position_of_iris_in_eye()}',
        #       f'v {face.vertical_position_of_iris_in_eye()}')

    output.show_output()
