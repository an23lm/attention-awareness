import mediapipe as mp
import math


class FaceDetection:
    __right_iris_indices = [473, 474, 475, 476, 477]
    __left_iris_indices = [468, 469, 470, 471, 472]
    __left_eye_indices = [33, 145, 159, 133]
    __right_eye_indices = [263, 374, 386, 362]
    __midway_between_eyes_indices = [168]
    __silhouette_top = [10]
    __silhouette_bottom = [152]

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def process(self, rgb_frame, frame_dim):
        output = self.face_mesh.process(rgb_frame)
        self.landmark_points = output.multi_face_landmarks
        self.frame_dim = frame_dim
        self.focal_length = 55

    def is_face_detected(self):
        if self.landmark_points:
            return True
        return False

    def right_iris(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__right_iris_indices]

    def left_iris(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__left_iris_indices]

    def left_eye(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__left_eye_indices]

    def right_eye(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__right_eye_indices]

    def midway_between_eyes(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__midway_between_eyes_indices]

    def silhouette_top(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__silhouette_top]

    def silhouette_bottom(self):
        if not self.is_face_detected():
            return None

        landmarks = self.landmark_points[0].landmark
        return [landmarks[x] for x in FaceDetection.__silhouette_bottom]

    def width_of_iris(self):
        if not self.is_face_detected():
            return None

        left_iris_coordinates = self.left_iris()
        right_iris_coordinates = self.right_iris()

        left_iris_corner_1 = (
            left_iris_coordinates[1].x, left_iris_coordinates[1].y)
        left_iris_corner_2 = (
            left_iris_coordinates[3].x, left_iris_coordinates[3].y)

        right_iris_corner_1 = (
            right_iris_coordinates[1].x, right_iris_coordinates[1].y)
        right_iris_corner_2 = (
            right_iris_coordinates[3].x, right_iris_coordinates[3].y)

        left_width = math.dist(left_iris_corner_1, left_iris_corner_2)
        right_width = math.dist(right_iris_corner_1, right_iris_corner_2)

        width_average = (left_width + right_width) / 2

        return width_average

    def width_of_eyes(self):
        if not self.is_face_detected():
            return None

        left_eye_coordinates = self.left_eye()
        right_eye_coordinates = self.right_eye()

        left_eye_corner_1 = (
            left_eye_coordinates[0].x, left_eye_coordinates[0].y, left_eye_coordinates[0].z)
        left_eye_corner_2 = (
            left_eye_coordinates[3].x, left_eye_coordinates[3].y, left_eye_coordinates[3].z)

        right_eye_corner_1 = (
            right_eye_coordinates[0].x, right_eye_coordinates[0].y, right_eye_coordinates[0].z)
        right_eye_corner_2 = (
            right_eye_coordinates[3].x, right_eye_coordinates[3].y, right_eye_coordinates[3].z)

        left_width = math.dist(left_eye_corner_1, left_eye_corner_2)
        right_width = math.dist(right_eye_corner_1, right_eye_corner_2)

        width_average = (left_width + right_width) / 2

        return width_average

    def height_of_eyes(self):
        if not self.is_face_detected():
            return None

        left_eye_coordinates = self.left_eye()
        right_eye_coordinates = self.right_eye()

        left_eye_center_bottom = (
            left_eye_coordinates[1].x, left_eye_coordinates[1].y)
        left_eye_center_top = (
            left_eye_coordinates[2].x, left_eye_coordinates[2].y)

        right_eye_center_bottom = (
            right_eye_coordinates[1].x, right_eye_coordinates[1].y)
        right_eye_center_top = (
            right_eye_coordinates[2].x, right_eye_coordinates[2].y)

        left_height = math.dist(left_eye_center_bottom, left_eye_center_top)
        right_height = math.dist(right_eye_center_bottom, right_eye_center_top)

        avg_height = (left_height + right_height) / 2

        return (avg_height, left_height, right_height)

    def horizontal_position_of_iris_in_eye(self):
        if not self.is_face_detected():
            return None

        left_pupil = self.left_iris()[0]
        right_pupil = self.right_iris()[0]

        left_pupil_center = (left_pupil.x, left_pupil.y, left_pupil.z)
        right_pupil_center = (right_pupil.x, right_pupil.y, right_pupil.z)

        left_eye = self.left_eye()
        right_eye = self.right_eye()

        left_eye_corner_left = (left_eye[0].x, left_eye[0].y, left_eye[0].z)
        right_eye_corner_right = (
            right_eye[3].x, right_eye[3].y, right_eye[3].z)

        left_dist = math.dist(left_eye_corner_left, left_pupil_center)
        right_dist = math.dist(right_eye_corner_right, right_pupil_center)

        iris_width = self.width_of_iris()
        eye_width = self.width_of_eyes()

        left_dist_position = 200 * \
            (left_dist - iris_width/2) / (eye_width - iris_width) - 50
        right_dist_position = 200 * \
            (right_dist - iris_width/2) / (eye_width - iris_width) - 50

        avg_dist_from_left_corner = (left_dist + right_dist) / 2
        avg_position = 200 * (avg_dist_from_left_corner -
                              iris_width/2) / (eye_width - iris_width) - 50

        return (clamp_percentage_range(avg_position), clamp_percentage_range(left_dist_position), clamp_percentage_range(right_dist_position))

    def vertical_position_of_iris_in_eye(self):
        if not self.is_face_detected():
            return None

        left_eye = self.left_eye()
        right_eye = self.right_eye()

        left_eye_center_bottom = (left_eye[1].x, left_eye[1].y, left_eye[1].z)
        left_eye_center_top = (left_eye[2].x, left_eye[2].y, left_eye[2].z)

        right_eye_center_bottom = (
            right_eye[1].x, right_eye[1].y, right_eye[1].z)
        right_eye_center_top = (right_eye[2].x, right_eye[2].y, right_eye[2].z)

        left_dist = math.dist(left_eye_center_bottom, left_eye_center_top)
        right_dist = math.dist(right_eye_center_bottom, right_eye_center_top)

        face_distance = self.distance_from_camera()

        left_dist_cm = face_distance * left_dist / self.focal_length
        right_dist_cm = face_distance * right_dist / self.focal_length

        left_wide = left_dist_cm >= 0.025
        left_regular = left_dist_cm >= 0.017
        left_narrow = left_dist_cm >= 0.010

        right_wide = right_dist_cm >= 0.025
        right_regular = right_dist_cm >= 0.017
        right_narrow = right_dist_cm >= 0.010

        return ((left_wide, left_regular, left_narrow), (right_wide, right_regular, right_narrow))

    def yaw(self):
        if not self.is_face_detected():
            return None

        left_eye_corner = self.left_eye()[0]
        right_eye_corner = self.right_eye()[0]

        yaw = calculate_angle(left_eye_corner.x, left_eye_corner.z,
                              right_eye_corner.x, right_eye_corner.z)

        return yaw

    def pitch(self):
        if not self.is_face_detected():
            return None

        face_top = self.silhouette_top()[0]
        face_bottom = self.silhouette_bottom()[0]

        pitch = calculate_angle(face_top.y, face_top.z,
                                face_bottom.y, face_bottom.z)

        return pitch

    def distance_from_camera(self):
        if not self.is_face_detected():
            return None

        iris_diameter = 11.7
        iris_width_px = self.width_of_iris() * self.frame_dim[1]

        depth = 2 * iris_diameter * self.focal_length/iris_width_px

        return depth

    def is_eyelid_open(self):
        left_eye = self.left_eye()
        right_eye = self.right_eye()

        left_eye_center_bottom = (left_eye[1].x, left_eye[1].y, left_eye[1].z)
        left_eye_center_top = (left_eye[2].x, left_eye[2].y, left_eye[2].z)

        right_eye_center_bottom = (
            right_eye[1].x, right_eye[1].y, right_eye[1].z)
        right_eye_center_top = (right_eye[2].x, right_eye[2].y, right_eye[2].z)

        left_dist = math.dist(left_eye_center_bottom, left_eye_center_top)
        right_dist = math.dist(right_eye_center_bottom, right_eye_center_top)

        return (left_dist > 0.007 or right_dist > 0.007, left_dist > 0.007, right_dist > 0.007)

    def is_in_attention_range(self):
        yaw = self.yaw() * 100 * 1.25
        pitch = self.pitch() * 100

        yaw_threshold = 37
        pitch_up_threshold = -43
        pitch_down_threshold = 30

        eyelid_any_open, eyelid_left_open, eyelid_right_open = self.is_eyelid_open()

        _, left_iris_x, right_iris_x = self.horizontal_position_of_iris_in_eye()
        left_iris_y_state, right_iris_y_state = self.vertical_position_of_iris_in_eye()

        iris_x = right_iris_x if (yaw < 0) else left_iris_x
        iris_y_state = right_iris_y_state if (yaw < 0) else left_iris_y_state

        gaze_x = yaw + iris_x - 50
        face_distance = self.distance_from_camera()

        max_attention_angle_x = math.degrees(math.atan(25/face_distance))

        print(f'yaw {yaw},',
              f'pitch {pitch},',
              f'iris pos {iris_x - 50},',
              f'gaze x {gaze_x},',
              f'face dist {face_distance},',
              f'max x ang {max_attention_angle_x}',
              f'iris y {iris_y_state}')

        if (abs(yaw) < yaw_threshold and not eyelid_any_open):
            return (False, "eyes closed")
        elif (yaw <= -yaw_threshold and not eyelid_right_open):
            return (False, "looking right & left eye closed")
        elif (yaw >= yaw_threshold and not eyelid_left_open):
            return (False, "looking left & right eye closed")

        if (pitch < pitch_up_threshold):
            return (False, 'looking up 1')
        elif (pitch > pitch_down_threshold):
            return (False, 'looking down 1')
        elif (pitch >= 10 and pitch <= pitch_down_threshold and not iris_y_state[0]):
            return (False, 'looking down 2')
        elif (pitch >= pitch_up_threshold and pitch <= 0 and iris_y_state[0] and abs(yaw) < 50):
            return (False, 'looking up 2')

        if (abs(iris_x) >= 40 and abs(yaw) >= 50 and abs(yaw) <= 100):
            return (True, '')

        angle_match = abs(gaze_x) < max_attention_angle_x

        return (angle_match, '')


def calculate_angle(a1, a2, b1, b2):
    return math.atan2(b2-a2, b1-a1)


def clamp_percentage_range(value):
    return max(0, min(value, 100))
