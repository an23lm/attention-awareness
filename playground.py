import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Face Mesh model
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face landmarks
        results = face_mesh.process(image)

        # Draw the face landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        thickness=1, color=(255, 255, 255)))

            # Get the position of the eyes and calculate the eye aspect ratio (EAR)
            left_eye = [
                results.multi_face_landmarks[0].landmark[159].x,
                results.multi_face_landmarks[0].landmark[159].y
            ]
            right_eye = [
                results.multi_face_landmarks[0].landmark[386].x,
                results.multi_face_landmarks[0].landmark[386].y
            ]
            eye_distance = ((left_eye[0] - right_eye[0])
                            ** 2 + (left_eye[1] - right_eye[1])**2)**0.5
            eye_aspect_ratio = eye_distance / \
                ((image.shape[0] + image.shape[1]) / 2)

            # Determine if the user is looking at the screen based on the eye aspect ratio
            if eye_aspect_ratio > 0.3:
                cv2.putText(image, "Looking at screen", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Not looking at screen", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Mediapipe Face Mesh", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
