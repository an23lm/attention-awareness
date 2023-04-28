import cv2
import dlib
import numpy as np

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

# Initialize dlib's facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the eye landmarks and the threshold for eye aspect ratio (EAR)
left_eye_landmarks = list(range(36, 42))
right_eye_landmarks = list(range(42, 48))
EAR_THRESHOLD = 0.25


def eye_aspect_ratio(eye_pts):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the image from BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks for the current face
        landmarks = predictor(gray, face)

        # Extract the left and right eye landmarks from the facial landmarks
        left_eye_pts = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_landmarks])
        right_eye_pts = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_landmarks])

        # Calculate the eye aspect ratio (EAR) for each eye
        left_eye_EAR = eye_aspect_ratio(left_eye_pts)
        right_eye_EAR = eye_aspect_ratio(right_eye_pts)
        avg_EAR = (left_eye_EAR + right_eye_EAR) / 2

        # Determine if the user is looking at the screen based on the average EAR
        if avg_EAR > EAR_THRESHOLD:
            cv2.putText(image, "Looking at screen", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Not looking at screen", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw the eye landmarks on the image
        cv2.polylines(image, [left_eye_pts], True, (0, 0, 255), 1)
        cv2.polylines(image, [right_eye_pts], True, (0, 0, 255), 1)

    cv2.imshow("dlib Eye Gaze", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
