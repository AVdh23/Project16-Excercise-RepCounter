import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)

# Weightlifting exercise variables
hands_up = False
count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of the hands
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Get coordinates of the shoulders
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate middle point between left hand and left shoulder
            left_middle = [(left_wrist[0] + left_shoulder[0]) / 2,
                           (left_wrist[1] + left_shoulder[1]) / 2]

            # Calculate middle point between right hand and right shoulder
            right_middle = [(right_wrist[0] + right_shoulder[0]) / 2,
                            (right_wrist[1] + right_shoulder[1]) / 2]

            # Calculate angle between hands
            hand_angle = calculate_angle(left_wrist, right_wrist, [0, 0])

            # Update weightlifting exercise status
            if hand_angle > 120 and left_wrist[1] < 0.9 and right_wrist[1] < 0.9:
                if not hands_up:
                    hands_up = True
                    count += 1
                    print("Count:", count)
                    time.sleep(0.5)  # Add a 0.5-second delay after counting a rep
            else:
                hands_up = False

        except:
            pass

        # Draw circles on hands
        cv2.circle(image, (int(left_wrist[0] * image.shape[1]), int(left_wrist[1] * image.shape[0])),
                   8, (0, 255, 0), thickness=-1)
        cv2.circle(image, (int(right_wrist[0] * image.shape[1]), int(right_wrist[1] * image.shape[0])),
                   8, (0, 255, 0), thickness=-1)

        # Draw circles on middle points of hands
        cv2.circle(image, (int(left_middle[0] * image.shape[1]), int(left_middle[1] * image.shape[0])),
                   8, (255, 0, 0), thickness=-1)
        cv2.circle(image, (int(right_middle[0] * image.shape[1]), int(right_middle[1] * image.shape[0])),
                   8, (255, 0, 0), thickness=-1)

        # Draw circles on shoulders
        cv2.circle(image, (int(left_shoulder[0] * image.shape[1]), int(left_shoulder[1] * image.shape[0])),
                   8, (0, 0, 255), thickness=-1)
        cv2.circle(image, (int(right_shoulder[0] * image.shape[1]), int(right_shoulder[1] * image.shape[0])),
                   8, (0, 0, 255), thickness=-1)

        # Display hand movement status and count
        if hands_up:
            cv2.putText(image, "Hands Up", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Hands Down", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the count in the top left corner
        cv2.putText(image, "Count: " + str(count), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Weightlifting Exercise', image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
