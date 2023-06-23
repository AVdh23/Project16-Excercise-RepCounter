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
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            middle_hand_left = [landmarks[mp_pose.PoseLandmark.LEFT_MIDDLE_FINGER_MCP].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_MIDDLE_FINGER_MCP].y]
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]

            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            middle_hand_right = [landmarks[mp_pose.PoseLandmark.RIGHT_MIDDLE_FINGER_MCP].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_MIDDLE_FINGER_MCP].y]
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

            # Calculate angle between hands
            if wrist_left and middle_hand_left and shoulder_left:
                hand_angle_left = calculate_angle(wrist_left, middle_hand_left, shoulder_left)
                if hand_angle_left > 120 and hand_angle_left < 180:
                    if not hands_up:
                        hands_up = True
                        count += 1
                        print("Count:", count)
                        time.sleep(1)  # Add a 1-second delay after counting a rep
                else:
                    hands_up = False

            if wrist_right and middle_hand_right and shoulder_right:
                hand_angle_right = calculate_angle(wrist_right, middle_hand_right, shoulder_right)
                if hand_angle_right > 120 and hand_angle_right < 180:

                    if not hands_up:
                        hands_up = True
                        count += 1
                        print("Count:", count)
                        time.sleep(1)  # Add a 1-second delay after counting a rep
                else:
                    hands_up = False

        except:
            pass

        # Draw lines and points
        if results.pose_landmarks:
            # Draw connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                      )

            # Draw points
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4)
                                      )

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
