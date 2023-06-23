import cv2
import numpy as np

# Load the YOLO weights and configuration
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Set the learning rate of the background subtractor (0 to 1)
learning_rate = 0.01

# Minimum confidence threshold for detected objects
min_confidence = 0.5

# Function to detect abandoned bags in a video
def detect_abandoned_bags(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize variables
    frames_without_detection = 0
    abandoned_bags = []

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            # End of video
            break

        # Apply background subtraction
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # Perform morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour
        for contour in contours:
            # Calculate the contour area
            area = cv2.contourArea(contour)

            # If the contour area exceeds the minimum area threshold, consider it as a potential bag
            if area > min_contour_area:
                # Get the bounding rectangle coordinates
                (x, y, w, h) = cv2.boundingRect(contour)

                # Check if the bag is already marked as abandoned
                if (x, y, w, h) not in abandoned_bags:
                    # Add the bag to the abandoned bags list
                    abandoned_bags.append((x, y, w, h))
                    frames_without_detection = 0
                else:
                    frames_without_detection = 0
            else:
                frames_without_detection += 1

        # Check if bags have not been detected for a certain number of frames
        for bag in abandoned_bags:
            if frames_without_detection >= min_frames_without_detection:
                # Mark the bag as abandoned
                (x, y, w, h) = bag
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Abandoned Bag Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the windows
    video.release()
    cv2.destroyAllWindows()

# Path to the video file you want to analyze
video_path = r"C:\Users\Acer pc\Downloads\abandoned-object-detection-master\abandoned-object-detection-master\video1.avi"

# Call the function to detect abandoned bags in the video
detect_abandoned_bags(video_path)
