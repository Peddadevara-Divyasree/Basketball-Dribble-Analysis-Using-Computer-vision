
### 1) speed calculation using image processing

import cv2
import numpy as np

# Function to calculate distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Read video file
cap = cv2.VideoCapture("C:/Users/SUNEETHA/Desktop/Internshala_assignment/stealth_health/main/WHATSAAP ASSIGNMENT.mp4")

# Initialize variables
prev_ball_position = None
total_distance = 0
frame_count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame (e.g., convert to grayscale, apply Gaussian blur, etc.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform object detection/tracking to locate the basketball
    # You can use methods like Hough Circle Transform or template matching
    # Here we'll just detect a bright object (assuming basketball is brighter than surroundings)
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour corresponds to the basketball
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        ball_position = (int(x), int(y))
        
        # Calculate distance moved by the basketball
        if prev_ball_position:
            total_distance += distance(prev_ball_position, ball_position)
        
        prev_ball_position = ball_position
        
    # Display frame with detected basketball
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Calculate dribble speed
# Assuming a fixed framerate of 30 fps for the video
fps = 30
dribble_speed = total_distance / (fps * frame_count)

print("Average dribble speed:", dribble_speed, "pixels/frame")

# Release resources
cap.release()
cv2.destroyAllWindows()


### 2) speed calculation using yolo model
"""
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:/Users/SUNEETHA/Desktop/Internshala_assignment/stealth_health/main/yolov3.weights", "C:/Users/SUNEETHA/Desktop/Internshala_assignment/stealth_health/main/yolov3.cfg")
classes = []
with open("C:/Users/SUNEETHA/Desktop/Internshala_assignment/stealth_health/main/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Read video file
cap = cv2.VideoCapture('C:/Users/SUNEETHA/Desktop/Internshala_assignment/stealth_health/main/WHATSAAP ASSIGNMENT.mp4')

# Variables for tracking
prev_center = None
speeds = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Processing outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class 0 is basketball
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Calculate speed
                center = (center_x, center_y)
                if prev_center:
                    # Assuming frame rate is 30 fps
                    speed = np.linalg.norm(np.array(center) - np.array(prev_center)) * 30 / 1000  # Convert pixels per frame to meters per second
                    speeds.append(speed)
                
                prev_center = center
                
    # Show frame with detected basketball
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Calculate average speed
average_speed = np.mean(speeds)
print("Average dribble speed:", average_speed, "m/s")

# Release resources
cap.release()
cv2.destroyAllWindows()

"""
