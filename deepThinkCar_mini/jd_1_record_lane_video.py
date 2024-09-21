'''
1. importing necessary modules
'''
import cv2
import os 
from jd_opencv_lane_detect import JdOpencvLaneDetect

'''
2. Creating object from classes
  1) OpenCV lane detecting object from JdOpencvLaneDetect class 
'''
# OpenCV line detector object
cv_detector = JdOpencvLaneDetect()

'''
3. Loading video file instead of using the camera
cv2.VideoCapture() function loads the video file.
'''
# Video file path (replace with the actual path of your video file)
video_path = 'deepThinkCar_mini/prediect/1.AVI'

# Load video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file: {video_path}")
else:
    print(f"Video file {video_path} loaded successfully.")

'''
4. Creating data folder, if not exist
Video file that comes from OpenCV driving is saved here. 
'''
# Find ./data folder for labeling data
try:
    if not os.path.exists('./data'):
        os.makedirs('./data')
except OSError:
    print("Failed to make ./data folder")

'''
5. Creating video recording object
In this first step, we record the lane-detected video as an AVI file.
To do this we use cv2.VideoWriter_fourcc() and cv2.VideoWriter()
'''

# Create video codec object. We use 'XVID' format.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Video writer object
video_orig = cv2.VideoWriter('./data/car_video_processed.avi', fourcc, 20.0, (320, 240))

'''
6. Processing the video for lane detection
This part detects lanes in the loaded video file.
'''
for i in range(30):
    ret, img_org = cap.read()
    if ret:
        lanes, img_lane = cv_detector.get_lane(img_org)
        angle, img_angle = cv_detector.get_steering_angle(img_lane, lanes)
        if img_angle is None:
            print("Can't find lane...")
            pass
        else:
            print(angle)
    else:
        print("Video loading error")
        break
        
'''
7. Perform lane detection on the entire video
When you press the 'q' key, it stops the video processing.
During processing, the video is saved with lane detection results. 
'''
# Lane detection routine
while True:
    ret, img_org = cap.read()
    if ret:
        # Save the original video frame
        video_orig.write(img_org)
        
        # Find lane angle
        lanes, img_lane = cv_detector.get_lane(img_org)
        angle, img_angle = cv_detector.get_steering_angle(img_lane, lanes)
        
        if img_angle is None:
            print("Can't find lane...")
            pass
        else:
            # Display the lane-detected image
            cv2.imshow('lane', img_angle)
            print(angle)
            
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("End of video or error reading frame.")
        break

'''
8. Finishing the video processing
Releasing occupied resources
'''
cap.release()
video_orig.release()
cv2.destroyAllWindows()
