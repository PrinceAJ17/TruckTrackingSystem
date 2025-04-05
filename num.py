#Importing object detection model
from ultralytics import YOLO
#Importing multi-object tracking class
from deep_sort_realtime.deepsort_tracker import DeepSort
#Importing OpenCV for video capturing
import cv2

#Pretrained model that can detect trucks
model = YOLO("yolov8n.pt")
#Initializing DeepSORT tracker with few parameters
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
#max_age refers to amount of frames required for a tracked object to go undetected before DeepSORT forgets it
#n_init refers to the amount of frames required to be detected until an object is tracked and assigned an ID
#nms_max_overlap determines how much overlap between bounding boxes are allowed before they're considered dupes

#Video capture ovject created to read from recording
capture = cv2.VideoCapture("SampleTruck.mp4")

#Creates a loop as long as the video feed is open
while capture.isOpened():
    #One frame is read through the video 
    #ret is the boolean value (True if frame red correctly) and frame is the image captured
    ret, frame = capture.read()
    #If loop created where if frame not captured properlt, then video feed closes while loop breaks
    #else the model (YOLOv8n) runs object detection on the frame and stores the results in an object
    if(ret == "False"):
        break
    else:
        results = model(frame)