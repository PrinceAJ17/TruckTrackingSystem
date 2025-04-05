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
capture = cv2.VideoCapture("../SampleTruck.mp4")

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

    #Creating an empty list for detected trucks
    detections = []

    #For each result "r" in results
    for r in results:
        #For each box from all the boxes of the results "r"
        for box in r.boxes:
            #Store the ID of each detected object
            cls = int(box.cls[0])
            #Store the confidence level of each object
            conf = float(box.conf[0])
            #From the objects deteced, store only trucks using their unique class ID (cls)
            if r.names[cls] == "truck":
                #Extract bounding box coordinates
                #Store upper left(x1,y1) and bottom right (x2,y2) coordinates and map them to be integers
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                shrink = 20
                x1+=shrink
                y1+=shrink
                x2-=shrink
                y2-=shrink
                #Add the detected bounding box and confidence to the list of detected trucks 
                detections.append(([x1,y1,x2,y2], conf, None))
    #Pass detected trucks with frames for tracking
    #Get back all object tracks with unique IDs
    tracks = tracker.update_tracks(detections, frame=frame)

    #For each detected tracked truck
    for track in tracks:
        #DeepSORT checks for unstable objects and only passes confirmed objects
        if track.is_confirmed():
            #Gets UID for truck
            track_id = track.track_id
            #Get bounding box coordinates
            ltrb = track.to_ltrb()
            x1,y1,x2,y2 = map(int, ltrb)
            shrink = 20
            x1+=shrink
            y1+=shrink
            x2-=shrink
            y2-=shrink
            #Draws green box around the truck
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            #Writes green text of Truck ID
            cv2.putText(frame, f"Truck {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Truck Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()