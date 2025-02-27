import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone

url = "#########################################/video"

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if(event==cv2.EVENT_MOUSEMOVE):
        point = [x,y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('test_4.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# out = cv2.VideoWriter('output.avi',fourcc, 5, (640,480))

output = cv2.VideoWriter('output_final.avi',cv2.VideoWriter_fourcc(*'MPEG'),30,(1020,500))

file = open('coco.names', 'r')
data = file.read()
class_list = data.split('\n')

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cy1=194
cy2=220
offset=6

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame=stream_read()

    count+=1
    if (count%3 != 0):
        continue

    frame=cv2.resize(frame, (1020,500))

    results=model.predict(frame)
    #print(results)

    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #print(px)

    list=[]

    for index,row in px.iterrows():
        #print(rows)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)

        ## for down going
        if (cy1<(cy+offset) and (cy1>cy-offset)):

            cv2.rectangle(frame, (x3,y3),(x4,y4),(0,0,255),2)
            cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
            persondown[id]=(cx,cy)

        if (id in persondown):
            if (cy2<(cy+offset) and (cy2>cy-offset)):
                cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,255),2)
                cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
                if counter1.count(id)==0:
                    counter1.append(id)
        
        ## for up going
        if (cy2<(cy+offset) and (cy2>cy-offset)):

            cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,0),2)
            cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
            personup[id]=(cx,cy)

        if (id in personup):
            if (cy1<(cy+offset) and (cy1>cy-offset)):
                cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,255),2)
                cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
                if counter2.count(id)==0:
                    counter2.append(id)

    cv2.line(frame,(3,cy1), (1018,cy1),(0,255,0),2)
    cv2.line(frame,(5,cy2), (1019,cy2),(0,255,255),2)

    #print(persondown)
    #print(counter1)
    #print(len(counter1)) #lenght means we can get the counnt who is going down
    downcount=len(counter1)
    upcount=len(counter2)
    
    cvzone.putTextRect(frame, f'Entry: {downcount}', (50,60), 2,2)
    cvzone.putTextRect(frame, f'Exit: {upcount}', (50,160), 2,2)
    output.write(frame)
    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xff==27:
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import pandas as pd
# from ultralytics import YOLO
# from tracker import Tracker
# import cvzone

# # Path to the video file (replace with your video URL or path)
# video_path = "3.mp4"

# # Load the YOLOv8 model
# model = YOLO('yolov8s.pt')

# # Read the class names from the coco.names file
# with open('coco.names', 'r') as file:
#     class_list = file.read().split('\n')

# # Video capture
# cap = cv2.VideoCapture(video_path)
# output = cv2.VideoWriter('output_final.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, (1020, 500))

# # Initialize tracker and counters
# tracker = Tracker()
# counter_down = []
# counter_up = []

# # Define reference lines for counting
# cy1 = 194  # Line for down counting
# cy2 = 220  # Line for up counting
# offset = 6  # Error margin

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize the frame
#     frame = cv2.resize(frame, (1020, 500))

#     # Predict with YOLOv8
#     results = model.predict(frame)
#     detections = results[0].boxes.data
#     detections_df = pd.DataFrame(detections).astype("float")

#     # Extract person detections
#     person_boxes = []
#     for _, row in detections_df.iterrows():
#         x1, y1, x2, y2, _, class_id = map(int, row[:6])
#         if class_list[class_id] == "person":
#             person_boxes.append([x1, y1, x2, y2])

#     # Update tracker with person detections
#     tracked_objects = tracker.update(person_boxes)

#     # Draw bounding boxes and count people crossing lines
#     for obj in tracked_objects:
#         x1, y1, x2, y2, obj_id = obj
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#         # Draw the bounding box and center
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
#         cvzone.putTextRect(frame, f'{obj_id}', (x1, y1), 1, 2)

#         # Count downwards movement
#         if cy1 - offset < cy < cy1 + offset:
#             if obj_id not in counter_down:
#                 counter_down.append(obj_id)

#         # Count upwards movement
#         if cy2 - offset < cy < cy2 + offset:
#             if obj_id not in counter_up:
#                 counter_up.append(obj_id)

#     # Draw the reference lines
#     cv2.line(frame, (0, cy1), (1020, cy1), (0, 255, 0), 2)  # Down line
#     cv2.line(frame, (0, cy2), (1020, cy2), (0, 255, 255), 2)  # Up line

#     # Display counts
#     cvzone.putTextRect(frame, f'Down: {len(counter_down)}', (50, 60), 2, 2)
#     cvzone.putTextRect(frame, f'Up: {len(counter_up)}', (50, 160), 2, 2)

#     # Write frame to output video
#     output.write(frame)

#     # Display the frame
#     cv2.imshow('RGB', frame)

#     # Break on pressing 'ESC'
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # Release resources
# cap.release()
# output.release()
# cv2.destroyAllWindows()
