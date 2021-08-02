import threading
import time
import cv2
import numpy as np
from threading import Thread

# detecting modules 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"] 

prtxt = 'models/MobileNetSSD_deploy.prototxt.txt'
model = 'models/MobileNetSSD_deploy.caffemodel'  
net = cv2.dnn.readNet(prtxt, model) 

# tracking tachniques
tracker = cv2.legacy.MultiTracker_create()

def detecting(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),	0.007843, (300, 300), 127.5) 
    net.setInput(blob)
    detections = net.forward() 

    do = True
    for i in range(frame.shape[2]):
        
        if len(detections) > 0:
            
            #i = np.argmax(detections[0, 0, :, 2])
        
            conf = detections[0, 0, i, 2]
            label = CLASSES[int(detections[0, 0, i, 1])]

            if conf > .75 and (label == 'person' or label == 'car'):
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                location = (startX, startY, endX, endY)
                # draw the bounding box and text for the object
                do = False
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)  
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)

                tracker.add(cv2.legacy.TrackerMOSSE_create(),frame, location) 

    return do       
    
def tracking(frame, counter):
    (h, w) = frame.shape[:2]
    (success, boxes) = tracker.update(frame)

    # check to see if the tracking was a success
    if success:
        for box in boxes:
            (startX, startY, endX, endY) = [int(v) for v in box] 
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(frame, 'ID:', (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)   
            xMid = int((startX + endX)/2)
            yMid = int((startY + endY)/2)
            cv2.circle(frame, (xMid,yMid), 1, (0,0,255), 3)
            if yMid == int(h/1.7):
                counter += 1
        return counter




if __name__ == '__main__':
    
    video = cv2.VideoCapture('Videos/IMG_4954.MOV')
    time.sleep(2)
    
    counter = 0
    do = True

    frame_rate = 30
    prev = 0

    while True:
        
        time_elapsed = time.time() - prev
        status , frame = video.read()                

        if status:

            if time_elapsed > 1./frame_rate: # for reducing the frame rate 
                prev = time.time()
                h, w = frame.shape[:2]
                
                # for detetcting the object only ones
                if do:
                    do = detecting(frame) 
                    
            
                counter = tracking(frame, counter)
            

                cv2.line(frame, (w, int(h/1.7)), (w-w,int(h/1.7)), (255, 0, 0), 2)

                cv2.putText(frame, 'number of cars {}'.format(counter), (10, h-10),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                cv2.imshow('Video', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break


cv2.destroyAllWindows()


