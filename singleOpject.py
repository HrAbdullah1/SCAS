import cv2


tracker = cv2.legacy.TrackerMOSSE_create()

initB = None
fps = cv2.CAP_PROP_FPS
video = cv2.VideoCapture(0)#'Videos/IMG_4954.MOV')

while video.isOpened():

    _ , frame = video.read()

    (H, W) = frame.shape[:2]

    if initB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box] 
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

        info = [
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps))
            ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)   
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('video', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    if  cv2.waitKey(25) & 0xFF == ord("s"):
        #select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initB)
        print(initB)
        
video.release()
cv2.destroyAllWindows()