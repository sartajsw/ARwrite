# import the necessary packages
from collections import deque
import numpy as np
import cv2
import imutils
import time
 
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (30, 30, 120)
greenUpper = (50, 255, 255)
l = 1024
pts = deque(maxlen=l)
then = time.time()
#kernelOpen=np.ones((5,5))
#kernelClose=np.ones((20,20))

# if a video path was not supplied, grab the reference
# to the webcam
vs = cv2.VideoCapture(0)
 
# allow the camera or video file to warm up
time.sleep(2.0)
# keep looping

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame[1]
 
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
 
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
#    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
#    mask=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    _,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # only proceed if the radius meets a minimum size       
        if radius > 25:        
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            then = time.time()

#        else
    now = time.time()
    diff = int(now-then)
    if diff>5:
        pts = deque()

#            if diff > 10 and len(pts)!=0:
#                print("Object Removed")
#                print("**************")                
#                print(diff)                
#                print("**************")
#                pts = deque()
#                then = time.time()
 
    # update the points queue
    pts.appendleft(center)
     # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
 
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(l/ float(i + 1)) * 0.0125 * radius)
        if thickness<3:
            thickness = 3
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
    # show the frame to our screen
    frame = cv2.flip(frame, 1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 

vs.release()
 
# close all windows
cv2.destroyAllWindows()
