import argparse
import datetime
import time
import cv2

dim = (600, 400) # resizing image

vc = cv2.VideoCapture("/home/peter/PycharmProjects/image_processing_assignment_1/coogan.mp4") # video capture [1]

out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))

rval, frame = vc.read()

firstFrame = None

while True:

    if frame is not None:

        rval, frame = vc.read()

    else:
        break

    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)

    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, H = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #for c in contours:
    # compute the bounding box for the contour, draw it on the frame,
    # and update the text
    max_area = 0
    for i in range(len(contours)):  # finding largest contour by area [3]
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            ci = i
    if len(contours) > 0:
        (x, y, w, h) = cv2.boundingRect(contours[ci])
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        moments = cv2.moments(contours[ci])
        if moments['m00'] != 0:  # this gives the centre of the moments [3]
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        center = (cx, cy)
        cv2.circle(resized, center, 5, [0, 0, 255], 2)  # draws small circle at the center moment

    cv2.imshow("FIRST", resized)
    #cv2.imshow("SECOND", firstFrame)
    #cv2.imshow("ABS-DIFF", frameDelta)
    #cv2.imshow("THRESH", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
