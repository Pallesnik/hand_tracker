import argparse
import datetime
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

vc = cv2.VideoCapture("lubin.mp4")  # video capture [1]

width = np.int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = np.int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
Nframes = np.int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)
(check, frame) = vc.read()

while (check):
    try:
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        # Get pointer to video frames from primary device
        imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB)
        thresh = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

        thresh1 = cv2.bitwise_and(resized, resized, mask=thresh)

        contours, H = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # for c in contours:
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
            # cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            moments = cv2.moments(contours[ci])
            if moments['m00'] != 0:  # this gives the centre of the moments [3]
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
            center = (cx, cy)
            cv2.circle(resized, center, 5, [0, 0, 255], 2)  # draws small circle at the center moment
            hull = cv2.convexHull(contours[ci])
            defects = cv2.convexityDefects(contours[ci], hull)

            if len(defects) > 0:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contours[ci][s][0])
                    end = tuple(contours[ci][e][0])
                    far = tuple(contours[ci][f][0])
                    cv2.line(resized, start, end, [0, 255, 0], 2)
                    cv2.circle(resized, far, 5, [0, 0, 255], -1)
            else:
                cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

        cv2.imshow("FIRST", resized)
        # cv2.imshow("SECOND", firstFrame)
        # cv2.imshow("ABS-DIFF", frameDelta)
        # cv2.imshow("THRESH", thresh)
        frame = cv2.flip(resized, 0)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rval, frame = vc.read()

    except:
        break

vc.release()
out.release()
