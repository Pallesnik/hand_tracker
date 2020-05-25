"""
Flesh detector
Author: Peter Kinsella, C16338263
Date: 24-05-2020

Process:
1. Load in the ground truth table and video
2. Convert frame to YCbCr colour space and threshold
3. Segment hand from the rest of the frame
4. Get the contours of the hand and sort contours by area to get largest one
5. Get Convex hull and convexity defects of largest contour
6. Use cosine rule to reduce defects down to only the ones in between fingers
7. print and count defects
8. Using number of defects determine the hand gesture being shown
9. Gather the main feature points of the hand using Shi-Tomasi method
10. Track points using Optical Flow method via Lucas-Kanade
11. Print final results for analysis
"""
import cv2
import numpy as np
import math
import time

time1 = int(round(time.time()))
vc = cv2.VideoCapture("lubin.mp4")  # video capture
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)  # write to an output mp4
(check, frame) = vc.read()
f = 0
# parameters for feature tracking
lk_params = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.01))
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
num_frames = 0 # initialisation of variables for testing
certain_frames = 0
paper = 0
rock = 0
scissors = 0

while check: # while video is being processed
    if frame is not None: # if there is a frame to process
        num_frames = num_frames + 1
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # shape used for morphology

        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) # resizing image
        resized2 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        min_YCrCb = np.array([0, 133, 77], np.uint8) # flesh detecton value range
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        # Get pointer to video frames from primary device
        imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB) # colour space conversion to YCrCb
        roi = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb) # Thresholding the image
        thresh = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, shape) # cleaning up the mask with a close morphology
        thresh1 = cv2.bitwise_and(resized, resized, mask=thresh) # ANDing the mask and the image

        contours, H = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # getting all contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True) # sorting contours
        max_area = 0

        for i in range(len(contours)):  # finding largest contour by area [3]
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                ci = i
        if len(contours) > 0: # if more than 0 contours
            moments = cv2.moments(contours[ci]) # get the moments of the largest contour
            if moments['m00'] != 0:  # this gives the centre of the moments [3]
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
            center = (cx, cy)
            cv2.circle(resized, center, 5, [0, 0, 255], 2)  # draws small circle at the center moment
            hull = cv2.convexHull(contours[ci]) # getting hull of the contour
            hull2 = cv2.convexHull(contours[ci], returnPoints=False)
            defects = cv2.convexityDefects(contours[ci], hull2) # getting all defects in the hull
            num_def = 0
            if defects is not None: # if there are defects
                for i in range(defects.shape[0]): # cycle through all defects
                    if defects.any():
                        s, e, f, d = defects[i, 0]  # get the start end and far points the defect
                        start = tuple(contours[ci][s][0])
                        end = tuple(contours[ci][e][0])
                        far = tuple(contours[ci][f][0])
                        # get the angle created by the start, end and far points
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                        # if angle > 90 draw a circle at the far point
                        if angle <= 90:
                            num_def += 1
                            cv2.circle(resized, far, 5, [255, 0, 0], -1)

                        cv2.line(num_def, start, end, [0, 255, 0], 2) # draw the hull and contour
                        cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                    #else:  # if there were no defects draw hull and contour
                        #cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        #cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                if num_def == 4:  # check number of defects and depending on number print label to output
                    paper = paper + 1
                    rock = 0
                    scissors = 0
                    certain_frames = certain_frames + 1
                    if paper >= 10:  # if detects paper 10 times in a row or more print it in green
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    else:  # else print in red
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 2:
                    paper = 0
                    rock = 0
                    scissors = scissors + 1
                    certain_frames = certain_frames + 1
                    if scissors >= 10:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    else:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 0:
                    paper = 0
                    rock = rock + 1
                    scissors = 0
                    certain_frames = certain_frames + 1
                    if rock >= 10:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    else:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    paper = 0
                    rock = 0
                    scissors = 0

            else:
                cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

        G = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)  # make a grayscale copy of image

        # get main feature points of image
        dots = cv2.goodFeaturesToTrack(G, maxCorners=500, qualityLevel=0.000001, minDistance=10)

        Z = np.float32(dots)
        Z = np.transpose(Z)

        # number of dots:
        N = np.shape(dots)[0]

        # drawing each dot:
        for n in range(0, N):
            j = int(dots[n, :, 0][0])
            i = int(dots[n, :, 1][0])

        # if this is not the first frame:
        if f > 0:
            # delete the dots:
            del dots
            # Calculate optical flow:
            dots, status, error = cv2.calcOpticalFlowPyrLK(prevImg=prevFrame, nextImg=G, prevPts=prevPts,
                                                           nextPts=None, **lk_params)

            # number of dots:
            N = np.shape(dots)[0]

            # If there are dots to draw:
            if N > 0:
                # drawing each dot:
                for n in range(0, N):
                    j = int(dots[n, :, 0][0])
                    i = int(dots[n, :, 1][0])

                    if status[n][0] == 0:
                        # if it didn't work, go back to the previous point:
                        dots[n] = prevPts[n]

                    if status[n][0] == 1:
                        #Make the successful Lucas-Kanade dots red:
                        resized[i - 3:i + 3, j - 3:j + 3] = (0, 0, 255)

        # Get the current frame and dots ready for the next round:
        prevPts = dots.copy()
        prevFrame = G.copy()

        cv2.imshow("Hand", resized)  # display final image
        #frame = cv2.flip(resized, 0) # flip image for output
        out.write(resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rval, frame = vc.read()

    else: # if no frames left end while loop
        print("done")
        time2 = int(round(time.time()))
        print("Elapsed Time",time2-time1)
        break

vc.release()
out.release()  # release an output mp4 file
