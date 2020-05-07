import argparse
import datetime
import time
import csv
import cv2
import numpy as np
import math
import xlrd
import matplotlib.pyplot as plt

vc = cv2.VideoCapture("lubin.mp4")  # video capture [1]
width = np.int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = np.int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
Nframes = np.int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image
last_position = [0] * 100

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)
(check, frame) = vc.read()
points_to_track = [0] *30
K = 20
f = 0
prevFrame = 0
lk_params = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.01))
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
num_frames = 0
certain_frames = 0
paper = 0
rock = 0
scissors = 0
def_rock = 0
def_paper = 0
def_scissors = 0
label_test = 3
correct_frames = 0

loc = ("test_labels.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

while check:
    if frame is not None:
        num_frames = num_frames + 1
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        # Get pointer to video frames from primary device
        imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB)
        roi = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        thresh = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, shape)
        thresh1 = cv2.bitwise_and(resized, resized, mask=thresh)

        contours, H = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        max_area = 0
        k = 0
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
            hull2 = cv2.convexHull(contours[ci], returnPoints=False)
            defects = cv2.convexityDefects(contours[ci], hull2)
            num_def = 0
            #print(defects)

            if defects is not None:
                for i in range(defects.shape[0]):
                    if defects.any():
                        #print("defect no. ", i)
                        s, e, f, d = defects[i, 0]
                        start = tuple(contours[ci][s][0])
                        end = tuple(contours[ci][e][0])
                        far = tuple(contours[ci][f][0])

                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                        # if angle > 90 draw a circle at the far point
                        if angle <= 90:
                            num_def += 1
                            cv2.circle(resized, far, 5, [255, 0, 0], -1)

                        cv2.line(num_def, start, end, [0, 255, 0], 2)
                        cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                    else:
                        cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                if num_def == 4:
                    paper = paper + 1
                    rock = 0
                    scissors = 0
                    label_test = 1
                    if paper == 10:
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 10
                    elif paper > 10:
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 1
                    else:
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 2:
                    paper = 0
                    rock = 0
                    scissors = scissors + 1
                    label_test = 2
                    if scissors == 10:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 10
                    elif scissors > 10:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 1
                    else:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 0:
                    paper = 0
                    rock = rock + 1
                    scissors = 0
                    label_test = 0
                    if rock == 10:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 10
                    elif rock > 10:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 1
                    else:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    paper = 0
                    rock = 0
                    scissors = 0
                    label_test = 3
                    certain_frames = certain_frames + 1

                if label_test == sheet.cell_value(num_frames, 1):
                    cv2.putText(resized, 'Label Matched', (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    correct_frames = correct_frames + 1

            else:
                cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

        cv2.imshow("FIRST", resized)
        # cv2.imshow("second", thresh1)
        frame = resized
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rval, frame = vc.read()

    #except:
    #    continue

    else:
        print("done")
        print(num_frames)
        print(certain_frames)
        print(correct_frames)
        percent_certain = (certain_frames/num_frames) * 100
        print(percent_certain)
        percent_correct = (correct_frames/num_frames) * 100
        print(percent_correct)
        break

vc.release()
out.release()
