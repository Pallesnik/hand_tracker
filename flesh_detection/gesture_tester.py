"""
Gesture Tester
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
9. Check if the guessed gesture matches that of the ground truth table
10. Count up correct and incorrect guesses for analysis later
11. Print final results for analysis
"""
import cv2
import numpy as np
import math
import xlrd

vc = cv2.VideoCapture("lubin.mp4")  # video capture [1]
width = np.int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = np.int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
Nframes = np.int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)
(check, frame) = vc.read()
prevFrame = 0
lk_params = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.01))
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
num_frames = 0                              # Initializing test values
certain_frames = 0
paper, rock, scissors, nothing = 0, 0, 0, 0
label_paper, label_rock, label_scissors, label_nothing = 0, 0, 0, 0
def_rock, def_paper, def_scissors = 0, 0, 0
label_test = 3
correct_frames = 0
true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
true_rock, true_paper, true_scissors, true_nothing = 0, 0, 0, 0
false_rock, false_paper, false_scissors, false_nothing = 0, 0, 0, 0
p_not_r, s_not_r, n_not_r = 0, 0, 0
r_not_p, s_not_p, n_not_p = 0, 0, 0
p_not_s, r_not_s, n_not_s = 0, 0, 0
p_not_n, s_not_n, r_not_n = 0, 0, 0
certainty_level = 10

loc = ("test_labels.xlsx") # loading in ground truth table
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

while check: # while the video is being processed
    if frame is not None: # if there are frames to process
        num_frames = num_frames + 1
        shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # shape used for morphology

        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) # resize image

        min_YCrCb = np.array([0, 133, 77], np.uint8) # colour space threshold range
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB) # converting colour space
        roi = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb) # thresholding and morphology
        thresh = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, shape)
        thresh1 = cv2.bitwise_and(resized, resized, mask=thresh)

        contours, H = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # gathering contours

        contours = sorted(contours, key=cv2.contourArea, reverse=True) # sorting contours

        max_area = 0
        k = 0
        for i in range(len(contours)):  # finding largest contour by area [3]
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                ci = i
        if len(contours) > 0: # if there are contours to test
            moments = cv2.moments(contours[ci]) # get the moments of the contour
            if moments['m00'] != 0:  # this gives the centre of the moments [3]
                cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
            center = (cx, cy)
            cv2.circle(resized, center, 5, [0, 0, 255], 2)  # draws small circle at the center moment
            hull = cv2.convexHull(contours[ci]) # getting convex hill of contour
            hull2 = cv2.convexHull(contours[ci], returnPoints=False)
            defects = cv2.convexityDefects(contours[ci], hull2) # gathering the defects
            num_def = 0

            if defects is not None: # if there are defects to analyse
                for i in range(defects.shape[0]):
                    if defects.any():
                        s, e, f, d = defects[i, 0] # get the start, end and far points of each defect
                        start = tuple(contours[ci][s][0])
                        end = tuple(contours[ci][e][0])
                        far = tuple(contours[ci][f][0])

                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) # use law of cosines to get angle
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                        # if angle > 90 draw a circle at the far point
                        if angle <= 90:
                            num_def += 1
                            cv2.circle(resized, far, 5, [255, 0, 0], -1)

                        cv2.line(num_def, start, end, [0, 255, 0], 2) # draw contour and hull
                        cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                    else:
                        cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                        cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

                # Test number of defects to determine which gesture is being shown
                if num_def == 4:
                    paper = paper + 1
                    rock = 0
                    scissors = 0
                    nothing = 0
                    label_test = 1
                    if paper == certainty_level: # level of certainty for displaying
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + certainty_level
                        false_pos += certainty_level # adding false positive values that will be removed if guess matches label
                        false_paper += certainty_level
                    elif paper > certainty_level:
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 1
                        false_pos += 1
                        false_paper += 1
                    else:
                        cv2.putText(resized, 'Paper', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 2:
                    paper = 0
                    rock = 0
                    scissors = scissors + 1
                    nothing = 0
                    label_test = 2
                    if scissors == certainty_level:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + certainty_level
                        false_pos += certainty_level
                        false_scissors += certainty_level
                    elif scissors > certainty_level:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + 1
                        false_pos += 1
                        false_scissors += 1
                    else:
                        cv2.putText(resized, 'Scissors', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif num_def == 0:
                    paper = 0
                    rock = rock + 1
                    scissors = 0
                    nothing = 0
                    label_test = 0
                    if rock == certainty_level:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        certain_frames = certain_frames + certainty_level
                        false_pos += certainty_level
                        false_rock += certainty_level
                    elif rock > certainty_level:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        false_pos += 1
                        certain_frames = certain_frames + 1
                        false_rock += 1
                    else:
                        cv2.putText(resized, 'Rock', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    paper = 0
                    rock = 0
                    scissors = 0
                    nothing = nothing + 1
                    label_test = 3
                    if nothing == certainty_level:
                        false_pos += certainty_level
                        false_nothing += certainty_level
                    elif nothing > certainty_level:
                        false_pos += 1
                        false_nothing += 1

                # if labels match guess, print an output as well as remove false positive values
                # and add true positive values
                if label_test == sheet.cell_value(num_frames, 1):
                    cv2.putText(resized, 'Label Matched', (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    correct_frames = correct_frames + 1
                    if label_test == 0:
                        label_paper, label_scissors, label_nothing = 0, 0, 0
                        label_rock += 1
                        if label_rock == certainty_level:
                            true_pos += certainty_level
                            true_rock += certainty_level
                            false_pos -= certainty_level
                            false_rock -= certainty_level
                        elif label_rock > certainty_level:
                            true_pos += 1
                            true_rock += 1
                            false_pos -= 1
                            false_rock -= 1

                    elif label_test == 1:
                        label_rock, label_scissors, label_nothing = 0, 0, 0
                        label_paper += 1
                        if label_paper == certainty_level:
                            true_pos += certainty_level
                            true_paper += certainty_level
                            false_pos -= certainty_level
                            false_paper -= certainty_level
                        elif label_paper > certainty_level:
                            true_pos += 1
                            true_paper += 1
                            false_pos -= 1
                            false_paper -= 1

                    elif label_test == 2:
                        label_paper, label_rock, label_nothing = 0, 0, 0
                        label_scissors += 1
                        if label_scissors == certainty_level:
                            true_pos += certainty_level
                            true_scissors += certainty_level
                            false_pos -= certainty_level
                            false_scissors -= certainty_level
                        elif label_scissors > certainty_level:
                            true_pos += 1
                            true_scissors += 1
                            false_pos -= 1
                            false_scissors -= 1

                    elif label_test == 3:
                        label_paper, label_rock, label_scissors = 0, 0, 0
                        label_nothing += 1
                        if label_nothing == certainty_level:
                            true_pos += certainty_level
                            true_nothing += certainty_level
                            false_pos -= certainty_level
                            false_nothing -= certainty_level
                        elif label_nothing > certainty_level:
                            true_neg += 1
                            true_nothing += 1
                            false_neg -= 1
                            false_nothing -= 1

                # if labels did not match add 1 to the value of the actual label on the truth table
                else:
                    if label_test == 0:
                        if sheet.cell_value(num_frames, 1) == 1:
                            p_not_r += 1
                        elif sheet.cell_value(num_frames, 1) == 2:
                            s_not_r += 1
                        elif sheet.cell_value(num_frames, 1) == 3:
                            n_not_r += 1
                    elif label_test == 1:
                        if sheet.cell_value(num_frames, 1) == 0:
                            r_not_p += 1
                        elif sheet.cell_value(num_frames, 1) == 2:
                            s_not_p += 1
                        elif sheet.cell_value(num_frames, 1) == 3:
                            n_not_p += 1
                    elif label_test == 2:
                        if sheet.cell_value(num_frames, 1) == 0:
                            r_not_s += 1
                        elif sheet.cell_value(num_frames, 1) == 1:
                            p_not_s += 1
                        elif sheet.cell_value(num_frames, 1) == 3:
                            n_not_s += 1
                    elif label_test == 3:
                        if sheet.cell_value(num_frames, 1) == 0:
                            r_not_n += 1
                        elif sheet.cell_value(num_frames, 1) == 1:
                            p_not_n += 1
                        elif sheet.cell_value(num_frames, 1) == 2:
                            s_not_n += 1

            else:
                cv2.drawContours(resized, [contours[ci]], 0, (0, 255, 0), 2)
                cv2.drawContours(resized, [hull], 0, (0, 0, 255), 2)

        cv2.imshow("Hand", resized) # print output
        frame = resized
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rval, frame = vc.read()

    else: # when code is finished print all results fro analysis
        print("done")
        print("Total Frames: ", num_frames)
        print("Certain Frames: ", certain_frames)
        print("Correct Frames: ", correct_frames)
        percent_certain = (certain_frames/num_frames) * 100
        print("Percent Certain: ", percent_certain)
        percent_correct = (correct_frames/num_frames) * 100
        print("Percent Correct: ", percent_correct)
        print("True Positives: ", true_pos)
        print("False Positives: ", false_pos)
        print("True Negatives: ", true_neg)
        print("False Negatives: ", false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_score = 2 * ((precision * recall) / (precision + recall))
        print("F1 score: ", f_score)
        specificity = true_neg / (true_neg + false_pos)
        false_pos_rate = 1 - specificity
        true_pos_rate = recall
        print("True Positive Rate: ", true_pos_rate)
        print("False Positive Rate: ", false_pos_rate)
        print("True Rock: ", true_rock)
        print("False Rock: ", false_rock)
        print("True Paper: ", true_paper)
        print("False Paper: ", false_paper)
        print("True Scissors: ", true_scissors)
        print("False Scissors: ", false_scissors)
        print("True Nothing: ", true_nothing)
        print("False Nothing: ", false_nothing)
        print("Paper not Rock", p_not_r)
        print("Scissors not Rock", s_not_r)
        print("Nothing not Rock", n_not_r)
        print("Rock not Paper", r_not_p)
        print("Scissors not Paper", s_not_p)
        print("Nothing not Paper", n_not_p)
        print("Rock not Scissors", r_not_s)
        print("Paper not Scissors", p_not_s)
        print("Nothing not Scissors", n_not_s)
        print("Rock not Nothing", r_not_n)
        print("Paper not Nothing", p_not_n)
        print("Scissors not Nothing", s_not_n)
        break

vc.release()
out.release() # release video
