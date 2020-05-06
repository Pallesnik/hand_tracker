import argparse
import datetime
import time
import tensorflow as tf
from tensorflow import keras
import h5py
from keras.preprocessing import image
import numpy as np
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

time_1 = int(round(time.time()))

vc = cv2.VideoCapture("../flesh_detection/lubin.mp4")  # video capture [1]

width = np.int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = np.int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
Nframes = np.int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image
dim2 = (100, 100)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)
(check, frame) = vc.read()
num_frames = 0
certain_frames = 0
shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

while check:
    if frame is not None:
        num_frames = num_frames + 1
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        imageYCrCb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCR_CB)
        roi = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
        thresh = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, shape)
        thresh1 = cv2.bitwise_and(resized, resized, mask=thresh)
        test_image = cv2.resize(thresh1, dim2, interpolation=cv2.INTER_AREA)
        img = np.expand_dims(test_image, axis=0)

        model = keras.models.load_model('gestures(rpsv7).h5')

        result = model.predict_classes(img / 255, batch_size=1)
        probabilities = model.predict(img / 255)
        print(probabilities)
        if probabilities[0][0] >= 0.95:
            cv2.putText(resized, "Paper", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        elif probabilities[0][1] >= 0.95:
            cv2.putText(resized, "Rock", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        elif probabilities[0][2] >= 0.95:
            cv2.putText(resized, "Scissors", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        cv2.imshow("FIRST", resized)
        frame = cv2.flip(resized, 0)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rval, frame = vc.read()

    #except:
    #    continue

    else:
        time_2 = int(round(time.time()))
        print("done")
        percent = (certain_frames / num_frames) * 100
        print(certain_frames)
        print(num_frames)
        print(percent)
        print("Time taken in seconds: ", (time_2 - time_1))
        break

vc.release()
out.release()