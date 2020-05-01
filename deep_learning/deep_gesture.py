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

vc = cv2.VideoCapture("../flesh_detection/lubin.mp4")  # video capture [1]

width = np.int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = np.int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
Nframes = np.int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = np.int(vc.get(cv2.CAP_PROP_FPS))
dim = (640, 480)  # resizing image
dim2 = (50, 50)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, FPS, dim, True)
(check, frame) = vc.read()
num_frames = 0
certain_frames = 0

while check:
    if frame is not None:
        num_frames = num_frames + 1
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        test_image = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        img = np.expand_dims(test_image, axis=0)

        model = keras.models.load_model('gestures.h5')

        result = model.predict_classes(img, batch_size=1)
        probabilities = model.predict_proba(img)
        print(probabilities)
        if result == [1]:
            cv2.putText(resized, "Rock", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        elif result == [0]:
            cv2.putText(resized, "Paper", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        elif result == [2]:
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
        print("done")
        percent = (certain_frames/num_frames) * 100
        print(percent)
        break

vc.release()
out.release()