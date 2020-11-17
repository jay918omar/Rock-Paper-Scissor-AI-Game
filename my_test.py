# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:44:19 2020

@author: TARUN OMAR
"""

from keras.models import load_model
import cv2
import numpy as np
import sys

filepath = r"C:\Users\TARUN OMAR\Desktop\30.jpg"

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rock-paper-scissors-model3.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
print(pred)
print(move_code)