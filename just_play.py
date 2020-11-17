# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:36:05 2020

@author: TARUN OMAR
"""

from keras.models import load_model
import cv2
import numpy as np
from random import choice
from time import sleep
import time


REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


def computer_chose(move):
    if move == "rock":
        return "paper"

    if move == "paper":
        return "scissors"

    if move == "scissors":
        return "rock"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
image_counter = 0
frame_set = []
start_time = time.time()

prev_move = None
level = "1"
total_win1 = 0
total_win2 = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "LEVEL: " + level,
                (500, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    if time.time() - start_time >= 3:
        # predict the move made
        pred = model.predict(np.array([img]))
        move_code = np.argmax(pred[0])
        user_move_name = mapper(move_code)
        start_time = time.time()
        if level == "2":
            if prev_move != user_move_name:
                if user_move_name != "none":
                    computer_move_name = computer_chose(user_move_name)
                    winner = calculate_winner(user_move_name, computer_move_name)
                    if winner == "User":
                        total_win1 = total_win1 + 1
                    else:
                        total_win2 = total_win2 + 1
                else:
                    computer_move_name = "none"
                    winner = "Waiting..."
            prev_move = user_move_name

        if total_win1 == 3 or total_win2 == 3:
            if level == "1":
                total_win2 = 0
                total_win1 = 0
                level = "2"
            else:
                if total_win1 > total_win2:
                    result = "You are the winner"
                else:
                    result = "You lose"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, result,
                            (400, 650), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                sleep(6)
                level = "1"
                total_win1 = 0
                total_win2 = 0

        if level == "1":
            if prev_move != user_move_name:
                if user_move_name != "none":
                    computer_move_name = choice(['rock', 'paper', 'scissors'])
                    winner = calculate_winner(user_move_name, computer_move_name)
                    if winner == "User":
                        total_win1 = total_win1 + 1
                    else:
                        total_win2 = total_win2 + 1
                else:
                    computer_move_name = "none"
                    winner = "Waiting..."
            prev_move = user_move_name

        # display the information

        cv2.putText(frame, "Your Move: " + user_move_name,
                    (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move_name,
                    (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner,
                    (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        if computer_move_name != "none":
            icon = cv2.imread(
                "images/{}.png".format(computer_move_name))
            icon = cv2.resize(icon, (400, 400))
            frame[100:500, 800:1200] = icon
            sleep(3)

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
