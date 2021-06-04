from modules.MpTools import TrackHandModule
import cv2
import math
import pyautogui
import numpy as np
import time

cap = cv2.VideoCapture(0)
hand = TrackHandModule.HandTrack(max_num_hands=1)


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,  1)
    hand.find_hand(frame)
    lmlist = hand.findPosition(frame)
    if len(lmlist) != 0:
        x1, y1 = int(lmlist[8][1]), int(lmlist[8][2])
        x2, y2 = int(lmlist[4][1]), int(lmlist[4][2])
        cenx, ceny = int(lmlist[9][1]), int(lmlist[9][2])
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.line(frame, (int(lmlist[8][1]), int(lmlist[8][2])), (int(lmlist[4][1]), int(lmlist[4][2])), (255, 0, 0), 4)
        cv2.circle(frame, (int(lmlist[8][1]), int(lmlist[8][2])),  10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (int(lmlist[4][1]), int(lmlist[4][2])), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

        lenght = math.hypot(x1-x2, y1-y2)

        if lenght < 50:
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            pyautogui.press("space")
            time.sleep(1 )
        else:
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Controller", frame)
    cv2.waitKey(1)
