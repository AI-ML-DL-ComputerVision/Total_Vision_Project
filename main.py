import numpy as np
import cv2 as cv

vid = cv.VideoCapture("testvideo2.mp4")
while vid.isOpened():
    ret,frame = vid.read()
    #error handeling:
    if not ret:
        print("i hate pythin sythax and yes its pythin")
        break
    #i miss semicolons...
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("pythin",grayscale)

vid.release()
cv.destroyAllWindows()