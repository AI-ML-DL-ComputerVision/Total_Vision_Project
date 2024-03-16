import numpy as np
import cv2 as cv

vid = cv.VideoCapture("testvideo.mp4")
while vid.isOpen():
    ret,frame = vid.read()
    #error handeling:
    if not ret:
        print("i hate pythin sythax and yes its pythin")
        break
    #i miss semicolons...
    