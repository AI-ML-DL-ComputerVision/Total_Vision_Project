# Kevin's
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

vid = cv.VideoCapture("testvideo2.mp4")


while vid.isOpened():
    ret,frame = vid.read()
    #error handeling:
    if not ret:
        print("i hate pythin sythax and yes its pythin")
        break
    #i miss semicolons...

    detector = cv.SimpleBlobDetector()
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(grayscale,25,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(grayscale,(x,y),3,255,-1)

    plt.imshow(grayscale)    

vid.release()
cv.destroyAllWindows()