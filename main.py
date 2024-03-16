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

    detector = cv.SimpleBlobDetector()
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    keypoints = detector.detect(grayscale)
    final = cv2.drawKeypoints(grayscale, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    

    cv.imshow("pythin",final)

vid.release()
cv.destroyAllWindows()