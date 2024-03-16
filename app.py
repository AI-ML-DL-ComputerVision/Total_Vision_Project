# Nathan's initial code to detect purple strips in a video
import cv2
import numpy as np
from scipy.spatial.distance import cdist

def process_frame(frame):
    """
    Process a single frame to detect corners representing purple strips.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([130, 30, 30])
    upper_purple = np.array([165, 80, 165])
    
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_parts = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(purple_parts, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.03)
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]
    corners = np.argwhere(dst > 0.01 * dst.max())
    centers = corners[:, [1, 0]]
    
    return frame, centers

cap = cv2.VideoCapture('testvideo.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame, centers = process_frame(frame)
    cv2.imshow('Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()