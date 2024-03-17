import cv2
import numpy as np
from matplotlib import pyplot as plt

# Variables to keep track of everything
numlist=[]
framenum=[]
framecount=0

# Function to detect corners in the frame
def detect_corners(frame):

    global numlist
    global framecount
    global framenum

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sharpen Image so its easier to find corners
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    im=cv2.filter2D(gray,-1,kernel)

    # Find corners using the Shi-Tomasi corner detection method
    corners = cv2.goodFeaturesToTrack(im, maxCorners=1000, qualityLevel=0.15, minDistance=7)
    corners = np.intp(corners)
    
    # Draw detected corners on the frame
    for corner in corners:
        x, y = np.ravel(corner)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    # Add to list of numvers
    num=str(int((corners.size/2)))
    numlist.append(num)
    framecount+=1
    framenum.append(framecount)

    # Display text on frame
    frame=cv2.putText(frame,num,(75,75),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0))
    frame=cv2.resize(frame,(0,0),fx=0.75,fy=0.75)
    
    return frame

# Open the video file
video = cv2.VideoCapture('testvideo3.mp4')

# Read the video
ret, frame = video.read()
if not video.isOpened():
    print("Error opening video file")
    exit()

# Loop
while True:
    if not ret:
        print("Error reading video frame")
        break
    
    # Detect corners in the frame
    frame_with_corners = detect_corners(frame)
    cv2.imshow("Corners Detection", frame_with_corners)
    
    # Exit the loop if 'q' is pressed
    key=cv2.waitKey(25)
    if key== ord('q'):
        break
    ret, frame = video.read()

# Release the video file and close the windows
video.release()
cv2.destroyAllWindows()

# Plot frame vs number of food
plt.plot(framenum,numlist)
plt.xlabel("Frame")
plt.ylabel("Number of Food")
plt.title("Food over period of video")
plt.show()