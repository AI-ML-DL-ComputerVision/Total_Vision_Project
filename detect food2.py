import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_circle_roi(input_video, output_video, center, radius):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Unable to open input video file.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        masked_frame[mask == 0] = [139, 126, 136]

        out.write(masked_frame)

    cap.release()
    out.release()

    return output_video

def corner_detection(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    max_corners = 500
    quality_level = 0.05  
    min_distance = 10 
    color = (0, 255, 0)  
    corner_counts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_count = 0

        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        if corners is not None:
            corners = np.int0(corners)
            corner_count += len(corners)

            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, color, -1)

        text = str(corner_count)
        position = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color_text = (255, 0, 0)  
        thickness = 2
        cv2.putText(frame, text, position, font, font_scale, color_text, thickness)
        
        cv2.imshow('Corner Detection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        corner_counts.append(corner_count)

    cap.release()
    cv2.destroyAllWindows()
    return corner_counts

def plot_line_graph(data):
    x = range(1, len(data) + 1)

    plt.plot(x, data, marker='o', linestyle='-')

    plt.title('Line Graph')
    plt.xlabel('Frame Number')
    plt.ylabel('Corner Count')

    plt.grid(True)  # 显示网格线
    plt.show()

if __name__ == "__main__":
    video_file = 'testvideo.mp4'  
    center = (628, 372)  
    radius = 440
    output_file = 'picked_file.avi'
    roi_video = extract_circle_roi(video_file, output_file, center, radius)
    if roi_video:
        print("done")
        number = corner_detection(roi_video)
        print(number)
        plot_line_graph(number)
    else:
        print("Error occurred during video processing.")