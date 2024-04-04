import cv2
import numpy as np

def draw_circle_on_video(video_file, center, radius):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制圆形
        cv2.circle(frame, center, radius, (0, 0, 255), 2)  # BGR格式，这里使用红色

        # 将帧写入输出视频文件
        out.write(frame)

        # 显示结果
        cv2.imshow('Circle on Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
            
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return height

if __name__ == "__main__":
   
    center = (628, 372)  # 圆心坐标 (x, y)
    radius = 472  # 圆的半径 
    # 在视频上绘制圆形
    draw_circle_on_video('testvideo.mp4', center, radius)
    