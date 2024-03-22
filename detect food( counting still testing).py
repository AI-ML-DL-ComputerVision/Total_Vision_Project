import cv2
import numpy as np

def derivative_video(input_file, output_file):
    # 打开视频文件
    cap = cv2.VideoCapture('testvideo.mp4')
    
    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 定义视频编码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), 0)
    
    # 前一帧
    ret, prev_frame = cap.read()
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为灰度图像
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算梯度
        gradient_x = cv2.Sobel(prev_frame_gray, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(prev_frame_gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 将梯度幅值转换为灰度图像
        gradient_magnitude = np.uint8(255 * (gradient_magnitude / np.max(gradient_magnitude)))
        
        # 写入处理后的帧
        out.write(cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR))
        
        # 显示处理后的视频
        cv2.imshow('Derivative Video', gradient_magnitude)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 更新前一帧
        prev_frame_gray = frame_gray
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_file = 'input_video.mp4'
    output_file = 'derivative_video.avi'
    derivative_video(input_file, output_file)

import cv2
import numpy as np

def capture_video(gradient_magnitude):
    # 在此处添加捕获视频的代码
    # 返回视频文件路径
    return 'input_video.mp4'

def count_white_pixels(input_fi):
    # 打开视频文件
    cap = cv2.VideoCapture(input_fi)
    
    white_pixels_counts = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将帧转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算白色像素点数量
        white_pixels_count = np.sum(gray_frame >= 230)
        white_pixels_counts.append(white_pixels_count)
    
    # 释放资源
    cap.release()
    
    return white_pixels_counts

# 示例用法
if __name__ == "__main__":
    input_file = capture_video('Derivative Video')
    white_pixels_counts = count_white_pixels(input_fi)
    print("每一帧内的白色像素点数量：", white_pixels_counts)