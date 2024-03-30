import cv2
import numpy as np

def derivative_video(input_file, output_file):
    # 打开视频文件
    cap = cv2.VideoCapture(input_file)
    
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
    
    white_pixels_counts = []
    status = []
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
        
        gray_image = cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR)
        
        white_pixels_count = np.sum(gray_image >= 230)  # 调整阈值根据具体情况
        
        white_pixels_counts.append(white_pixels_count)
       
        # 写入处理后的帧
        out.write(cv2.cvtColor(gradient_magnitude, cv2.COLOR_GRAY2BGR))

        cv2.imshow(output_file, gradient_magnitude)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 更新前一帧
        prev_frame_gray = frame_gray
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return white_pixels_counts
    
def count_above_threshold(numbers, threshold):
    count = 0
    for num in numbers:
        if num > threshold:
            count += 1
    return count

def count_frames(video_file):
    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return -1
    
    frame_count = 0
    all_frames = []
    NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    # 释放资源
    cap.release()
    
    return NUM

if __name__ == "__main__":
    input_file = 'testvideo.mp4'  
    output_file = 'output_video.avi'
    counts = derivative_video(input_file, output_file)
    threshold = 1000
    count = count_above_threshold(counts, threshold)
    total=count_frames(input_file)
    
    print("每一帧内的白色像素点数量：", counts)
    print("总帧数", total)
    print("列表中大于", threshold, "的值的个数为：", count)