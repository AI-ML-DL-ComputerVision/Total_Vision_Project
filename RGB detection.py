from PIL import Image

def get_rgb_value(image_path, x, y):
    """
    Get the RGB value of a specific point in the image.
    """
    # 打开图片
    image = Image.open(image_path)
    # 获取图片指定位置的像素值
    pixel = image.getpixel((x, y))
    return pixel

def main():
    # 图片路径
    image_path = "rgb提取.png"
    # 要检测的像素点的坐标
    x = 1
    y = 1

    # 获取 RGB 值
    rgb_value = get_rgb_value(image_path, x, y)
    print(rgb_value)

if __name__ == "__main__":
    main()
