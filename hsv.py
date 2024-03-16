"""
This script reads an image and prints the HSV values of each pixel.
"""
from PIL import Image

image_path = 'rgb.png'
img = Image.open(image_path)

img_hsv = img.convert('HSV')

for x in range(img_hsv.width):
    for y in range(img_hsv.height):
        h, s, v = img_hsv.getpixel((x, y))
        print(f"Pixel at ({x}, {y}): (H: {h}, S: {s}, V: {v})")
