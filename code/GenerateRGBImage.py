import numpy as np
import matplotlib.pyplot as plt
import cv2

def GenerateRGBImage(size):
    M, N, O = size
    img = np.zeros(size, dtype=np.uint8)
    img[:200, :, :] = np.array([255, 0, 0])
    img[200:400, :, :] = np.array([0, 255, 0])
    img[400:, :, :] = np.array([0, 0, 255])
    return img

if __name__ == "__main__":
    size = (600, 600, 3)
    img = GenerateRGBImage(size)
    cv2.imwrite("./images/Color.jpg", img)
    # plt.imshow(img)
    # plt.show()
    

