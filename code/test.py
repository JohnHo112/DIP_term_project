import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *

if __name__ == "__main__":
    img = plt.imread("./images/Bird.jpg")  # read image
    img = Normalized64(img)
    size = img.shape

    # Device Characteristic Modeling
    XYZ = CharacteristicModel(img, M1, gamma1)
    XYZ = Normalized64(XYZ)

    # # Get white point
    # img_white = np.ones_like(img)
    # XYZw = CharacteristicModel(img_white, M1, gamma1)
    # XYZw = Normalized(XYZw)

    # # CIECAM02
    # perceptual_attributes = CIECAM02(XYZ, XYZw).Forward()

    plt.imshow(img)
    plt.title("Original img")
    plt.figure()
    plt.imshow(XYZ)
    plt.title("XYZ")
    # plt.figure()
    # plt.imshow(XYZ1)
    # plt.title("XYZ1")
    plt.show()