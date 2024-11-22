import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *

if __name__ == "__main__":
    img = plt.imread("./images/Bird.jpg")  # read image
    size = img.shape

    # Device Characteristic Modeling
    XYZ = CharacteristicModel(img, M1, gamma1)
    XYZ = Normalized(XYZ)

    # Get white point
    img_white = np.ones_like(img)
    XYZw = CharacteristicModel(img_white, M1, gamma1)
    XYZw = Normalized(XYZw)

    # CIECAM02
    CIECAM02(XYZ, XYZw).Forward()


    



    # plt.imshow(img)
    # plt.title("Original img")
    # plt.figure()
    # plt.imshow(XYZ)
    # plt.title("XYZ")
    # plt.figure()
    # plt.imshow(XYZw)
    # plt.title("XYZw")
    plt.show()