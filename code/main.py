import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *
from Inverse_CIECAM02 import *
from PostGamutMapping import *

def GenerateRGBImage(size):
    M, N, O = size
    img = np.zeros(size, dtype=np.uint8)
    img[:200, :, :] = np.array([255, 0, 0])
    img[200:400, :, :] = np.array([0, 255, 0])
    img[400:, :, :] = np.array([0, 0, 255])
    return img

if __name__ == "__main__":
    RGBi = plt.imread("./images/Bird.jpg")  # read image
    RGBi = RGBi[:, :, :3]
    RGBi = Normalized255To1(RGBi)
    size = RGBi.shape

    # Device Characteristic Modeling
    XYZi = CharacteristicModel(RGBi, M1, gamma1)

    # Get white point
    Wf = WhitePoint(RGBi, M1, gamma1)

    # CIECAM02
    perceptual_attributes = CIECAM02(XYZi, Wf).Forward()
    h = perceptual_attributes["h"]
    J = perceptual_attributes["J"]
    C = perceptual_attributes["C"]

    # Get white point
    Wl = WhitePoint(RGBi, M2, gamma2)

    # Inverse CIECAM02
    XYZe = InverseCIECAM02(Wl, h, J, C).Forward()

    # Post Gamut Mapping
    RGBe = PostGamutMapping(XYZe, RGBi, gamma2, J, C)

    plt.figure()
    plt.imshow(RGBi)
    plt.title("Original img")
    plt.figure()
    plt.imshow(RGBe)
    plt.title("RGBe")
    plt.show()