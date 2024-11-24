import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *
from Inverse_CIECAM02 import *
from PostGamutMapping import *

if __name__ == "__main__":
    img = plt.imread("./images/MorningView.jpg")  # read image
    img = img[:, :, :3]
    size = img.shape

    # Device Characteristic Modeling
    XYZ = CharacteristicModel(img, M1, gamma1)
    XYZ = Normalized(XYZ)

    # Get white point
    img_white = np.ones_like(img)
    XYZw = CharacteristicModel(img_white, M1, gamma1)
    XYZw = Normalized(XYZw)

    # CIECAM02
    perceptual_attributes = CIECAM02(XYZ, XYZw).Forward()
    h = perceptual_attributes["h"]
    J = perceptual_attributes["J"]
    C = perceptual_attributes["C"]

    # Inverse CIECAM02
    XYZe = InverseCIECAM02(XYZ, XYZw, h, J, C).Forward()

    # Post Gamut Mapping
    RGBe = PostGamutMapping(XYZe, img, gamma2, J, C)

    plt.imshow(img)
    plt.title("Original img")
    plt.figure()
    plt.imshow(XYZ)
    plt.title("XYZ")
    plt.figure()
    plt.imshow(XYZw)
    plt.title("XYZw")
    plt.figure()
    plt.imshow(XYZe)
    plt.title("XYZe")
    plt.figure()
    plt.imshow(RGBe)
    plt.title("RGBe")
    plt.show()