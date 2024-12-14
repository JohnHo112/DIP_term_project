import numpy as np
import matplotlib.pyplot as plt
from ulit import *
from table1 import *
from  CharacteristicModel import *
from CIECAM02 import *
from Inverse_CIECAM02 import *
from PostGamutMapping import *
from ContrastAdjustment import *

from BlueCompensation import *
from WhiteBalance import *
from AdaptiveAdjustment import *

if __name__ == "__main__":
    RGBi = plt.imread("./images/test2.jpg")  # read image
    RGBi = RGBi[:, :, :3]
    RGBi = Normalized255To1(RGBi)
    size = RGBi.shape

    RGB = RGBEqualization(RGBi)
    
    # RGB to XYZ
    XYZi = CharacteristicModel(RGBi, sRGBM, sRGBgamma)
    # Get white point
    Wf = WhitePoint(XYZi, sRGBM, sRGBgamma)

    # CIECAM02
    perceptual_attributes = CIECAM02(XYZi, Wf).Forward()
    h = perceptual_attributes["h"]
    J = perceptual_attributes["J"]+1  # +1 for avoid numerical problem
    C = perceptual_attributes["C"]

    blue_mask = CreateHueMask(h, 200, 280)
    all_mask = np.bool_(np.ones((size[0], size[1])))

    C1 = SimpleEqualization(C)
    C2 = CLAHE(C)
    C3 = Chroma_adjustment(C, C2, all_mask)
    C4 = Chroma_adjustment(C, C2, blue_mask)

    plt.imshow(C)
    plt.title("original C")

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(C1)
    plt.title("global equalization C")
    plt.subplot(2, 2, 2)
    plt.imshow(C2)
    plt.title("CLAHE equalization C")
    plt.subplot(2, 2, 3)
    plt.imshow(C3)
    plt.title("CLAHE equalization with adjust C")
    plt.subplot(2, 2, 4)
    plt.imshow(C4)
    plt.title("CLAHE equalization with blue mask adjust C")
    plt.figure()

    # Get white point
    Wl = WhitePoint(XYZi, sRGBM, sRGBgamma)

    # ********** not change **********
    XYZe = InverseCIECAM02(Wl, h, J, C).Forward()
    RGBe = PostGamutMapping(XYZe, RGBi, sRGBM, sRGBgamma, J, C)

    # ********** global equalization **********
    XYZe1 = InverseCIECAM02(Wl, h, J, C1).Forward()
    RGBe1 = PostGamutMapping(XYZe1, RGBi, sRGBM, sRGBgamma, J, C1)

    # ********** CLAHE equalization **********
    XYZe2 = InverseCIECAM02(Wl, h, J, C2).Forward()
    RGBe2 = PostGamutMapping(XYZe2, RGBi, sRGBM, sRGBgamma, J, C2) 

    # ********** CLAHE equalization with adjust **********
    XYZe3 = InverseCIECAM02(Wl, h, J, C3).Forward()
    RGBe3 = PostGamutMapping(XYZe3, RGBi, sRGBM, sRGBgamma, J, C3) 

    # ********** CLAHE equalization with blue mask adj **********
    XYZe4 = InverseCIECAM02(Wl, h, J, C4).Forward()
    RGBe4 = PostGamutMapping(XYZe4, RGBi, sRGBM, sRGBgamma, J, C4) 

    plt.imshow(RGBi)
    plt.title("original image")
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(RGBe1)
    plt.title("global equalization")
    plt.subplot(2, 2, 2)
    plt.imshow(RGBe2)
    plt.title("CLAHE")
    plt.subplot(2, 2, 3)
    plt.imshow(RGBe3)
    plt.title("CLAHE adj")
    plt.subplot(2, 2, 4)
    plt.imshow(RGBe4)
    plt.title("CLAHE blue adj")
    plt.show()