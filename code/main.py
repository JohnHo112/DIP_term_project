import numpy as np
import matplotlib.pyplot as plt
from  CharacteristicModel import *
from ulit import *
from table1 import *
from CIECAM02 import *
from Inverse_CIECAM02 import *
from PostGamutMapping import *
from BlueCompensation import *
from WhiteBalance import *
from AdaptiveAdjustment import *

if __name__ == "__main__":
    RGBi = plt.imread("./images/LowContrast1.jpg")  # read image
    RGBi = RGBi[:, :, :3]
    RGBi = Normalized255To1(RGBi)
    size = RGBi.shape
    
    # Device Characteristic Modeling
    XYZi = CharacteristicModel(RGBi, sRGBM, sRGBgamma)

    # Get white point
    Wf = WhitePoint(XYZi, sRGBM, sRGBgamma)
    Wf[:, :, 0] = 1
    Wf[:, :, 1] = 1 
    Wf[:, :, 2] = 1
    Wf = CharacteristicModel(Wf, sRGBM, sRGBgamma)

    # CIECAM02
    perceptual_attributes = CIECAM02(XYZi, Wf).Forward()
    h = perceptual_attributes["h"]
    h = np.uint16(h)
    J = perceptual_attributes["J"]
    C = perceptual_attributes["C"]
    original_max_C = np.max(C)
    C = NormalizedOneChannel(C)
    C = Normalized1To255(C)
    pC = CreatePDF(C, 256)
    plt.bar(range(len(pC)), pC)
    plt.figure()
    C = np.uint8(C)
    C = Equalization(C, 256)
    pC = CreatePDF(C, 256)
    plt.bar(range(len(pC)), pC)
    plt.figure()
    C = C/255
    C = C*original_max_C

    # Get white point
    Wl = WhitePoint(XYZi, sRGBM, sRGBgamma)
    Wl[:, :, 0] = 1
    Wl[:, :, 1] = 1 
    Wl[:, :, 2] = 1
    Wl = CharacteristicModel(Wl, sRGBM, sRGBgamma)

    # Inverse CIECAM02
    XYZe = InverseCIECAM02(Wl, h, J, C).Forward()

    # Post Gamut Mapping
    RGBe = PostGamutMapping(XYZe, RGBi, sRGBM, sRGBgamma, J, C)

    plt.figure()
    plt.imshow(RGBi)
    plt.title("original image")
    plt.figure()
    plt.imshow(RGBe)
    plt.title("RGBe")
    plt.show()