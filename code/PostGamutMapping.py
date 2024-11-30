import numpy as np
from table1 import *
from ulit import *

def PostGamutMapping(XYZe, img, gamma, J, C):
    # Step 1: XYZe -> RGBe
    # Inverse M2
    M2_inv = np.linalg.inv(M2)

    # Calculate RGBe
    RGBel = np.transpose(np.tensordot(M2_inv, XYZe, axes=([1], [2])), (1, 2, 0))
    # print(np.max(RGBel < 0))
    RGPp = RGBel**(1/gamma)
    # print(RGPp)

    # Step 2: RGB with a hard threshold
    RGBc = np.clip(RGPp, 0, 1)

    # Step 3: Blend the clipped pixel value with the original pixel value
    a = (1-J*C)*RGBc[:, :, 0]+J*C*img[:, :, 0]
    b = (1-J*C)*RGBc[:, :, 1]+J*C*img[:, :, 1]
    c = (1-J*C)*RGBc[:, :, 2]+J*C*img[:, :, 2]

    RGBe = np.transpose(np.array([a, b, c]), (1, 2, 0))
    # RGBe = Normalized(RGBe)
    
    return RGBe