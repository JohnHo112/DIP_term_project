import numpy as np
from table1 import *
from ulit import *

def PostGamutMapping(XYZe, img, M, gamma, J, C):
    # Step 1: XYZe -> RGBe
    # Inverse M2
    M2_inv = np.linalg.inv(M)

    # Calculate RGBe
    RGBel = np.transpose(np.tensordot(M2_inv, XYZe, axes=([1], [2])), (1, 2, 0))
    # Keep the sign
    RGPp = np.sign(RGBel)*np.abs(RGBel)**(1/gamma)

    # Step 2: RGB with a hard threshold
    RGBc = np.clip(RGPp, 0, 1)

    # Normalized J and C to 0-1
    J = NormalizedOneChannel(J)
    C = NormalizedOneChannel(C)
    
    # Step 3: Blend the clipped pixel value with the original pixel value
    a = (1-J*C)*RGBc[:, :, 0]+J*C*img[:, :, 0]
    b = (1-J*C)*RGBc[:, :, 1]+J*C*img[:, :, 1]
    c = (1-J*C)*RGBc[:, :, 2]+J*C*img[:, :, 2]

    RGBe = np.transpose(np.array([a, b, c]), (1, 2, 0)) 
    return RGBc