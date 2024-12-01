import numpy as np
import matplotlib.pyplot as plt
from ulit import *
from table1 import *

# RGBi = plt.imread("./images/MorningView.jpg")  # read image
# RGBi = RGBi[:, :, :3]
# RGBi = Normalized255To1(RGBi)

# XYZ1 = np.zeros_like(RGBi, dtype=np.float64)
# XYZ1 = np.transpose(np.tensordot(M1, RGBi, axes=([1], [2])), (1, 2, 0))
# print(XYZ1.shape)
# print(XYZ1)

# RGBi = np.transpose(RGBi, (2, 0, 1))
# XYZ2 = np.zeros_like(RGBi, dtype=np.float64)
# XYZ2[0,:,:] = M1[0, 0]*RGBi[0,:,:]+M1[0, 1]*RGBi[1,:,:]+M1[0, 2]*RGBi[2,:,:]
# XYZ2[1,:,:] = M1[1, 0]*RGBi[0,:,:]+M1[1, 1]*RGBi[1,:,:]+M1[1, 2]*RGBi[2,:,:]
# XYZ2[2,:,:] = M1[2, 0]*RGBi[0,:,:]+M1[2, 1]*RGBi[1,:,:]+M1[2, 2]*RGBi[2,:,:]
# XYZ2 = np.transpose(XYZ2, (1, 2, 0))
# print(XYZ2.shape)
# print(XYZ2)

# # print(np.min(np.round(XYZ1)==np.round(XYZ2)))
# print(np.max(XYZ1-XYZ2))

print(np.deg2rad(180))
