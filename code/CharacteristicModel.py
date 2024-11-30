import numpy as np

def CharacteristicModel(img, M, gamma):
    XYZ = np.zeros_like(img, dtype=np.float64)
    XYZ = img**gamma
    XYZ = np.transpose(np.tensordot(M, XYZ, axes=([1], [2])), (1, 2, 0)).astype(np.float64)
    return XYZ

def WhitePoint(img, M, gamma):
    XYZw = np.ones_like(img, dtype=np.float64)
    XYZw = XYZw**gamma
    XYZw = np.transpose(np.tensordot(M, XYZw, axes=([1], [2])), (1, 2, 0)).astype(np.float64)
    return XYZw
