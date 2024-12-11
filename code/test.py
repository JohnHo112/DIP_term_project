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
import colour
from tqdm import tqdm


def CreatePDF(img, bars_num):
    M, N = img.shape
    pr = np.zeros(bars_num)
    for m in range(M):
        for n in range(N):
            pr[img[m, n]] += 1
    pr /= M*N

    return pr

def find_closest_index(array, value):
    return np.abs(array - value).argmin()

def Matching(pr, pz):
    L = len(pr)
    sk = np.zeros(L)
    Gz = np.zeros(L)
    zr = np.zeros(L) 
    for k in range(L):
        sk[k] = (L-1)*np.sum(pr[:k+1])
    sk = np.int16(np.round(sk))

    for k in range(L):
        Gz[k] = (L-1)*np.sum(pz[:k+1])
    Gz = np.int16(np.round(Gz))

    for r in range(len(zr)):
        closest_index = find_closest_index(Gz, sk[r])
        zr[r] = closest_index

    return zr

def Transformation(img, zr):
    M, N = img.shape
    new_img = np.zeros_like(img)
    for m in range(M):
        for n in range(N):
            new_img[m, n] = zr[img[m, n]]
    new_img = np.int16(np.round(new_img))
    return new_img

if __name__ == "__main__":
    RGBi = plt.imread("./images/9.jpg")  # read image
    RGBi = RGBi[:, :, :3]
    RGBi = Normalized255To1(RGBi)
    RGBi = cv2.resize(RGBi, (400, 200))

    M, N, O = RGBi.shape

    H = np.zeros((M, N))
    J = np.zeros((M, N))
    C = np.zeros((M, N))

    # Device Characteristic Modeling
    XYZi = CharacteristicModel(RGBi, sRGBM, sRGBgamma)

    XYZ_w1 = [0.654, 0.698, 0.659]
    XYZ_w2 = [0.765, 0.816, 0.882]
    L_A = 63
    Y_b = 25
    surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]

    for m in tqdm(range(M)):
        for n in range(N):
            specification = colour.XYZ_to_CIECAM02(XYZi[m, n, :], XYZ_w1, L_A, Y_b, surround)
            J[m, n] = specification.J
            C[m, n] = specification.M
            H[m, n] = specification.h
    jch = np.array([J,C,H])
    jch = np.transpose(jch, axes=(1, 2, 0))

    XYZe = np.zeros_like(RGBi)
    print("CIECAM02")

    for m in tqdm(range(M)):
        for n in range(N):
            JCH = colour.CAM_Specification_CIECAM02(J=jch[m, n, 0], C=jch[m, n, 1], h=jch[m, n, 2])
            XYZe[m, n, :] = colour.CIECAM02_to_XYZ(JCH, XYZ_w2, L_A, Y_b, surround)

    # Post Gamut Mapping
    RGBe = PostGamutMapping(XYZe, RGBi, sRGBM, sRGBgamma, J, C)

    img = plt.imread("./images/10.jpg")  # read image
    img = img[:, :, :3]
    img = Normalized255To1(img)
    img = cv2.resize(img, (400, 200))

    plt.imshow(RGBi)
    plt.title("RGBi")
    plt.figure()
    plt.imshow(img)
    plt.title("img")
    plt.figure()
    plt.imshow(RGBe)
    plt.title("RGBe")
    plt.show()