import numpy as np
import matplotlib.pyplot as plt
import cv2
from ulit import *

def CLAHE(A):
    A_max = np.max(A)
    A = NormalizedOneChannel(A)
    A = Normalized1To255(A)
    A = np.uint8(A)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    A = clahe.apply((A).astype(np.uint8))
    A = Normalized255To1(A)
    A = A*A_max
    return A

def SimpleEqualization(A):
    A_max = np.max(A)
    A = NormalizedOneChannel(A)
    A = Normalized1To255(A)
    # pA = CreatePDF(A, 256)
    # plt.bar(range(len(pA)), pA)
    # plt.title("Histogram before")
    # plt.figure()
    A = np.uint8(A)
    A = Equalization(A, 256)
    # pA = CreatePDF(A, 256)
    # plt.bar(range(len(pA)), pA)
    # plt.title("Histogram after")
    # plt.figure()
    A = Normalized255To1(A)
    A = A*A_max
    return A

def RGBEqualization(img):
    img = Normalized1To255(img)
    R_channel = img[:, :, 0]
    G_channel = img[:, :, 1]
    B_channel = img[:, :, 2]
    # R_channel_eq = Equalization(R_channel, 256)
    # G_channel_eq = Equalization(G_channel, 256)
    # B_channel_eq = Equalization(B_channel, 256)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    R_channel_eq = clahe.apply((R_channel).astype(np.uint8))
    G_channel_eq = clahe.apply((G_channel).astype(np.uint8))
    B_channel_eq = clahe.apply((B_channel).astype(np.uint8))
    new_img = np.array([R_channel_eq, G_channel_eq, B_channel_eq])
    new_img = np.transpose(new_img, axes=(1, 2, 0))
    new_img = Normalized255To1(new_img)
    return new_img

# **********  Useful function  **********
# Simple equalization
def Equalization(img, bar_num):
    pr = CreatePDF(img, bar_num)
    sk = np.zeros(len(pr))
    for k in range(len(sk)):
        sk[k] = (len(sk)-1)*np.sum(pr[:k])
    return Transformation(img, sk)

# Matching equalization
def Matching(img, pr, pz):
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

    return Transformation(img, zr)

def CreatePDF(img, bar_num):
    M, N = img.shape
    pr = np.zeros(bar_num)
    for m in range(M):
        for n in range(N):
            pr[img[m, n]] += 1
    pr /= M*N

    return pr

def find_closest_index(array, value):
    return np.abs(array - value).argmin()

def Transformation(img, zr):
    M, N = img.shape
    new_img = np.zeros_like(img)
    for m in range(M):
        for n in range(N):
            new_img[m, n] = zr[img[m, n]]
    new_img = np.int16(np.round(new_img))
    return new_img

def Gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def PlotHistorgram(img, name):
    img = np.uint8(NormalizedOneChannel(img)*255)
    pimg = CreatePDF(img, 256)
    plt.bar(range(len(pimg)), pimg)
    plt.title(name)
    plt.figure() 

def gamma_correction(img, gamma):
    img_max = np.max(img)
    img = np.uint8(NormalizedOneChannel(img)*255)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    corrected_image = cv2.LUT(img, table)
    corrected_image = Normalized255To1(corrected_image)*img_max

    return corrected_image

def Chroma_adjustment(original_C, C, mask):
    thr = np.abs(np.average(original_C[mask])-np.average(C[mask]))
    diff = np.abs(original_C - C)
    condition = (diff >= thr) & mask
    C = np.where(condition, 0.8*original_C+0.2*C, C)
    return C