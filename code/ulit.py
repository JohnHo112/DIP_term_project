import numpy as np
import matplotlib.pyplot as plt

# Normalizeed RGB channel 
def Normalized255To1(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N, O = img.shape
    for o in range(O):
        normalized_image[:, :, o] = (img[:, :, o])/(255)

    return normalized_image

def Normalized1To255(img):
    img = np.uint8(img*255)
    return img

def Normalized(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N, O = img.shape
    for o in range(O):
        channel_min = np.min(img[:, :, o])
        channel_max = np.max(img[:, :, o])
        normalized_image[:, :, o] = (img[:, :, o]-channel_min)/(channel_max-channel_min)

    return normalized_image

def NormalizedALLChannel(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    min = np.min(img)
    max = np.max(img)
    normalized_image = (img-min)/(max-min)

    return normalized_image

def NormalizedOneChannel(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N= img.shape
    channel_min = np.min(img)
    channel_max = np.max(img)
    normalized_image = (img-channel_min)/(channel_max-channel_min)

    return normalized_image

def EstimateWhitePointTransform(W_RGB):
    # 標準 D65 白點 (W_XYZ)
    W_XYZ = np.array([0.9504, 1.0000, 1.0888])

    # sRGB 到 XYZ 的標準轉換矩陣（D65 下）
    M_sRGB = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])

    # 步驟 1: 線性化 W_RGB（去 gamma 校正）
    def linearize(rgb):
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    W_RGB_linear = linearize(W_RGB)

    # 步驟 2: 使用標準轉換矩陣計算影像白點在 XYZ 空間的值
    W_XYZ_image = np.dot(M_sRGB, W_RGB_linear)

    # 步驟 3: 計算比例因子，生成新的轉換矩陣
    scaling_factors = W_XYZ / W_XYZ_image  # 按通道比例匹配目標白點
    M_adjusted = M_sRGB * scaling_factors[:, None]  # 調整矩陣

    # 輸出調整後的轉換矩陣
    return M_adjusted

def GetW_RGB(img):
     # 計算每個像素的 R+G+B 值
    rgb_sum = np.sum(img, axis=2)
    
    # 找到最大值的位置
    max_index = np.unravel_index(np.argmax(rgb_sum), rgb_sum.shape)
    
    # 提取對應像素的 [B, G, R] 值（OpenCV 預設讀取為 BGR 格式）
    b, g, r = img[max_index]
    return np.array([r, g, b])

def CreatePDF(img, bar_num):
    M, N = img.shape
    pr = np.zeros(bar_num)
    for m in range(M):
        for n in range(N):
            pr[img[m, n]] += 1
    pr /= M*N

    return pr

def Equalization(img, bar_num):
    pr = CreatePDF(img, bar_num)
    sk = np.zeros(len(pr))
    for k in range(len(sk)):
        sk[k] = (len(sk)-1)*np.sum(pr[:k])

    return Transformation(img, sk)

def find_closest_index(array, value):
    return np.abs(array - value).argmin()

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