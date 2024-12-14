import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
# from ContrastAdjustment import *

def Normalized255To1(img):
    normalized_image = img/255
    return normalized_image

def Normalized1To255(img):
    normalized_image = np.uint8(img*255)
    return normalized_image

def Normalized(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N, O = img.shape
    for o in range(O):
        channel_min = np.min(img[:, :, o])
        channel_max = np.max(img[:, :, o])
        normalized_image[:, :, o] = (img[:, :, o]-channel_min)/(channel_max-channel_min)

    return normalized_image

def NormalizedOneChannel(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N= img.shape
    channel_min = np.min(img)
    channel_max = np.max(img)
    normalized_image = (img-channel_min)/(channel_max-channel_min)

    return normalized_image

def CreateHueMask(h, low_degree, high_degree):
    return ((h>low_degree)&(h<high_degree))

def GammaCorrection(img, gamma):
    img_max = np.max(img)
    new_img = NormalizedOneChannel(img)*255
    new_img = 255*(new_img/255)**gamma
    new_img = Normalized255To1(new_img)*img_max
    return new_img

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


def PSNR(original, enhanced):
    # 確保影像格式為浮點型，避免溢出
    #original = original.astype(np.float64)
    #enhanced = enhanced.astype(np.float64)
    
    # 計算 MSE（均方誤差）
    mse = np.mean((original - enhanced) ** 2)
    
    
    # 計算 PSNR
    max_pixel = 255.0  # 假設影像像素範圍為 0-255
    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    print(f"PSNR Value: {psnr}")


def SSIM(reference_image, enhanced_image):
    #reference_image = reference_image.astype(np.uint8)
    #enhanced_image = enhanced_image.astype(np.uint8)
    ssim_value, ssim_map = ssim(reference_image, enhanced_image, multichannel=True, full=True)
    print(f"SSIM Value: {ssim_value}")

    # 顯示 SSIM 差異圖
    #plt.figure(figsize=(8, 6))
    #plt.imshow(ssim_map, cmap='viridis')  # 選擇 'viridis' 或其他喜歡的 colormap
    #plt.colorbar()  # 添加顏色條
    #plt.title(f"SSIM Difference Map\nSSIM Value: {ssim_value:.4f}")
    #plt.axis('off')  # 關閉座標軸
    #plt.show()
