import numpy as np


def adaptive_adjustment(h, C, J):
    # 根據統計量調整 h, C, J
    # 色相（h）
    std_hue = np.std(h)
    adjustment_factor_hue = std_hue / np.mean(h)
    #print(adjustment_factor_hue)
    h = h**adjustment_factor_hue
    
    # 色度（C）
    std_chroma = np.std(C)
    adjustment_factor_chroma = std_chroma / np.mean(C)  # 用標準差/均值來調整色度的偏差
    adjustment_factor_chroma
    C = C ** adjustment_factor_chroma  # 根據調整係數來變換 C
    
    # 亮度（J）
    range_brightness = np.max(J) - np.min(J)
    adjustment_factor_brightness = range_brightness / (1.7*np.mean(J))  # 用亮度範圍與均值的比值來調整亮度
    adjustment_factor_brightness = np.clip(adjustment_factor_brightness, 1, 1.1)
    J = J ** adjustment_factor_brightness  # 根據調整係數來變換 J
    #J = J

    return h, C, J

