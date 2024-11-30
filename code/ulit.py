import numpy as np

def Normalized255To1(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N, O = img.shape
    for o in range(O):
        normalized_image[:, :, o] = (img[:, :, o])/(255)

    return normalized_image

def Normalized(img):
    normalized_image = np.zeros_like(img, dtype=np.float64)
    M, N, O = img.shape
    for o in range(O):
        channel_min = np.min(img[:, :, o])
        channel_max = np.max(img[:, :, o])
        normalized_image[:, :, o] = (img[:, :, o]-channel_min)/(channel_max-channel_min)

    return normalized_image