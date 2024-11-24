import numpy as np

def Normalized(img):
    min_val = img.min()
    max_val = img.max()
    normalized_image = (img - min_val) / (max_val - min_val)
    normalized_image = normalized_image.astype(np.float64)

    return normalized_image