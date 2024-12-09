import numpy as np

def Blue_Channel_compensation(img):
    Ir = img[:,:,0]
    Ig = img[:,:,1]
    Ib = img[:,:,2]

    IR_bar = np.average(img[:,:,0])
    IG_bar = np.average(img[:,:,1])
    IB_bar = np.average(img[:,:,2])
    
    Ibc = Ib + (IG_bar-IB_bar)/(IR_bar+IG_bar+IB_bar) * Ig

    img_compensation = np.transpose(np.array([Ir,Ig,Ibc]), (1,2,0))

    return img_compensation