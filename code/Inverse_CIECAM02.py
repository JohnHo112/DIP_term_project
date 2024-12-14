import numpy as np
from ulit import *


class InverseCIECAM02:
    
    def __init__(self, Wl, h, J, C, La=63, Yb=25, c=0.69, Nc=1, F=1):
        self.XYZw = Wl
        self.h = h
        self.J = J
        self.C = C
        self.La = La
        self.Yb = Yb
        self.c = c
        self.Nc = Nc
        self.F = F
        self.MCAT02 = np.array(
            [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834]
            ]
        )
        self.MH = np.array(
            [
                [0.38971, 0.68898, -0.07868],
                [-0.22981, 1.18340, 0.04641],
                [0.00000, 0.00000, 1.00000]
            ]
        )
        self.M = np.array(
            [
                [2, 1, 1/20],
                [1, -12/11, 1/11],
                [1/9, 1/9, -2/9]
            ]

        )

    def LMSConversion(self, XYZ):
        LMS = np.transpose(np.tensordot(self.MCAT02, XYZ, axes=([1], [2])), (1, 2, 0))
        return LMS
    
    def ChromaticTransform(self, XYZ):
        # XYZ -> LMS 
        LMS = self.LMSConversion(XYZ)
        LMSw = self.LMSConversion(self.XYZw)

        # Compute the degree of adaptation
        D = self.F*(1-(1/3.6)*np.exp(-(self.La+42)/92))
    
        # Von-Kries-Type Gain control
        Lc = ((100/LMSw[:, :, 0])*D+1-D)*LMS[:, :, 0]
        Mc = ((100/LMSw[:, :, 1])*D+1-D)*LMS[:, :, 1]
        Sc = ((100/LMSw[:, :, 2])*D+1-D)*LMS[:, :, 2]
        LMSc = np.transpose(np.array([Lc, Mc, Sc]), (1, 2, 0))

        # Compute necessary parameter
        k = 1/(5*self.La+1)
        FL = 0.2*k**4*5*self.La+0.1*(1-k**4)**2*(5*self.La)**(1/3)

        # Convert the adapted LMS value (Lc, Mc, Sc) to Hunt-Pointer-Estévez (HPE) space for response compression
        LMSp = np.transpose(np.tensordot(np.linalg.inv(self.MCAT02), LMSc, axes=([1], [2])), (1, 2, 0))
        LMSp = np.transpose(np.tensordot(self.MH, LMSp, axes=([1], [2])), (1, 2, 0))

        # Non-Linear compression
        Lap = (400*(FL*LMSp[:, :, 0]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 0]/100)**(0.42))+0.1
        Map = (400*(FL*LMSp[:, :, 1]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 1]/100)**(0.42))+0.1
        Sap = (400*(FL*LMSp[:, :, 2]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 2]/100)**(0.42))+0.1
        LMSap = np.array([Lap, Map, Sap])
        LMSap = np.transpose(LMSap, (1, 2, 0))

        return LMSap

    def Inverse_flow(self, LMSwap):
        # Calculate the necessary parameters 
        n = self.Yb/self.XYZw[:, :, 1]
        Nbb = 0.725*(1/n)**0.2
        z = 1.48+(n)**0.5
        k = 1/(5*self.La+1)
        FL = 0.2*k**4*5*self.La+0.1*(1-k**4)**2*(5*self.La)**(1/3)
        D = self.F*(1-(1/3.6)*np.exp(-(self.La+42)/92))

        # Step 1: Calculate t from C and J
        t = (self.C/(np.sqrt(self.J/100)*(1.64-0.29**n)**0.73))**(1/0.9)

        # Step 2: Calculate et form h
        et = (np.cos(self.h*np.pi/180+2)+3.8)

        # Step 3: Calculate A from Aw and J
        # get the Lw, Mw, Sw
        Lwap = LMSwap[:, :, 0]
        Mwap = LMSwap[:, :, 1]
        Swap = LMSwap[:, :, 2]
        Aw = (2*Lwap+Mwap+1/20*Swap-0.305)*Nbb
        A = Aw*(self.J/100)**(1/(self.c*z))

        # Step 4: Calculate a and b from t, et, h and A
        e = 12500/13*self.Nc*Nbb*et
        Ap = A/Nbb+0.305
        k = Ap/(e/(t+1e-10)+11/23*np.cos(np.deg2rad(self.h))+108/23*np.sin(np.deg2rad(self.h))+1e-10)
        a = k*np.cos(np.deg2rad(self.h))
        b = k*np.sin(np.deg2rad(self.h))

        # Step 5: Calculate La' Ma' and Sa' from A, a and b
        Apab = np.transpose(np.array([Ap, a, b]), (1, 2, 0))
        M_inv = np.linalg.inv(self.M)
        LMSap = np.transpose(np.tensordot(M_inv, Apab, axes=([1], [2])), (1, 2, 0))

        # Step 6: Use the inverse nonlinearity to compute L', M' and S'
        Lp = np.sign(LMSap[:, :, 0]-0.1)*100/FL*np.abs((27.13/((400/(LMSap[:, :, 0]-0.1+1e-10)-1+1e-10))))**(1/0.42)
        Mp = np.sign(LMSap[:, :, 1]-0.1)*100/FL*np.abs((27.13/((400/(LMSap[:, :, 1]-0.1+1e-10)-1+1e-10))))**(1/0.42)
        Sp = np.sign(LMSap[:, :, 2]-0.1)*100/FL*np.abs((27.13/((400/(LMSap[:, :, 2]-0.1+1e-10)-1+1e-10))))**(1/0.42)
        LMSp = np.transpose(np.array([Lp, Mp, Sp]), (1, 2, 0))

        # Step 7: Convert to Lc, Mc, and Sc via linear transform
        MH_inv = np.linalg.inv(self.MH)
        LMSc = np.transpose(np.tensordot(MH_inv, LMSp, axes=([1], [2])), (1, 2, 0))
        LMSc = np.transpose(np.tensordot(self.MCAT02, LMSc, axes=([1], [2])), (1, 2, 0))

        # Step 8: Invert the  chromatic adaptation transform to compute L, M and S and then X, Y, and Z
        LMSw = self.LMSConversion(self.XYZw)
        L = LMSc[:, :, 0]/(100/LMSw[:, :, 0]*D+1-D)
        M = LMSc[:, :, 1]/(100/LMSw[:, :, 1]*D+1-D)
        S = LMSc[:, :, 2]/(100/LMSw[:, :, 2]*D+1-D)
        LMS = np.transpose(np.array([L, M, S]), (1, 2, 0))

        # Calculate X, Y, Z
        MCAT02_inv = np.linalg.inv(self.MCAT02)
        XYZ = np.transpose(np.tensordot(MCAT02_inv, LMS, axes=([1], [2])), (1, 2, 0))

        return XYZ

    def Forward(self):
        LMSwap = self.ChromaticTransform(self.XYZw)
        XYZ = self.Inverse_flow(LMSwap)

        return XYZ
