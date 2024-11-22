import numpy as np
from ulit import *

class CIECAM02:
    def __init__(self, XYZ, XYZw, La=63, Yb=25, c=0.69, Nc=1, F=1):
        self.XYZ = XYZ
        self.XYZw = XYZw
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
    
    def LMSConversion(self, XYZ):
        LMS = np.transpose(np.tensordot(self.MCAT02, XYZ, axes=([0], [2])), (1, 2, 0))
        return LMS
    
    def ChromaticTransform(self):
        # XYZ -> LMS 
        # Warning
        # 1. using normalized for LMS Conversion have negative number
        # 2. bias for computing Lc, Mc, Sc divide zero problem
        LMS = self.LMSConversion(self.XYZ)
        LMS = Normalized(LMS)+1
        LMSw = self.LMSConversion(self.XYZw)
        LMSw = Normalized(LMSw)+1

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
        LMSp = np.transpose(np.tensordot(np.linalg.inv(self.MCAT02), LMSc, axes=([0], [2])), (1, 2, 0))
        LMSp = np.transpose(np.tensordot(self.MH, LMSp, axes=([0], [2])), (1, 2, 0))

        # Non-Linear compression
        Lap = (400*(FL*LMSp[:, :, 0]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 0]/100))+0.1
        Map = (400*(FL*LMSp[:, :, 1]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 1]/100))+0.1
        Sap = (400*(FL*LMSp[:, :, 2]/100)**(0.42))/(27.13+(FL*LMSp[:, :, 2]/100))+0.1

        LMSap = np.array([Lap, Map, Sap])
        LMSap = np.transpose(LMSap, (1, 2, 0))

        return LMSap

    def OpponentColorConversion(self, LMSap):
        Lap = LMSap[:, :, 0]
        Map = LMSap[:, :, 1]
        Sap = LMSap[:, :, 2]

        # Conversion from the cone-response space to opponent space
        C1 = Lap-Map
        C2 = Map-Sap
        C3 = Sap-Lap

        return np.array([C1, C2, C3])

    def ComputingPreceptualAttributes(self, Cs, LMSap):
        Lap = LMSap[:, :, 0]
        Map = LMSap[:, :, 1]
        Sap = LMSap[:, :, 2]
        C1 = Cs[0]
        C2 = Cs[1]
        C3 = Cs[2]

        # Compute necessary parameters
        n = self.Yb/self.XYZw[:, :, 1]
        Nbb = 0.725*(1/n)**0.2
        z = 1.48+(n)**0.5

        A = (2*Lap+Map+1/20*Sap-0.305)*Nbb

        a = C1-1/11*C2
        b = 1/2*(C2-C1+C1-C3)/4.5

        # H
        h_rad = np.arctan2(b, a)
        h_deg = np.degrees(h_rad)
        h = np.mod(h_deg, 360)

        # Lightness

        # Brightness

        # Chroma

        # Colorfulness

        # Saturation

        


    def Forward(self):
        LMSap = self.ChromaticTransform()
        Cs = self.OpponentColorConversion(LMSap)
        self.ComputingPreceptualAttributes(Cs, LMSap)

