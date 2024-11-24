import numpy as np
from ulit import *


class InverseCIECAM02:
    
    def __init__(self,XYZ, XYZw, h, J, C, La=100, Yb=25, c=0.69, Nc=1, F=1):
        self.XYZ = XYZ
        self.XYZw = XYZw
        self.La = La
        self.Yb = Yb
        self.C = C
        self.J = J
        self.h = h
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
        self.M1 = np.array(
            [
                [2, 1, 1/20],
                [1, -12/11, 1/11],
                [1/9, 1/9, -2/9]
            ]

        )

    def LMSConversion(self, XYZ):
        LMS = np.transpose(np.tensordot(self.MCAT02, XYZ, axes=([0], [2])), (1, 2, 0))
        return LMS

    def Inverse_flow(self):
        # Calculate the necessary parameters 
        n = self.Yb/self.XYZw[:, :, 1]
        Nbb = 0.725*(1/n)**0.2
        z = 1.48+(n)**0.5
        k = 1/(5*self.La+1)
        FL = 0.2*k**4*5*self.La+0.1*(1-k**4)**2*(5*self.La)**(1/3)
        D = self.F*(1-(1/3.6)*np.exp(-(self.La+42)/92))

        # Step 1: Calculate t from C and J
        t = (self.C/ (np.sqrt(self.J/100)*((1.64-0.29**n)**0.73)))**(1/0.9)


        # Step 2: Calculate et form h
        et = np.cos(self.h*np.pi/180+2)+3.8


        # Step 3: Calculate A from Aw and J
        # get the Lw, Mw, Sw
        LMSw = self.LMSConversion(self.XYZw)
        Lw = LMSw[:, :, 0]
        Mw = LMSw[:, :, 1]
        Sw = LMSw[:, :, 2]

        Aw = (2*Lw+Mw+1/20*Sw-0.305)*Nbb
        A = Aw*(self.J/100)**(1/self.c*z)


        # Step 4: Calculate a and b from t, et, h and A
        e = 12500/13*self.Nc*Nbb*et
        # A1 means A'
        A1 = A/Nbb+0.305
        k = A1/(e/t+11/23*np.cos(self.h)+108/23*np.sin(self.h))
        a = k*np.cos(self.h)
        b = k*np.sin(self.h)

        # Step 5: Calculate La' Ma' and Sa' from A, a and b
        # Inverse of M1
        M1_inv = np.linalg.inv(self.M1)  
        # Stack A1, a, b into a 3xN matrix
        inputs = np.stack([A1, a, b], axis=0) 

        # Compute LMSa
        LMSa = np.transpose(np.tensordot(M1_inv, inputs, axes=([1], [0])), (0, 1, 2))
        LMSa = Normalized(LMSa)+0.1
        # print(np.min((100/FL*((27.13*(LMSa[0, :, :]-0.1))))))
        # print(np.min((400-(LMSa[0, :, :]-0.1))))

        # Step 6: Use the inverse nonlinearity to compute L', M' and S'
        L_1 = (100/FL*((27.13*(LMSa[0, :, :]-0.1))/(400-(LMSa[0, :, :]-0.1))))**(1/0.42)
        M_1 = (100/FL*((27.13*(LMSa[1, :, :]-0.1))/(400-(LMSa[1, :, :]-0.1))))**(1/0.42)
        S_1 = (100/FL*((27.13*(LMSa[2, :, :]-0.1))/(400-(LMSa[2, :, :]-0.1))))**(1/0.42)

        # Step 7: Convert to Lc, Mc, and Sc via linear transform
        LMS_1 = np.stack([L_1, M_1, S_1], axis=0) 
        Inv_MH = np.linalg.inv(self.MH)

        LMSc = np.transpose(np.tensordot(Inv_MH, LMS_1, axes=([1], [0])), (0, 1, 2))

        # Step 8: Invert the  chromatic adaptation transform to compute L, M and S and then X, Y, and Z
        L = LMSc[0, :, :]/(100*D/Lw+1-D)
        M = LMSc[1, :, :]/(100*D/Mw+1-D)
        S = LMSc[2, :, :]/(100*D/Sw+1-D)

        # Calculate X, Y, Z
        LMS = np.stack([L, M, S], axis=0)
        XYZe = np.transpose(np.tensordot(self.MCAT02, LMS, axes=([1], [0])), (0, 1, 2))

        return XYZe

    def Forward(self):
        XYZe = self.Inverse_flow()
        XYZe = np.transpose(XYZe, (1, 2, 0))

        return XYZe
