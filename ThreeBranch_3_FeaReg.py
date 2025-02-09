import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ThreeBranch_Net(nn.Module):

    def __init__(self, Dim=[3, 34, 31], Depth=3, KS_1=3, KS_2=3, KS_3=3):
        super(ThreeBranch_Net, self).__init__()

        block1_1 = []
        block1_2 = []
        block2_1 = []
        block2_2 = []
        block3_1 = []
        block3_2 = []

        for i in range(Depth):
            block1_1 += [nn.Conv2d(128, 128, KS_1, 1, int(KS_1 / 2)), nn.ReLU()]
            block1_2 += [nn.Conv2d(128, 128, KS_1, 1, int(KS_1 / 2)), nn.ReLU()]
            block2_1 += [nn.Conv2d(128, 128, KS_2, 1, int(KS_2 / 2)), nn.ReLU()]
            block2_2 += [nn.Conv2d(128, 128, KS_2, 1, int(KS_2 / 2)), nn.ReLU()]
            block3_1 += [nn.Conv2d(128, 128, KS_3, 1, int(KS_3 / 2)), nn.ReLU()]
            block3_2 += [nn.Conv2d(128, 128, KS_3, 1, int(KS_3 / 2)), nn.ReLU()]

        self.layerIn1_1 = nn.Conv2d(Dim[0], 64, 3, 1, 1)
        self.layerIn1_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.layerIn2_1 = nn.Conv2d(Dim[1], 64, 3, 1, 1)
        self.layerIn2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.layerIn3_1 = nn.Conv2d(Dim[2], 64, 3, 1, 1)
        self.layerIn3_2 = nn.Conv2d(64, 128, 3, 1, 1)

        # Shared Imformation extraction layer, between three input
        self.Infor = nn.Sequential(
            *[
                nn.Conv2d(387, 128, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1, 1),
            ]
        )

        self.grid_smple = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, 1, 1),
        )

        # Shared Imformation exchange layer between two stage
        self.layerX_G = nn.Conv2d(387, 128, 3, 1, 1)
        self.layerX_M = nn.Conv2d(128, 128, 3, 1, 1)
        self.layerX_B = nn.Conv2d(128, 128, 3, 1, 1)

        self.layerY_G = nn.Conv2d(387, 128, 3, 1, 1)
        self.layerY_M = nn.Conv2d(128, 128, 3, 1, 1)
        self.layerY_B = nn.Conv2d(128, 128, 3, 1, 1)

        self.layerZ_G = nn.Conv2d(387, 128, 3, 1, 1)
        self.layerZ_M = nn.Conv2d(128, 128, 3, 1, 1)
        self.layerZ_B = nn.Conv2d(128, 128, 3, 1, 1)

        self.layerOut1 = nn.Conv2d(384, 256, 3, 1, 1)
        self.layerOut2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.layerOut3 = nn.Conv2d(128,  Dim[2], 3, 1, 1)

        # backbone of three branch in two stage
        self.branch1_1 = nn.Sequential(*block1_1)
        self.branch1_2 = nn.Sequential(*block1_2)
        self.branch2_1 = nn.Sequential(*block2_1)
        self.branch2_2 = nn.Sequential(*block2_2)
        self.branch3_1 = nn.Sequential(*block3_1)
        self.branch3_2 = nn.Sequential(*block3_2)

        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

        self.theta = torch.from_numpy(np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])).type(torch.cuda.FloatTensor)

    def forward(self, x, y, z):
        grid_ = F.affine_grid(
            self.theta.unsqueeze(0).repeat(x.shape[0], 1, 1),
            [x.shape[0], 128, x.shape[2], x.shape[3]], align_corners=True
        )

        a = torch.ones(x.shape[2], x.shape[3]).unsqueeze(0)
        b = torch.zeros(x.shape[2], x.shape[3]).unsqueeze(0)
        self.MaskX = torch.cat((b, a, a), 0).unsqueeze(0).cuda()
        self.MaskY = torch.cat((a, b, a), 0).unsqueeze(0).cuda()
        self.MaskZ = torch.cat((a, a, b), 0).unsqueeze(0).cuda()

        # Input processing
        outIn_1 = self.layerIn1_2(self.Relu(self.layerIn1_1(x)))
        outIn_2 = self.layerIn2_2(self.Relu(self.layerIn2_1(y)))
        outIn_3 = self.layerIn3_2(self.Relu(self.layerIn3_1(z)))

        # First stage
        out1 = self.branch1_1(outIn_1)
        out2 = self.branch2_1(outIn_2)
        out3 = self.branch3_1(outIn_3)

        grid = self.grid_smple(out2).permute(0, 2, 3, 1)
        out3 = F.grid_sample(out3, grid+grid_)

        Infor_x = self.Infor(torch.cat((out1, out2, out3, self.MaskX.repeat(out1.shape[0],1,1,1)),1))
        out1_G = self.Sig(self.layerX_G(torch.cat((out1, out2, out3, self.MaskX.repeat(out1.shape[0],1,1,1)),1)))
        out1_M = self.layerX_M(Infor_x)
        out1_B = self.layerX_B(Infor_x)
        out_1 = out1_G*out1 + (1-out1_G)*(out1*out1_M+out1_B)

        Infor_y = self.Infor(torch.cat((out1, out2, out3, self.MaskY.repeat(out1.shape[0],1,1,1)),1))
        out2_G = self.Sig(self.layerY_G(torch.cat((out1, out2, out3, self.MaskY.repeat(out1.shape[0],1,1,1)),1)))
        out2_M = self.layerY_M(Infor_y)
        out2_B = self.layerY_B(Infor_y)
        out_2 = out2_G*out2 + (1-out2_G)*(out2*out2_M+out2_B)

        Infor_Z = self.Infor(torch.cat((out1, out2, out3, self.MaskZ.repeat(out1.shape[0],1,1,1)),1))
        out3_G = self.Sig(self.layerZ_G(torch.cat((out1, out2, out3, self.MaskZ.repeat(out1.shape[0],1,1,1)),1)))
        out3_M = self.layerZ_M(Infor_Z)
        out3_B = self.layerZ_B(Infor_Z)
        out_3 = out3_G*out3 + (1-out3_G)*(out3*out3_M+out3_B)

        # Second Stage
        out1 = self.branch1_2(out_1)
        out2 = self.branch2_2(out_2)
        out3 = self.branch3_2(out_3)

        # grid = self.grid_smple(out2).permute(0, 2, 3, 1)
        # out3_ = F.grid_sample(out3, grid+grid_)

        Infor_x = self.Infor(torch.cat((out1, out2, out3, self.MaskX.repeat(out1.shape[0],1,1,1)),1))
        out1_G = self.Sig(self.layerX_G(torch.cat((out1, out2, out3, self.MaskX.repeat(out1.shape[0],1,1,1)),1)))
        out1_M = self.layerX_M(Infor_x)
        out1_B = self.layerX_B(Infor_x)
        out_1 = out1_G * out1 + (1 - out1_G) * (out1 * out1_M + out1_B) + outIn_1

        Infor_y = self.Infor(torch.cat((out1, out2, out3, self.MaskY.repeat(out1.shape[0],1,1,1)),1))
        out2_G = self.Sig(self.layerY_G(torch.cat((out1, out2, out3, self.MaskY.repeat(out1.shape[0],1,1,1)),1)))
        out2_M = self.layerY_M(Infor_y)
        out2_B = self.layerY_B(Infor_y)
        out_2 = out2_G * out2 + (1 - out2_G) * (out2 * out2_M + out2_B) + outIn_2

        Infor_Z = self.Infor(torch.cat((out1, out2, out3, self.MaskZ.repeat(out1.shape[0],1,1,1)),1))
        out3_G = self.Sig(self.layerZ_G(torch.cat((out1, out2, out3, self.MaskZ.repeat(out1.shape[0],1,1,1)),1)))
        out3_M = self.layerZ_M(Infor_Z)
        out3_B = self.layerZ_B(Infor_Z)
        out_3 = out3_G * out3 + (1 - out3_G) * (out3 * out3_M + out3_B) + outIn_3

        # grid = self.grid_smple(out2).permute(0, 2, 3, 1)
        # out_3 = F.grid_sample(out_3, grid+grid_)

        # Output processing
        out = self.Relu(self.layerOut1(torch.cat((out_1, out_2, out_3), 1)))
        out = self.Relu(self.layerOut2(out))
        out = self.layerOut3(out)

        return out


