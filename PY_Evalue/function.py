import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ReshapeTo2D(nn.Module):

    def __init__(self):
        super(ReshapeTo2D, self).__init__()

    def forward(self,x):
        return torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

class ReshapeTo3D(nn.Module):
    def __init__(self):
        super(ReshapeTo3D, self).__init__()

    def forward(self,x):
        return  torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))

class TransDimen(nn.Module):
    def __init__(self):
        super(TransDimen, self).__init__()

    def forward(self,x):
        #print(x.shape)
        return torch.Tensor.permute(x,[0,2,1])

def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr

def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    #print(sam.max(),sam.min())
    #plt.figure(1);plt.imshow(sam, cmap='gray')
    #plt.show()
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum


class L_Dspec(nn.Module):
    def __init__(self,in_channel,out_channel,P_init):
        super(L_Dspec, self).__init__()
        self.in_channle = in_channel
        self.out_channel = out_channel
        self.P = nn.Parameter(P_init)

    def forward(self,input):
        S = input.shape
        out = torch.reshape(input,[S[1],S[2]*S[3]])
        out = torch.matmul(self.P,out)

        return torch.reshape(out,[self.out_channel,S[2],S[3]])

class Apply(nn.Module):
    def __init__(self, what, dim, *args):
        super(Apply, self).__init__()
        self.dim = dim
        self.what = what

    def forward(self, input):
        inputs = []
        for i in range(input.size(self.dim)):
            inputs.append(self.what(input.narrow(self.dim, i, 1)))
        return torch.cat(inputs, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class FineNet_SelfAtt(nn.Module):

    def __init__(self):
        super(FineNet_SelfAtt, self).__init__()
        self.Conv1 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv2 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv3 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv4 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Conv5 = nn.Conv2d(31, 31, 3, 1, 1)
        self.Relu = nn.ReLU()
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(self.Relu(out))
        out = self.Conv3(self.Relu(out))

        Z = self.Conv5(self.Relu(out))
        M = self.Sig(self.Conv4(self.Relu(out)))

        out = M*out + (1-M)*Z

        return out + x


class FineNet_ResiAtt(nn.Module):
    
    def __init__(self):
        super(FineNet_ResiAtt, self).__init__()

        branch_1 = [
            nn.Conv2d(31, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 31, 3, 1, 1)
        ]

        branch_2 = [
            nn.Conv2d(31, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 31, 3, 1, 1),
            nn.Sigmoid()
        ]

        self.branch_1 = nn.Sequential(*branch_1)
        self.branch_2 = nn.Sequential(*branch_2)

    def forward(self, x):

        E = self.branch_1(x)
        M = self.branch_2(x)
        #M = self.branch_2(torch.cat((x,E),1))

        out = x*(1-M) + E*M

        return out


class FineNet_ResiAtt_Shared(nn.Module):

    def __init__(self):
        super(FineNet_ResiAtt_Shared, self).__init__()

        self.layer_1 = nn.Conv2d(31, 64, 3, 1, 1)
        self.layer_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.layer_3 = nn.Conv2d(64, 31, 3, 1, 1)
        self.layer_4 = nn.Conv2d(62, 31, 3, 1, 1)

        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self, x):

        out = self.Relu(self.layer_1(x))
        out = self.Relu(self.layer_2(out))

        E = self.layer_3(out)
        #M = self.sig(self.layer_4(out))
        M = self.sig(self.layer_4(torch.cat((x,E),1)))

        out = x * (1 - M) + E * M

        return out

class FineNet_SelfResiAtt(nn.Module):

    def __init__(self):
       super(FineNet_SelfResiAtt,self).__init__()
     
       self.layer_1 = nn.Conv2d(31, 31, 3, 1, 1)
       self.layer_2 = nn.Conv2d(31, 31, 3, 1, 1)
 
       branch_1 = [
           nn.Conv2d(31, 64, 3, 1, 1),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, 1, 1),
           nn.ReLU(),
           nn.Conv2d(64, 31, 3, 1, 1)
       ]

       branch_2 = [
           nn.Conv2d(31, 64, 3, 1, 1),
           nn.ReLU(),
           nn.Conv2d(64, 64, 3, 1, 1),
           nn.ReLU(),
           nn.Conv2d(64, 31, 3, 1, 1),
           nn.ReLU()
       ]

       self.branch_1 = nn.Sequential(*branch_1)
       self.branch_2 = nn.Sequential(*branch_2)
       self.Sig = nn.Sigmoid()

    def forward(self, x):

        #E = self.branch_1(x)
        Att = self.branch_2(x)

        M = self.Sig(self.layer_1(Att))
        B = self.layer_2(M*Att)

        return x + B


