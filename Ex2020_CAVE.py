import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from downsampler import *
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
from model import *
import time
import copy
from torch.nn.parameter import Parameter
import os
from function import *
from STN import *
from L1_Focal_loss import *
from DownKernel import *
from ThreeBranch_3_FeaReg import *

names_CAVE_test = [
    'real_and_fake_apples', 'superballs', 'chart_and_stuffed_toy', 'hairs',  'fake_and_real_lemons',
    'fake_and_real_lemon_slices', 'fake_and_real_sushi', 'egyptian_statue', 'glass_tiles', 'jelly_beans',
    'fake_and_real_peppers', 'clay', 'pompoms', 'watercolors', 'fake_and_real_tomatoes', 'flowers', 'paints',
    'photo_and_face', 'cloth', 'beads'
]

names_Harvard_test = [
    'imgb9', 'imgh3','imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 
    'imgf7', 'imgc2', 'imgf5','imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 
    'imgb0', 'imgd8', 'imgb8'
]

names_ICLV_test = [
    'BGU_HS_00003', 'BGU_HS_00007', 'BGU_HS_00013', 'BGU_HS_00017', 'BGU_HS_00023', 'BGU_HS_00027', 'BGU_HS_00033',
    'BGU_HS_00037', 'BGU_HS_00043', 'BGU_HS_00047', 'BGU_HS_00053', 'BGU_HS_00057', 'BGU_HS_00063', 'BGU_HS_00067',
    'BGU_HS_00073', 'BGU_HS_00077', 'BGU_HS_00083', 'BGU_HS_00087', 'BGU_HS_00093', 'BGU_HS_00097', 'BGU_HS_00103',
    'BGU_HS_00107', 'BGU_HS_00113', 'BGU_HS_00117', 'BGU_HS_00123', 'BGU_HS_00127', 'BGU_HS_00133', 'BGU_HS_00137',
    'BGU_HS_00143', 'BGU_HS_00147', 'BGU_HS_00153', 'BGU_HS_00157', 'BGU_HS_00163', 'BGU_HS_00167', 'BGU_HS_00173',
    'BGU_HS_00177', 'BGU_HS_00183', 'BGU_HS_00187', 'BGU_HS_00193', 'BGU_HS_00197', 'BGU_HS_00203', 'BGU_HS_00207',
    'BGU_HS_00213', 'BGU_HS_00217', 'BGU_HS_00223', 'BGU_HS_00227', 'BGU_HS_00233', 'BGU_HS_00237', 'BGU_HS_00243',
    'BGU_HS_00247'
]

##parameter setting
U_spa = 0   # 0 refers to the spatial downsampler is known 
U_spc = 0
pretrain = 0
num_steps = 500		#500 
ReNew = 50		#50
Stop_Regis = 10		#100
Dsets_name = 'L1 Noised LR_HSI 30NSR'
factor = 8
Loss_used = 'L'   #there has 'SAM','MSE','L' loss
lr_i = 1.2e-3    #defalt
lr_stn = 2e-3
ld = 0.7         #the lr decrease rate

a = 1.5  #the coefficient of weight

Rad = 1
Shift = 5
Amp = 5
aa = 1
bb = 1
cc = 1

path = 'CAVE_X{}_R{}_S{}_A{}_40N/'.format(factor, Rad, Shift, Amp)
N_path = './TestSet/CAVE/'+path
N_path_2 = './TestSet/CAVE/CAVE_MSI/'

model_path = './Fusion_models/ThreeBranches_FeatureRegistration_{}_V3/model_14.pth'.format(factor)

save_path = './Results/Parameter_{}_{}_{}_'.format(aa, bb, cc)+path

if not os.path.exists(save_path):
    os.mkdir(save_path)

dir_data = './TestSet/CAVE/CAVE/'

D_net = 3

#get the spectual downsample func
def create_P():
    P_ = [[2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
          [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2,  2,  1,  1,  2,  2,  2,  2,  2],
          [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]
    P = np.array(P_, dtype=np.float32)
    div_t = np.sum(P, axis=1)
    for i in range(3):
        P[i,] = P[i,]/div_t[i]
    return P

#get PSNR
def PSNR_GPU(im_true, im_fake):
    data_range = 1
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C*H*W)
    Ifake = im_fake.clone().resize_(C*H*W)
    #mse = nn.MSELoss(reduce=False)
    err = torch.pow(Itrue-Ifake,2).sum(dim=0, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)


#get SAM
def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone().resize_(C, H*W)
    Ifake = im_fake.clone().resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0).resize_(H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=0).sqrt_().resize_(H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=0).sqrt_().resize_(H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (H*W)
    return sam


# main processing
if __name__ =='__main__':

    #define the lossfunction & optimizer
    mse = nn.MSELoss()
    L1Loss = nn.L1Loss()
    k = 0
    files = os.listdir(dir_data)

    #loading the pretrained fusion model
    model = torch.load(model_path).cuda()

    for name in names_CAVE_test:
        lr = lr_i
        k += 1
        data = sio.loadmat(dir_data+name+'_ms.mat')
        
        print('Producing with the {}th image:{}'.format(k,name))
        im_gt = data['data']
        p = create_P()
        p = Variable(torch.from_numpy(p.copy()), requires_grad=False).cuda()

        net_input = np.zeros(im_gt.shape, dtype=np.float32)
        #trans the data to Variable
        im_gt = Variable(torch.from_numpy(im_gt.copy()),requires_grad=False).type(torch.cuda.FloatTensor).cuda()
        im_gt = im_gt/(torch.max(im_gt)-torch.min(im_gt))
        s = im_gt.shape
        GT = im_gt.view(s[0],s[1]*s[2])
        
        #Known the spatial downsampler
        down_spa = Downsampler(n_planes=im_gt.shape[0], factor=factor, kernel_type='gauss83', phase=0,preserve_size=True).type(torch.cuda.FloatTensor)
        down_RGB = Downsampler(n_planes=3, factor=factor, kernel_type='gauss83', phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

        N_data = sio.loadmat(N_path+name+'.mat')
        im_h_ = Variable(torch.from_numpy(N_data['LR_HSI']), requires_grad=False).type(torch.cuda.FloatTensor).unsqueeze(0)

        N_data = sio.loadmat(N_path_2 + name + '.mat')
        im_m = torch.from_numpy(N_data['HR_MSI'])
        im_m = Variable(im_m, requires_grad=False).type(torch.cuda.FloatTensor).unsqueeze(0)

        #Spatial transformation

        lr_size = int(512/factor)
        Resize = nn.Upsample(size=[lr_size, lr_size], mode='nearest')
        RGB_down = down_RGB(im_m)
        
        l1 = nn.L1Loss()
       
        start_time = time.time()

        net = FineNet_SelfAtt().cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        LR_up = F.interpolate(im_h_, [512, 512], mode='bilinear')
        im_cat = torch.cat((LR_up, im_m), 1)

        with torch.no_grad():
            Input = model(im_m, im_cat, LR_up)

        psnr_ = PSNR_GPU(im_gt, Input.squeeze())
        sam_ = SAM_GPU(im_gt, Input.squeeze())
        print('The PSNR and SAM of rough restored HSI are {} and {}.'.format(psnr_, sam_))


        STN = STN_Net(31).cuda()
        optimizer_STN = torch.optim.Adam(STN.parameters(), lr=lr_stn, weight_decay=1e-4)

        f = open(save_path+name+'_result.txt','a+')
        f.write('\n\n\nThe result of PSNR & SAM is :\n\n')
        f.write('The experiment:At {} times SR & All the downsampler is unknown & the DSpa is {}.\n\n'.format(factor,Dsets_name))
        f2 = open(save_path+name+'_loss.txt','w')
        f2.write('\n\n\nThe experiment loss is :\n\n')

        print('Stage three : Producing with the {} image'.format(name))
        HSI_i = Variable(torch.zeros([1, 31, 512, 512]), requires_grad=False).type(torch.cuda.FloatTensor)
        weight_h = Variable(torch.ones([1, 31, lr_size, lr_size]), requires_grad=False).type(torch.cuda.FloatTensor)
        weight_m = Variable(torch.ones([1, 3, 512, 512]), requires_grad=False).type(torch.cuda.FloatTensor)
        for i in range(num_steps):
            im_h = STN(Resize(im_h_))
            output = net(Input)
            HSI_RGB = torch.matmul(p, im_h.reshape(31, lr_size**2)).view(1, 3, lr_size, lr_size)

            S = output.shape
            if U_spa == 0:
                Dspa_O = down_spa(output)
                Dspa_O = Dspa_O.view(Dspa_O.shape[1],Dspa_O.shape[2],Dspa_O.shape[3])
            else:
                Dspa_O = downs(output)
                Dspa_O = torch.squeeze(Dspa_O,0)
            if U_spc == 0:
                out = output.view(S[1],S[2]*S[3])
                Dspc_O = torch.matmul(p,out).view(im_m.shape[1],S[2],S[3])
            else:
                Dspc_O = down_spc(output)

            #zero the grad
            optimizer.zero_grad()
            optimizer_STN.zero_grad()


            if i > Stop_Regis:
                loss = aa*l1(Dspa_O, im_h.squeeze()) + bb*l1(Dspc_O, im_m.squeeze()) + cc*l1(RGB_down, HSI_RGB)  #Ours Weighted loss

            else:
                loss = l1(HSI_RGB, RGB_down) + l1(Dspc_O, im_m.squeeze()) #pertrain loss, aim to learn the STN first

            '''if i == Stop_Regis:
                lr = 1e-5
                for param_group in optimizer_STN.param_groups:
                    param_group['lr'] = lr
                    # for param_group in optimizer_d.param_groups:'''

            #backward the loss
            loss.backward(retain_graph=True)
            #optimize the parameter
            optimizer.step()
            optimizer_STN.step()

            if i%ReNew == ReNew-1:
                im_cat = torch.cat((output.detach(), im_m), 1).type(torch.cuda.FloatTensor)
                weight = torch.exp(torch.abs(output.detach() - HSI_i) * a)
                HSI_i = output.detach()
                weight_h = Variable(down_spa(weight).squeeze(), requires_grad=False).type(torch.cuda.FloatTensor)
                weigth_m = Variable(torch.matmul(p,weight.view(S[1], S[2]*S[3])).view(im_m.shape[1], S[2], S[3]), 
                         requires_grad=False).type(torch.cuda.FloatTensor)

            if i%10 == 0:
                f2.write('At step {},the loss is {}\n'.format(i,loss.detach().cpu()))
        
            if i % 50 == 0:
                out = Variable(output,requires_grad=False).cuda()
                out = out.view(S[1],S[2],S[3])
                psnr = PSNR_GPU(im_gt,out)
                sam = SAM_GPU(im_gt,out)
                f.write('{},{}\n'.format(psnr,sam))
                print('**{}**{}**At the {}th loop the loss&PSNR&SAM is {} {},{}.'.format(k,Dsets_name,i,loss.data,psnr,sam))
            if i % 1000 == 0:
                #change the learning rate
                lr = ld*lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        data = {}
        out = np.array(output.squeeze().detach().cpu())
        data['data'] = out
        data['LR_HSI'] = np.array(im_h.squeeze().detach().cpu())
        sio.savemat(save_path+name+'_r.mat',data)
        torch.save(STN, save_path+name+'_STN.pth')

        used_time = time.time()-start_time
        f.write('The training time is :{}.'.format(used_time))
        f.close()
        f2.close()

        torch.save(net,save_path+'model.pth')
        torch.cuda.empty_cache()




