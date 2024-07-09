import torch
import numpy as np
import scipy.io as sio
from LoadDataset_Batch import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from torch.nn.functional import upsample
from DownKernel import *
from downsampler import *

from ThreeBranch_3_FeaReg import *
from AffineTrans import *


lr = 1e-4
batch_size = 6
factor = 4

save_path = './Fusion_models/ThreeBranches_FeatureRegistration_{}_V3/'.format(factor)
if not os.path.exists(save_path):
    os.mkdir(save_path)

model = nn.DataParallel(ThreeBranch_Net().cuda())
down_spa = Downsampler(n_planes=31, factor=factor, kernel_type='gauss83', phase=0,preserve_size=True).type(torch.cuda.FloatTensor)

Train_image_num = 12   # CAVE:12, Harvard:20, ICLV:20

dataset = LoadDataset(Path='../Datasets/CAVE/', patch_size=128, stride=64, datasets='CAVE',  Data_Aug=True, up_mode='bicubic', Normlization=1, Train_image_num=Train_image_num)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# loss & optimizer
L1 = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# spatial and spectral downsampler
P = sio.loadmat('P_N_V2.mat')
P = Variable(torch.unsqueeze(torch.from_numpy(P['P']), 0)).type(torch.cuda.FloatTensor)


# learning rate decay
def LR_Decay(optimizer, n):
    lr_d = lr * (0.7**n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_d


Minimum_Loss = 1000
k_num = 0
n_num = 1

n = 0

start_time = time.time()

f = open(save_path+'loss.txt', 'w+')
f.write('The total loss is : \n\n\n')


# Training

for epoch in range(150):
    print('*'*10, 'The {}th epoch for training with DataAug_noSkip_3KS.'.format(epoch+1), '*'*10)
    running_loss = 0  # the total loss
    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0
    for iteration, Data in enumerate(data_loader, 1):

        GT = Data.type(torch.cuda.FloatTensor)

        # Generate the HR_MSI
        HR_MSI = torch.matmul(P, GT.reshape(-1, GT.shape[1], GT.shape[2]*GT.shape[3])).reshape(-1,3,GT.shape[2],GT.shape[3])
        # Generate the LR_HSI
        with torch.no_grad():
            LR_HSI = down_spa(GT)
        rota = torch.randint(0, 50, (1,))[0]/10
        shift = torch.randint(0, 5, (4,))
        a = torch.randint(0, 5, (1,))[0]
        size = torch.randint(0, int(LR_HSI.shape[2]/2), (2, ))
        LR_HSI = AffineTrans_Single(LR_HSI, rota/360*2*3.14, shift, a)
        UP_HSI = upsample(LR_HSI, size=[GT.shape[2], GT.shape[3]], mode='bilinear')

        # Generate the UP_HSI
        Input_1 = Variable(HR_MSI, requires_grad=False).type(torch.cuda.FloatTensor)
        Input_2 = Variable(torch.cat((UP_HSI, HR_MSI), 1), requires_grad=False).type(torch.cuda.FloatTensor)
        Input_3 = Variable(UP_HSI, requires_grad=False).type(torch.cuda.FloatTensor)

        out = model(Input_1, Input_2, Input_3)

        loss = L1(out, GT)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data.cpu()

    if epoch % 10 == 0:
        LR_Decay(optimizer, n)
        n += 1
        print('Adjusting the learning rate by timing 0.8.')

    print('*'*10, 'At the {}-th epoch, the loss is {:.4f}.'.format(epoch, running_loss,), '*'*10)
    f.write('The loss at {}th epoch is{:.4f}.\n'.format(epoch, running_loss))

    if epoch % 10 == 9:
        torch.save(model, save_path+'model_'+str(int(epoch/10))+'.pth')

T = time.time()-start_time

print('Total training time is {}'.format(T))
f.write('Total traing time is {}.\n'.format(T))

f.close()







