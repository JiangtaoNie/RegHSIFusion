import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import os
import copy
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.autograd import Variable
from ThreeBranch_3 import *

names_CAVE = [
    'real_and_fake_apples', 'superballs', 'chart_and_stuffed_toy', 'hairs',  'fake_and_real_lemons',
    'fake_and_real_lemon_slices', 'fake_and_real_sushi', 'egyptian_statue', 'glass_tiles', 'jelly_beans',
    'fake_and_real_peppers', 'clay', 'pompoms', 'watercolors', 'fake_and_real_tomatoes', 'flowers', 'paints',
    'photo_and_face', 'cloth', 'beads'
]

names_Harvard = [
    'imgb9', 'imgh3','imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 'imgf7', 'imgc2', 'imgf5',
    'imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 'imgb0', 'imgd8', 'imgb8'
]

names_ICLV = [
    'BGU_HS_00001', 'BGU_HS_00030', 'BGU_HS_00060', 'BGU_HS_00090', 'BGU_HS_00120', 'BGU_HS_00150', 'BGU_HS_00180',
    'BGU_HS_00020', 'BGU_HS_00040', 'BGU_HS_00050', 'BGU_HS_00070', 'BGU_HS_00080', 'BGU_HS_00100', 'BGU_HS_00110',
    'BGU_HS_00130', 'BGU_HS_00140', 'BGU_HS_00160', 'BGU_HS_00170', 'BGU_HS_00190', 'BGU_HS_00200', 'BGU_HS_00010',
]

factor = 8

# model_path = './Fusion_models/ThreeBranches_FeatureRegistration_4_V1/'
# model_path = './Fusion_models/ThreeBranches_FeatureRegistration_{}_V3/'.format(factor)
model_path = './Fusion_models/ThreeBranches_FeatureRegistration_4_V3/'.format(factor)
# model_path = './Fusion_models/ThreeBranches_Baseline_{}_V1/'.format(factor)
# model_path = './Fusion_models/ThreeBranches_Baseline_4_V1/'
net = torch.load(model_path+'model_14.pth')

path = 'CAVE_X{}_R1_S5_A5_40N/'.format(factor)
# path = 'CAVE_83_S{}_Rota5_RDShift5/'.format(factor)
test_path = './TestSet/CAVE/'+path
test_path_2 = './TestSet/CAVE/CAVE_MSI/'

save_path = './Results/CAVE/Fusion_X4'+path
# save_path = './Results/Baseline_WOFeaReg'+test_path[10:]

if not os.path.exists(save_path):
    os.mkdir(save_path)

for name in names_CAVE:
    print('Dealing with the image: {}'.format(name))
    data = sio.loadmat(test_path + name)
    LR_HSI = torch.from_numpy(data['LR_HSI']).unsqueeze(0).type(torch.cuda.FloatTensor)
    data = sio.loadmat(test_path_2 + name)
    HR_MSI = torch.from_numpy(data['HR_MSI']).unsqueeze(0).type(torch.cuda.FloatTensor)
    UP_HSI = upsample(LR_HSI, size=[HR_MSI.shape[2], HR_MSI.shape[3]], mode='bilinear')

    # Generate the UP_HSI
    Input_1 = Variable(HR_MSI, requires_grad=False).type(torch.cuda.FloatTensor)
    Input_2 = Variable(torch.cat((UP_HSI, HR_MSI), 1), requires_grad=False).type(torch.cuda.FloatTensor)
    Input_3 = Variable(UP_HSI, requires_grad=False).type(torch.cuda.FloatTensor)

    with torch.no_grad():
        out = net(Input_1, Input_2, Input_3)

    sio.savemat(save_path+name+'.mat', {'SR':out.squeeze().detach().cpu().numpy()})
    torch.cuda.empty_cache()

