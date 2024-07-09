from function import *
from SSIM import *
import numpy as np
import scipy.io as sio
import torch
import os
#import h5py
import matplotlib.pyplot as plt


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


def RMSE_torch(x1, x2):
    MSE = torch.mean((x1-x2)**2)
    return torch.sqrt(MSE)*255



out_path = ''


GT_path = ''
names = names_CAVE_test

RMSE_SUM = 0
PSNR_SUM = 0
SAM_SUM = 0
SSIM_SUM = 0
K = 0

for name in names:

    data_RE = sio.loadmat(out_path+name+'.mat')

    data_GT = sio.loadmat(GT_path+name+'_ms.mat')

    RE = torch.from_numpy(data_RE['SR']).squeeze().type(torch.FloatTensor)
    GT = torch.from_numpy(data_GT['data']).type(torch.FloatTensor)
    GT = GT/(torch.max(GT)-torch.min(GT))
    
    SSIM = ssim_GPU(GT.unsqueeze(0),RE.unsqueeze(0))
    RMSE = RMSE_torch(GT, RE)
    PSNR = PSNR_GPU(GT, RE)
    SAM  = SAM_GPU(GT, RE)

    print('The result of image {0:26} {1:.4f}, {2:.2f}, {3:.2f}, {4:.4f}'.format(name,RMSE,PSNR,SAM,SSIM))
    if PSNR >= 10 and SAM <= 20:
        RMSE_SUM += RMSE
        PSNR_SUM += PSNR
        SAM_SUM  += SAM
        SSIM_SUM += SSIM
        K += 1

print('The average result of this dataset are:{0:.4f}, {1:.2f}, {2:.2f}, {3:.4f}'.format(RMSE_SUM/K, PSNR_SUM/K, SAM_SUM/K, SSIM_SUM/K))





