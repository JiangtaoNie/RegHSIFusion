import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
# from scipy.misc import imresize
import copy
# from Spa_downs import *
import torch.nn.functional as F
from torch.nn.functional import upsample

names_CAVE = [
    'balloons_ms','thread_spools_ms', 'fake_and_real_food_ms', 'face_ms','feathers_ms', 'cd_ms', 'real_and_fake_peppers_ms',
    'stuffed_toys_ms', 'sponges_ms', 'oil_painting_ms', 'fake_and_real_strawberries_ms', 'fake_and_real_beers_ms',
    'real_and_fake_apples', 'superballs', 'chart_and_stuffed_toy', 'hairs',  'fake_and_real_lemons',
    'fake_and_real_lemon_slices', 'fake_and_real_sushi', 'egyptian_statue', 'glass_tiles', 'jelly_beans',
    'fake_and_real_peppers', 'clay', 'pompoms', 'watercolors', 'fake_and_real_tomatoes', 'flowers', 'paints',
    'photo_and_face', 'cloth', 'beads'
]

names_Harvard = [
    'imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4', 'imgb1', 'imge1', 'imga6', 'imgh2', 'imgb3', 'imgf3',
    'imgf4', 'imge0', 'imgd3', 'img2', 'imgf2', 'imge5', 'imgc8', 'imge2', 'imgc7', 'imgb9', 'imgh3',
    'imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 'imgf7', 'imgc2', 'imgf5',
    'imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 'imgb0', 'imgd8', 'imgb8'
]

names_ICLV = [
    'BGU_HS_00001','BGU_HS_00030','BGU_HS_00060','BGU_HS_00090','BGU_HS_00120','BGU_HS_00150','BGU_HS_00180',
    'BGU_HS_00020', 'BGU_HS_00040', 'BGU_HS_00050', 'BGU_HS_00070', 'BGU_HS_00080', 'BGU_HS_00100', 'BGU_HS_00110',
    'BGU_HS_00130', 'BGU_HS_00140', 'BGU_HS_00160', 'BGU_HS_00170', 'BGU_HS_00190', 'BGU_HS_00200','BGU_HS_00010',
]

Rotation = ['1', '2', '3', '4', '5', '-5', '-4', '-3', '-2', '-1']


class LoadDataset(data.Dataset):
    def __init__(self, Path, patch_size=128, stride=64, datasets='CAVE', Data_Aug=False, up_mode='bicubic', Normlization=0, Train_image_num=12):
        super(LoadDataset, self).__init__()

        if datasets == 'CAVE':
            self.names = names_CAVE
        elif datasets == 'Harvard':
            self.names = names_Harvard
        elif datasets == 'ICLV':
            self.names = names_ICLV
        else:
            assert 'wrong dataset name'

        self.path = Path
        self.P_S = patch_size
        self.stride = stride
        self.DA = Data_Aug
        self.Image_size = 512
        self.up_mode = up_mode
        self.Norm = Normlization
        self.img_num = Train_image_num
        self.P_N = int(self.Image_size/self.stride)

    def __getitem__(self, Index):

        P_S = self.P_S
        S = self.stride
        P_N = self.P_N

        if self.DA:
            Aug = 2
        else:
            Aug = 1

        Image_size = self.Image_size
        Patches = P_N**2
        Image_I = int(Index/Aug/Patches)
        Patch_I = int(Index/Aug%Patches)

        HSI = sio.loadmat(self.path+self.names[Image_I]+'.mat')['data']

        if self.Norm == 0:
            HSI = HSI/1.9321
        elif self.Norm == 1:
            HSI = HSI/(np.max(HSI)-np.min(HSI))

        X = int(Patch_I/P_N) #X,Y is patch index in image
        Y = int(Patch_I%P_N)

        s = int(S/8)       ### set the scal factor as 8
        p_s = int(P_S/8)

        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            GT = HSI[:, -P_S:, Y * S: Y * S + P_S]
        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, X * S:X * S + P_S, -P_S:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, -P_S: , -P_S: ]
        else:
            GT = HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
        
        # Data augmantation
        if self.DA :
            if Index%2 == 1:
                a = np.random.randint(0,6,1)
                if a[0] == 0:
                    GT = copy.deepcopy(np.flip(GT, 1))  # flip the array upside down
                    # LR_HSI = copy.deepcopy(np.flip(LR_HSI, 1))  # flip the array upside down
                elif a[0] == 1:
                    GT = copy.deepcopy(np.flip(GT, 2))  # flip the array left to right
                    # LR_HSI = copy.deepcopy(np.flip(LR_HSI, 2))  # flip the array left to right
                elif a[0] == 2:
                    GT = copy.deepcopy(np.rot90(GT, 1, [1, 2]))  # Rotate 90 degrees clockwise
                    # LR_HSI = copy.deepcopy(np.rot90(LR_HSI, 1, [1, 2]))  # Rotate 90 degrees clockwise
                elif a[0] == 3:
                    GT = copy.deepcopy(np.rot90(GT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                    # LR_HSI = copy.deepcopy(np.rot90(LR_HSI, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                elif a[0] == 4:
                    GT = copy.deepcopy(np.roll(GT, int(GT.shape[1] / 2), 1))  # Roll the array up
                    # LR_HSI = copy.deepcopy(np.roll(LR_HSI, int(LR_HSI.shape[1] / 2), 1))  # Roll the array up
                elif a[0] == 5:
                    GT = np.roll(GT, int(GT.shape[1] / 2), 1)  # Roll the array up & left
                    # LR_HSI = np.roll(LR_HSI, int(LR_HSI.shape[1] / 2), 1)  # Roll the array up & left
                    GT = copy.deepcopy(np.roll(GT, int(GT.shape[2] / 2), 2))
                    # LR_HSI = copy.deepcopy(np.roll(LR_HSI, int(LR_HSI.shape[2] / 2), 2))
        GT = torch.from_numpy(GT)
        # LR_HSI = torch.Tensor(LR_HSI)
        return GT #, LR_HSI

    def __len__(self):

        if self.DA:
            Aug = 2
        else:
            Aug = 1

        return int(self.P_N**2*self.img_num*Aug)
