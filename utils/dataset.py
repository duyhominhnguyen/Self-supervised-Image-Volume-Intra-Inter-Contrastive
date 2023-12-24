import os
import time 
import torch
import random
import torchvision
import numpy as np
import torchio as tio
from random import sample
from PIL import ImageFilter
from einops import rearrange,repeat
from torch import nn , Tensor
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
    
    
class MaskGenerator:
    '''
    Generate MASK for MASK PREDICTION STEP
    '''
    def __init__(self, n_frames = 32, mask_ratio=0.1):

        self.n_frames = n_frames
        self.mask_ratio = mask_ratio
        self.mask_count = int(np.ceil(self.n_frames * self.mask_ratio))
        
    def __call__(self):
        indices_to_mask = torch.randn((self.n_frames,1))
        indices_to_mask = indices_to_mask.topk(
            k= self.mask_count,
            dim=0,
            )
        indices_to_mask = indices_to_mask.indices
        bitmask = torch.zeros((self.n_frames,1))
        bitmask = bitmask.scatter(
            dim=0,
            index=indices_to_mask,
            value=1
            )
        bitmask = bitmask.bool()
        return bitmask

class TrainingDataset_Mask(Dataset):
    '''
    Training dataset for MASK task using TORCHIO AUGMENTATION
    '''
    def __init__(self,
                 data_path,
                 data_name, 
                 target_shape,
                 mask_ratio,
                 augment = []
                ):
        super(TrainingDataset_Mask, self).__init__()
        
        
        self.data_path = os.path.join(data_path, data_name)
            
        self.img_names  = os.listdir(self.data_path)
        self.augment = augment 
        self.transforms = tio.transforms.CropOrPad(target_shape = target_shape)
        self.mask_generator = MaskGenerator(n_frames = target_shape[0], mask_ratio=mask_ratio)
        self.cache = {}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        
        if index in self.cache:
            return self.cache[index]
        
        img_name_i = self.img_names[index]
        img_name_i = os.path.join(self.data_path,img_name_i) 
        img_i = np.load(img_name_i)
        
        #  process data step   
        img_i = torch.from_numpy(img_i)
        #if ("LUNA2016" in self.data_path) or  ("LiTS2017" in self.data_path) or ("BraTS2018" in self.data_path) :
        img_i = img_i.permute((2,0,1))
        img_i = torch.unsqueeze(img_i, 0)
        if self.transforms :
            img_i = self.transforms(img_i)
                
        #  augmentation step         
        if not self.augment :
            raise ValueError('Need to augment data for using SwAV')
        else :
            randList =  sample(range(len(self.augment)),2)
            view1 = self.augment[randList[0]](img_i).squeeze()
            view2 = self.augment[randList[1]](img_i).squeeze()
            
        view1 =  repeat(view1, 'd h w -> d c h w', c=3)
        view2 =  repeat(view2, 'd h w -> d c h w', c=3)
        mask = self.mask_generator()  
        data = {
            'view1': view1.float().contiguous(),
            'view2': view2.float().contiguous(),
            'mask' : mask,
            'img_name' : img_name_i
        }
        self.cache[index] = data
        return data
    
    
#  ========================================================== SSL DATASET IMPLEMENTATION =================================================
    
    
class TrainingDataset_SSL(Dataset):
    '''
    Training dataset for MASK task using TORCHIO AUGMENTATION
    '''
    def __init__(self,
                 data_path,
                 data_name, 
                 target_shape,
                 augment = []
                ):
        
        
        self.data_path = os.path.join(data_path, data_name)
        self.img_names  = os.listdir(self.data_path)
        self.augment = augment 
        self.transforms = tio.transforms.CropOrPad(target_shape = target_shape)
        self.cache = {}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        
        if index in self.cache:
            return self.cache[index]
        
        img_name_i = self.img_names[index]
        img_name_i = os.path.join(self.data_path,img_name_i) 
        img_i = np.load(img_name_i)

        img_i = torch.from_numpy(img_i)
        img_i = img_i.permute((2,0,1))
        img_i = torch.unsqueeze(img_i, 0)
        if self.transforms :
            img_i = self.transforms(img_i)
                
        if not self.augment :
            raise ValueError('Need to augment data for using SwAV')
        else :
            randList =  sample(range(len(self.augment)),2)
            view1 = self.augment[randList[0]](img_i).squeeze()
            view2 = self.augment[randList[1]](img_i).squeeze()
            
        view1 =  repeat(view1, 'd h w -> d c h w', c=3)
        view2 =  repeat(view2, 'd h w -> d c h w', c=3)
        data = {
            'index': index,
            'view1': view1.float().contiguous(),
            'view2': view2.float().contiguous(),
            'img_name' : img_name_i
        }
        self.cache[index] = data
        return data
    
    
#  ========================================================== Other FUNCTION ============================================================


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
    
def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
    


class OurDataset(Dataset):
    def __init__(self, data_path,
                 split, device):
        self.data_path = data_path + split
        print(self.data_path)
        self.img_names  = os.listdir(self.data_path)
        self.device = device
        self.cache = {}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        img_name_i = self.img_names[index]
        class_i = float(int(img_name_i[-5]))
        img_name_i = os.path.join(self.data_path, img_name_i)
        img_i =  torch.tensor(np.load(img_name_i))
        img_i = torch.permute(img_i, (2,0,1))
        #  expand to shape (32, 3 , 64 , 64)
        target_shape = [32, 3, 64, 64]
        img_i = img_i[:, None, :, :].expand(target_shape)

        data = {
            'image': img_i.float().contiguous(),
            'mask': class_i,
            'mask_file' : class_i,
            'img_file' : img_name_i
        }
        self.cache[index] = data
        return data
