import logging
from os import listdir
import os
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
import cv2
from skimage.transform import resize
from torch.utils.data import Dataset

class mySegmentationDataset(Dataset):
    def __init__(self, root_dir: str, nonlabel_path: str, havelabel_path: str, dataset: str, scale: float = 1.0):
        self.root_dir = root_dir
        self.nonlabel_path = nonlabel_path
        self.havelabel_path = havelabel_path
        self.dataset_name = dataset
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        with open(self.nonlabel_path, 'r') as nlf:
            lines = nlf.readlines()
            non_label_lines = [line.strip().split(' ')[:2] for line in lines]
        
        with open(self.havelabel_path, 'r') as hlf:
            lines = hlf.readlines()
            have_label_lines = [line.strip().split(' ')[:2] for line in lines]

        choose_non_lable_lines = np.random.choice(len(non_label_lines), size = len(have_label_lines))
        non_label_lines = np.array(non_label_lines, dtype= object)
        have_label_lines = np.array(have_label_lines, dtype= object)
        self.ids = np.concatenate([non_label_lines[choose_non_lable_lines], have_label_lines], axis= 0)
        # self.ids = os.listdir(images_dir) #[splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.') and image_type in file]
        # print(len(self.ids))
        # if datasetname == "las_mri":
        #     self.ids = [f for f in self.ids if image_type in f]
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(self, img, scale, is_mask):
        # img = cv2.resize(img, (224, 224)) #sai
        img = resize(img, (224, 224), order=0, preserve_range=True, anti_aliasing=False).astype('uint8') #sai
        # print(np.unique(img))
        img = np.asarray(img)
        if not is_mask:
            img = np.expand_dims(img, axis=0)
        return img

    @classmethod
    def load(self, filename, is_mask=False):
        if is_mask:
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            return cv2.imread(filename, 0)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img_file = os.path.join(self.root_dir, self.ids[idx][0])
        mask_file = os.path.join(self.root_dir, self.ids[idx][1])
        # print(img_file)

        mask = self.load(mask_file, is_mask=True)
        img = self.load(img_file, is_mask=False)

        assert mask is not None, mask_file
        assert img is not None, img_file
        assert img.size == mask.size, img_file + " @@ " + mask_file
        if self.dataset_name.startswith("las"):
            mask[mask == 30] = 1
            mask[mask == 60] = 2 # main predict
            mask[mask == 90] = 3
            mask[mask == 120] = 4
            mask[mask == 150] = 5
            mask[mask == 180] = 6
            mask[mask == 210] = 7
            mask[mask > 7] = 0
        else:
            mask[mask > 1] = 1 

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        data = {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy().astype(int)).long().contiguous()
        }
        self.cache[idx] = data
        return data

    def get_3d_iter(self):
        # Brats18_2013_0_1_flair_frame_0.png
        from itertools import groupby
        keyf = lambda idx : self.ids[idx].split("_frame_")[0]
        sorted_ids = sorted(range(len(self.ids)), key=lambda i : self.ids[i])
        for _, items in groupby(sorted_ids, key=keyf):
            images = []
            masks = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks.append(d['mask'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            _3d_data = {'image': images, 'mask': masks}
            yield _3d_data

class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0,
                 image_type : str = "_flair_", datasetname ="las_mri"):
        self.dataset_name = datasetname
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.image_type = image_type

        self.ids = os.listdir(images_dir) #[splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.') and image_type in file]

        if datasetname == "bts":
            self.ids = [f for f in self.ids if image_type in f]
        if len(self.ids) == 0:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(self, img, scale, is_mask):
        # img = cv2.resize(img, (224, 224), cv2.IMREAD_UNCHANGED) #sai
        img = resize(img, (224, 224), order=0, preserve_range=True, anti_aliasing=False).astype('uint8') #sai
        # print(np.unique(img))        
        img = np.asarray(img)
        if not is_mask:
            img = np.expand_dims(img, axis=0)
        return img

    @classmethod
    def load(self, filename, is_mask=False):
        if is_mask:
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            return cv2.imread(filename, 0)

    def get_mask_file(self, image_file):
        if self.dataset_name != "bts":
            return image_file.replace("image","label")
        else:
            return image_file.replace(self.image_type, "_seg_")

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        name = self.ids[idx]
        mask_name = self.get_mask_file(name)
        # print(name)
        mask_file = os.path.join(self.masks_dir, mask_name) #list(self.masks_dir.glob(mask_name+'.*'))
        img_file = os.path.join(self.images_dir, name) #list(self.images_dir.glob(name + '.*'))
        mask = self.load(mask_file, is_mask=True)
        img = self.load(img_file, is_mask=False)

        assert mask is not None, mask_file
        assert img is not None, img_file
        assert img.size == mask.size, img_file + " @@ " + mask_file
        # segment the whole tumor
        if self.dataset_name.startswith("las"):
            mask[mask == 30] = 1
            mask[mask == 60] = 2 # main predict
            mask[mask == 90] = 3
            mask[mask == 120] = 4
            mask[mask == 150] = 5
            mask[mask == 180] = 6
            mask[mask == 210] = 7
            mask[mask > 7] = 0
        else:
            mask[mask>1]=1 
        #mask[mask==4]=1
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        data = {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy().astype(int)).long().contiguous()
            ,'mask_file' : mask_file          
            ,'img_file' : img_file
        }
        self.cache[idx] = data
        return data
        
    def get_3d_iter(self):
        # Brats18_2013_0_1_flair_frame_0.png
        from itertools import groupby
        keyf = lambda idx : self.ids[idx].split("_frame_")[0]
        sorted_ids = sorted(range(len(self.ids)), key=lambda i : self.ids[i])
        for _, items in groupby(sorted_ids, key=keyf):
            images = []
            masks = []
            for idx in items:
                d = self.__getitem__(idx)
                images.append(d['image'])
                masks.append(d['mask'])
            # store third dimension in image channels
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            _3d_data = {'image': images, 'mask': masks}
            yield _3d_data


