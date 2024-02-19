from torch.utils.data import Dataset
import os
import torch
import numpy as np
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip)
import cv2

class Dataset_PSDM_train(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'ct')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'ct')))

        self.transforms = Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.3, value=None,
                             mask_value=None, border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.3), VerticalFlip(p=0.3)], p=0.8)
        self.len = len(self.file_name_list)

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct = np.load(os.path.join(file_dir, 'ct', file_name))[:, :, np.newaxis]
        dose = np.load(os.path.join(file_dir, 'dose', file_name))[:, :, np.newaxis]

        Mask_Brainstem = np.load(os.path.join(file_dir, 'Mask_Brainstem', file_name))[:, :, np.newaxis]
        Mask_Esophagus = np.load(os.path.join(file_dir, 'Mask_Esophagus', file_name))[:, :, np.newaxis]
        Mask_Larynx = np.load(os.path.join(file_dir, 'Mask_Larynx', file_name))[:, :, np.newaxis]
        Mask_LeftParotid = np.load(os.path.join(file_dir, 'Mask_LeftParotid', file_name))[:, :, np.newaxis]
        Mask_Mandible = np.load(os.path.join(file_dir, 'Mask_Mandible', file_name))[:, :, np.newaxis]
        Mask_possible_dose_mask = np.load(os.path.join(file_dir, 'Mask_possible_dose_mask', file_name))[:, :, np.newaxis]
        Mask_PTV56 = np.load(os.path.join(file_dir, 'Mask_PTV56', file_name))[:, :, np.newaxis]
        Mask_PTV63 = np.load(os.path.join(file_dir, 'Mask_PTV63', file_name))[:, :, np.newaxis]
        Mask_PTV70 = np.load(os.path.join(file_dir, 'Mask_PTV70', file_name))[:, :, np.newaxis]
        Mask_RightParotid = np.load(os.path.join(file_dir, 'Mask_RightParotid', file_name))[:, :, np.newaxis]
        Mask_SpinalCord = np.load(os.path.join(file_dir, 'Mask_SpinalCord', file_name))[:, :, np.newaxis]

        PSDM_Brainstem = np.load(os.path.join(file_dir, 'PSDM_Brainstem', file_name))[:, :, np.newaxis]
        PSDM_Esophagus = np.load(os.path.join(file_dir, 'PSDM_Esophagus', file_name))[:, :, np.newaxis]
        PSDM_Larynx = np.load(os.path.join(file_dir, 'PSDM_Larynx', file_name))[:, :, np.newaxis]
        PSDM_LeftParotid = np.load(os.path.join(file_dir, 'PSDM_LeftParotid', file_name))[:, :, np.newaxis]
        PSDM_Mandible = np.load(os.path.join(file_dir, 'PSDM_Mandible', file_name))[:, :, np.newaxis]
        PSDM_possible_dose_mask = np.load(os.path.join(file_dir, 'PSDM_possible_dose_mask', file_name))[:, :, np.newaxis]
        PSDM_PTV56 = np.load(os.path.join(file_dir, 'PSDM_PTV56', file_name))[:, :, np.newaxis]
        PSDM_PTV63 = np.load(os.path.join(file_dir, 'PSDM_PTV63', file_name))[:, :, np.newaxis]
        PSDM_PTV70 = np.load(os.path.join(file_dir, 'PSDM_PTV70', file_name))[:, :, np.newaxis]
        PSDM_RightParotid = np.load(os.path.join(file_dir, 'PSDM_RightParotid', file_name))[:, :, np.newaxis]
        PSDM_SpinalCord = np.load(os.path.join(file_dir, 'PSDM_SpinalCord', file_name))[:, :, np.newaxis]

        PTVs_mask = 70.0 / 70. * Mask_PTV70 + 63.0 / 70. * Mask_PTV63 + 56.0 / 70. * Mask_PTV56

        data_all = np.concatenate([ct, dose, PTVs_mask, Mask_Brainstem, Mask_Esophagus, Mask_Larynx, Mask_LeftParotid, Mask_Mandible,
                         Mask_possible_dose_mask, Mask_RightParotid, Mask_SpinalCord, PSDM_Brainstem, PSDM_Esophagus, PSDM_Larynx,
                         PSDM_LeftParotid, PSDM_Mandible, PSDM_possible_dose_mask, PSDM_PTV56, PSDM_PTV63, PSDM_PTV70,
                         PSDM_RightParotid, PSDM_SpinalCord], axis=-1)

        data_all = self.transforms(image=data_all)['image']

        ct = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        dis = torch.from_numpy(data_all[:, :, 2:]).permute(2, 0, 1)
        return ct, dis, dose

    def __len__(self):
        return self.len

class Dataset_PSDM_val(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'ct')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'ct')))
        self.len = len(self.file_name_list)

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct = np.load(os.path.join(file_dir, 'ct', file_name))[:, :, np.newaxis]
        dose = np.load(os.path.join(file_dir, 'dose', file_name))[:, :, np.newaxis]

        Mask_Brainstem = np.load(os.path.join(file_dir, 'Mask_Brainstem', file_name))[:, :, np.newaxis]
        Mask_Esophagus = np.load(os.path.join(file_dir, 'Mask_Esophagus', file_name))[:, :, np.newaxis]
        Mask_Larynx = np.load(os.path.join(file_dir, 'Mask_Larynx', file_name))[:, :, np.newaxis]
        Mask_LeftParotid = np.load(os.path.join(file_dir, 'Mask_LeftParotid', file_name))[:, :, np.newaxis]
        Mask_Mandible = np.load(os.path.join(file_dir, 'Mask_Mandible', file_name))[:, :, np.newaxis]
        Mask_possible_dose_mask = np.load(os.path.join(file_dir, 'Mask_possible_dose_mask', file_name))[:, :, np.newaxis]
        Mask_PTV56 = np.load(os.path.join(file_dir, 'Mask_PTV56', file_name))[:, :, np.newaxis]
        Mask_PTV63 = np.load(os.path.join(file_dir, 'Mask_PTV63', file_name))[:, :, np.newaxis]
        Mask_PTV70 = np.load(os.path.join(file_dir, 'Mask_PTV70', file_name))[:, :, np.newaxis]
        Mask_RightParotid = np.load(os.path.join(file_dir, 'Mask_RightParotid', file_name))[:, :, np.newaxis]
        Mask_SpinalCord = np.load(os.path.join(file_dir, 'Mask_SpinalCord', file_name))[:, :, np.newaxis]

        PSDM_Brainstem = np.load(os.path.join(file_dir, 'PSDM_Brainstem', file_name))[:, :, np.newaxis]
        PSDM_Esophagus = np.load(os.path.join(file_dir, 'PSDM_Esophagus', file_name))[:, :, np.newaxis]
        PSDM_Larynx = np.load(os.path.join(file_dir, 'PSDM_Larynx', file_name))[:, :, np.newaxis]
        PSDM_LeftParotid = np.load(os.path.join(file_dir, 'PSDM_LeftParotid', file_name))[:, :, np.newaxis]
        PSDM_Mandible = np.load(os.path.join(file_dir, 'PSDM_Mandible', file_name))[:, :, np.newaxis]
        PSDM_possible_dose_mask = np.load(os.path.join(file_dir, 'PSDM_possible_dose_mask', file_name))[:, :, np.newaxis]
        PSDM_PTV56 = np.load(os.path.join(file_dir, 'PSDM_PTV56', file_name))[:, :, np.newaxis]
        PSDM_PTV63 = np.load(os.path.join(file_dir, 'PSDM_PTV63', file_name))[:, :, np.newaxis]
        PSDM_PTV70 = np.load(os.path.join(file_dir, 'PSDM_PTV70', file_name))[:, :, np.newaxis]
        PSDM_RightParotid = np.load(os.path.join(file_dir, 'PSDM_RightParotid', file_name))[:, :, np.newaxis]
        PSDM_SpinalCord = np.load(os.path.join(file_dir, 'PSDM_SpinalCord', file_name))[:, :, np.newaxis]

        PTVs_mask = 70.0 / 70. * Mask_PTV70 + 63.0 / 70. * Mask_PTV63 + 56.0 / 70. * Mask_PTV56

        data_all = np.concatenate([ct, dose, PTVs_mask, Mask_Brainstem, Mask_Esophagus, Mask_Larynx, Mask_LeftParotid, Mask_Mandible,
                         Mask_possible_dose_mask, Mask_RightParotid, Mask_SpinalCord, PSDM_Brainstem, PSDM_Esophagus, PSDM_Larynx,
                         PSDM_LeftParotid, PSDM_Mandible, PSDM_possible_dose_mask, PSDM_PTV56, PSDM_PTV63, PSDM_PTV70,
                         PSDM_RightParotid, PSDM_SpinalCord], axis=-1)

        ct = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        dis = torch.from_numpy(data_all[:, :, 2:]).permute(2, 0, 1)
        return ct, dis, dose

    def __len__(self):
        return self.len