from scipy import ndimage
from Nii_utils import NiiDataWrite, NiiDataRead
import os
import numpy as np
import shutil

data_dir = r'preprocessed_data'

for pats in ['train-pats', 'validation-pats', 'test-pats']:
    save_dir = os.path.join(data_dir, '{}_preprocess'.format(pats))
    for ID in os.listdir(os.path.join(data_dir, pats)):
        os.makedirs(os.path.join(save_dir, ID), exist_ok=True)
        print(pats, ID)
        for name in ['PTV70', 'PTV63', 'PTV56', 'possible_dose_mask', 'Brainstem', 'SpinalCord', 'RightParotid',
                     'LeftParotid', 'Esophagus', 'Larynx', 'Mandible']:
            if os.path.exists(os.path.join(data_dir, pats, ID, '{}.nii.gz'.format(name))):
                shutil.copy(os.path.join(data_dir, pats, ID, '{}.nii.gz'.format(name)),
                            os.path.join(save_dir, ID, 'Mask_{}.nii.gz'.format(name)))
                mask, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, pats, ID, '{}.nii.gz'.format(name)), as_type=np.uint8)
                dis_map_p = ndimage.morphology.distance_transform_edt(mask, sampling=spacing)
                dis_map_n = ndimage.morphology.distance_transform_edt(1-mask, sampling=spacing)
                dis_map = (dis_map_p - dis_map_n) / 100
                NiiDataWrite(os.path.join(save_dir, ID, 'PSDM_{}.nii.gz'.format(name)), dis_map, spacing, origin, direction)
            else:
                NiiDataWrite(os.path.join(save_dir, ID, 'PSDM_{}.nii.gz'.format(name)),
                             np.zeros((128, 128, 128), np.uint8), spacing, origin, direction)
                NiiDataWrite(os.path.join(save_dir, ID, 'Mask_{}.nii.gz'.format(name)),
                             np.zeros((128, 128, 128), np.uint8), spacing, origin, direction)
        for name in ['ct', 'dose']:
            shutil.copy(os.path.join(data_dir, pats, ID, '{}.nii.gz'.format(name)),
                        os.path.join(save_dir, ID, '{}.nii.gz'.format(name)))