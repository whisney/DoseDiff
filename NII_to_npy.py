import os
from Nii_utils import NiiDataRead
import numpy as np

original_dir = r'preprocessed_data'
save_dir = r'preprocessed_data/NPY'

name_list = ['ct',  'dose', 'Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx', 'Mask_LeftParotid', 'Mask_Mandible',
             'Mask_possible_dose_mask', 'Mask_PTV56', 'Mask_PTV63', 'Mask_PTV70', 'Mask_RightParotid', 'Mask_SpinalCord',
             'PSDM_Brainstem', 'PSDM_Esophagus', 'PSDM_Larynx', 'PSDM_LeftParotid', 'PSDM_Mandible', 'PSDM_possible_dose_mask',
             'PSDM_PTV56', 'PSDM_PTV63', 'PSDM_PTV70', 'PSDM_RightParotid', 'PSDM_SpinalCord']

for phase in ['train', 'validation', 'test']:
    for name in name_list:
        os.makedirs(os.path.join(save_dir, phase, name), exist_ok=True)

for phase in ['train', 'validation', 'test']:
    for ID in os.listdir(os.path.join(original_dir, '{}-pats_preprocess'.format(phase))):
        print(ID)
        mask, _, _, _ = NiiDataRead(os.path.join(original_dir, '{}-pats_preprocess'.format(phase), ID, 'Mask_possible_dose_mask.nii.gz'))
        for name in name_list:
            print(name)
            img, _, _, _ = NiiDataRead(os.path.join(original_dir, '{}-pats_preprocess'.format(phase), ID, '{}.nii.gz'.format(name)))
            if name == 'ct':
                img = np.clip(img, 0, 2500)
                img = img / 1250 - 1
            elif name == 'dose':
                img = np.clip(img, 0, 80)
                img = img / 40 - 1

            for i in range(img.shape[2]):
                mask_one = mask[:, :, i]
                if mask_one.max() > 0:
                    print(i)
                    img_one = img[:, :, i]
                    np.save(os.path.join(save_dir, phase, name, '{}_{}.npy'.format(ID, i)), img_one)