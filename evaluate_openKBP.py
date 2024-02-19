import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

"""
These codes are modified from https://github.com/ababier/open-kbp
"""


def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif


def get_DVH_metrics(_dose, _mask, mode, spacing=None):
    output = {}

    if mode == 'target':
        _roi_dose = _dose[_mask > 0]
        # D1
        output['D1'] = np.percentile(_roi_dose, 99)
        # D95
        output['D95'] = np.percentile(_roi_dose, 5)
        # D99
        output['D99'] = np.percentile(_roi_dose, 1)

    elif mode == 'OAR':
        if spacing is None:
            raise Exception('calculate OAR metrics need spacing')

        _roi_dose = _dose[_mask > 0]
        _roi_size = len(_roi_dose)
        _voxel_size = np.prod(spacing)
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100 / _voxel_size))
        # D_0.1_cc
        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc / _roi_size * 100
        output['D_0.1_cc'] = np.percentile(_roi_dose, fractional_volume_to_evaluate)
        # Dmean
        output['mean'] = np.mean(_roi_dose)
    else:
        raise Exception('Unknown mode!')

    return output


def get_Dose_score_and_DVH_score(prediction_dir, gt_dir):
    list_dose_dif = []
    list_DVH_dif = []

    list_patient_ids = tqdm(os.listdir(prediction_dir))
    for patient_id in list_patient_ids:
        pred_nii = sitk.ReadImage(prediction_dir + '/' + patient_id + '/dose.nii.gz')
        pred = sitk.GetArrayFromImage(pred_nii)

        gt_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/dose.nii.gz')
        gt = sitk.GetArrayFromImage(gt_nii)

        # Dose dif
        possible_dose_mask_nii = sitk.ReadImage(gt_dir + '/' + patient_id + '/possible_dose_mask.nii.gz')
        possible_dose_mask = sitk.GetArrayFromImage(possible_dose_mask_nii)
        list_dose_dif.append(get_3D_Dose_dif(pred, gt, possible_dose_mask))

        # DVH dif
        for structure_name in ['Brainstem',
                               'SpinalCord',
                               'RightParotid',
                               'LeftParotid',
                               'Esophagus',
                               'Larynx',
                               'Mandible',

                               'PTV70',
                               'PTV63',
                               'PTV56']:
            structure_file = gt_dir + '/' + patient_id + '/' + structure_name + '.nii.gz'

            # If the structure has been delineated
            if os.path.exists(structure_file):
                structure_nii = sitk.ReadImage(structure_file, sitk.sitkUInt8)
                structure = sitk.GetArrayFromImage(structure_nii)

                spacing = structure_nii.GetSpacing()
                if structure_name.find('PTV') > -1:
                    mode = 'target'
                else:
                    mode = 'OAR'
                pred_DVH = get_DVH_metrics(pred, structure, mode=mode, spacing=spacing)
                gt_DVH = get_DVH_metrics(gt, structure, mode=mode, spacing=spacing)

                for metric in gt_DVH.keys():
                    list_DVH_dif.append(abs(gt_DVH[metric] - pred_DVH[metric]))

    return np.mean(list_dose_dif), np.std(list_dose_dif), np.mean(list_DVH_dif), np.std(list_DVH_dif)