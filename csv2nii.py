import pandas as pd
from Nii_utils import *
import matplotlib.pyplot as plt

rois = dict(
    oars=["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"],
    targets=["PTV56", "PTV63", "PTV70"],
    mask=['possible_dose_mask']
)
full_roi_list = sum(map(list, rois.values()), [])  # make a list of all rois
num_rois = len(full_roi_list)

origin = (0, 0, 0)
direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]

def load_file(file_path):
    """
    Load a file in one of the formats provided in the OpenKBP dataset
    """
    # if file_path.stem == "voxel_dimensions":
    #     return np.loadtxt(file_path)

    loaded_file_df = pd.read_csv(file_path, index_col=0)
    if loaded_file_df.isnull().values.any():  # Data is a mask
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:  # Data is a sparse matrix
        loaded_file = {"indices": loaded_file_df.index.values, "data": loaded_file_df.data.values}

    return loaded_file

def shape_data(data, key='image'):
    """Shapes into form that is amenable to tensorflow and other deep learning packages."""

    shaped_data = np.zeros((128, 128, 128, 1))

    if key == 'image':
        np.put(shaped_data, data["indices"], data["data"])
    else:
        np.put(shaped_data, data, int(1))

    return shaped_data.squeeze()

def csv2nii(provided_data_dir, save_dir):

    for dataset in os.listdir(provided_data_dir):    # train / val / test
        print(f'--------------------------------------{dataset}--------------------------------------')
        for patient in os.listdir(os.path.join(provided_data_dir, dataset)):
            print(patient)
            os.makedirs(os.path.join(save_dir, dataset, patient), exist_ok=True)
            patient_dir = os.path.join(provided_data_dir, dataset, patient)
            init_spacing = np.loadtxt(os.path.join(patient_dir, 'voxel_dimensions.csv'))
            spacing = init_spacing[[2, 0, 1]]

            # # mask
            all_mask_file = [i for i in os.listdir(patient_dir) if (i.endswith('.nii.gz') == False) and (i!= 'voxel_dimensions.csv')]
            for roi_idx, roi in enumerate(full_roi_list):
                if roi + '.csv' in all_mask_file:
                    save_path = os.path.join(save_dir, dataset, patient, roi + '.nii.gz')
                    data = load_file(os.path.join(patient_dir, roi + '.csv'))
                    mask = shape_data(data, key='mask')
                    mask = np.transpose(mask, (2, 0, 1))[::-1, :, :]
                    NiiDataWrite(save_path, mask, spacing, origin, direction, as_type=np.uint8)

            # ct + dose
            for file in ['ct.csv', 'dose.csv']:
                save_path = os.path.join(save_dir, dataset, patient, file[:-4] + '.nii.gz')
                data = load_file(os.path.join(patient_dir, file))
                mask = shape_data(data, key='image')
                mask = np.transpose(mask, (2, 0, 1))[::-1, :, :]
                NiiDataWrite(save_path, mask, spacing, origin, direction)

if __name__ == '__main__':
    data_dir = r'provided-data'
    save_dir = r'preprocessed_data'
    csv2nii(data_dir, save_dir)