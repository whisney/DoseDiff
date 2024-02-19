# DoseDiff
This is the repository for the paper "DoseDiff: Distance-aware Diffusion Model for Dose Prediction in Radiotherapy".

![DoseDiff](/pic/Figure1.jpg)

## Requirements
* python 3.6
* pytorch 1.10
* pydicom
* albumentations
* tensorboardX
* SimpleITK

## Training and evaluation on OpenKBP
### Data Preparation
Download [OpenKBP challenge repository](https://github.com/ababier/open-kbp), and copy the data folder "provided-data" into this repository. You can run data preprocess step with the following command:
```
cd DoseDiff-main

# Convert csv data into nii format
python csv2nii.py

# Generate distance map and missing ROI
python PSDM_OpenKBP.py

# Convert 3D into 2D for training
python NII_to_npy.py
```
### Model training
