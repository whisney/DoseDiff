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
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9999 train.py --gpu 0,1 --bs 16 --T 1000 --epoch 1600
```
### Model evaluation
```
CUDA_VISIBLE_DEVICES=0 python pred.py --gpu 0 --bs 64 --model_path trained_models/T1000_bs32_epoch1600/model_best_mae.pth --TTA 1 --T 1000 --ddim 8
```

## Citation
if you find this repository useful in your research, please consider citing:
```
@ARTICLE{10486983,
  author={Zhang, Yiwen and Li, Chuanpu and Zhong, Liming and Chen, Zeli and Yang, Wei and Wang, Xuetao},
  journal={IEEE Transactions on Medical Imaging}, 
  title={DoseDiff: Distance-Aware Diffusion Model for Dose Prediction in Radiotherapy}, 
  year={2024},
  volume={43},
  number={10},
  pages={3621-3633},
  keywords={Computed tomography;Predictive models;Planning;Biomedical imaging;Training;Radiation therapy;Noise reduction;Deep learning;diffusion model;dose prediction;radiotherapy;signed distance map},
  doi={10.1109/TMI.2024.3383423}}
```
