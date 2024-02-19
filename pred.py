import os
import numpy as np
import torch
import argparse
import shutil
from Nii_utils import NiiDataRead, NiiDataWrite
from guided_diffusion.unet import UNetModel_MS_Former
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from evaluate_openKBP import get_Dose_score_and_DVH_score

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--model_path', type=str, help='trained model path')
parser.add_argument('--TTA', type=int, default=0, help='0/1')
parser.add_argument('--bs', type=int, default=128, help='batchsize')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--ddim', type=str, default='8', help='ddim')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_dir = 'preprocessed_data/test-pats_preprocess'
img_size = (128, 128)

dis_channels = 20

new_dir = 'Results/ddim{}_TTA{}'.format(args.ddim, args.TTA)

if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
print(new_dir)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)

if args.TTA:
    TTA_num = 4
else:
    TTA_num = 1

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim{}'.format(args.ddim)),
                            betas=gd.get_named_beta_schedule("linear", args.T),
                            model_mean_type=(gd.ModelMeanType.EPSILON),
                            model_var_type=(gd.ModelVarType.FIXED_LARGE),
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)

net = UNetModel_MS_Former(image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
                       model_channels=96, out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32),
                       dropout=0,
                       channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)
net.cuda()
checkpoint = torch.load(args.model_path)
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
net.eval()

with torch.no_grad():
    for i, ID in enumerate(os.listdir(os.path.join(data_dir))):
        print('{} {}'.format(i, ID))
        CT_original, spacing, origin, direction = NiiDataRead(os.path.join(data_dir, ID, 'ct.nii.gz'))

        Mask_Brainstem, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Brainstem.nii.gz'))
        Mask_Esophagus, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Esophagus.nii.gz'))
        Mask_Larynx, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Larynx.nii.gz'))
        Mask_LeftParotid, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_LeftParotid.nii.gz'))
        Mask_Mandible, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_Mandible.nii.gz'))
        Mask_possible_dose_mask, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_possible_dose_mask.nii.gz'))
        Mask_PTV56, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_PTV56.nii.gz'))
        Mask_PTV63, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_PTV63.nii.gz'))
        Mask_PTV70, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_PTV70.nii.gz'))
        Mask_RightParotid, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_RightParotid.nii.gz'))
        Mask_SpinalCord, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'Mask_SpinalCord.nii.gz'))

        PSDM_Brainstem, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Brainstem.nii.gz'))
        PSDM_Esophagus, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Esophagus.nii.gz'))
        PSDM_Larynx, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Larynx.nii.gz'))
        PSDM_LeftParotid, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_LeftParotid.nii.gz'))
        PSDM_Mandible, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_Mandible.nii.gz'))
        PSDM_possible_dose_mask, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_possible_dose_mask.nii.gz'))
        PSDM_PTV56, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_PTV56.nii.gz'))
        PSDM_PTV63, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_PTV63.nii.gz'))
        PSDM_PTV70, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_PTV70.nii.gz'))
        PSDM_RightParotid, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_RightParotid.nii.gz'))
        PSDM_SpinalCord, _, _, _ = NiiDataRead(os.path.join(data_dir, ID, 'PSDM_SpinalCord.nii.gz'))

        PTVs_mask = 70.0 / 70. * Mask_PTV70 + 63.0 / 70. * Mask_PTV63 + 56.0 / 70. * Mask_PTV56

        CT = np.clip(CT_original, 0, 2500)
        CT = CT / 1250 - 1

        original_shape = CT.shape
        pred_rtdose = np.zeros(original_shape)

        n_num = original_shape[0] // args.bs
        n_num = n_num + 0 if original_shape[0] % args.bs == 0 else n_num + 1
        for n in range(n_num):
            if n == n_num - 1:
                CT_one = CT[n * args.bs:, :, :]
                dis_one = np.concatenate((PTVs_mask[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Brainstem[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Esophagus[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Larynx[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_LeftParotid[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_Mandible[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_possible_dose_mask[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_RightParotid[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          Mask_SpinalCord[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Brainstem[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Esophagus[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Larynx[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_LeftParotid[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_Mandible[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_possible_dose_mask[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_PTV56[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_PTV63[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_PTV70[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_RightParotid[n * args.bs:, :, :][np.newaxis, :, :, :],
                                          PSDM_SpinalCord[n * args.bs:, :, :][np.newaxis, :, :, :]), axis=0)

            else:
                CT_one = CT[n * args.bs: (n + 1) * args.bs, :, :]
                dis_one = np.concatenate(
                    (PTVs_mask[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_Brainstem[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_Esophagus[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_Larynx[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_LeftParotid[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_Mandible[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_possible_dose_mask[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_RightParotid[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     Mask_SpinalCord[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_Brainstem[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_Esophagus[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_Larynx[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_LeftParotid[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_Mandible[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_possible_dose_mask[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_PTV56[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_PTV63[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_PTV70[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_RightParotid[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :],
                     PSDM_SpinalCord[n * args.bs: (n + 1) * args.bs, :, :][np.newaxis, :, :, :]), axis=0)
            CT_one_tensor = torch.from_numpy(CT_one).unsqueeze(1).float()
            dis_one_tensor = torch.from_numpy(dis_one).float().permute(1, 0, 2, 3)
            for TTA_i in range(TTA_num):
                if TTA_i == 0:
                    CT_one_tensor_TTA = CT_one_tensor.cuda()
                    dis_one_tensor_TTA = dis_one_tensor.cuda()
                elif TTA_i == 1:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[2]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[2]).cuda()
                elif TTA_i == 2:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[3]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[3]).cuda()
                elif TTA_i == 3:
                    CT_one_tensor_TTA = torch.flip(CT_one_tensor, dims=[2, 3]).cuda()
                    dis_one_tensor_TTA = torch.flip(dis_one_tensor, dims=[2, 3]).cuda()
                noise = None
                pred_rtdose_one = diffusion.ddim_sample_loop(net, (
                CT_one_tensor_TTA.size(0), 1, img_size[0], img_size[1]),
                                                             model_kwargs={'ct': CT_one_tensor_TTA,
                                                                           'dis': dis_one_tensor_TTA},
                                                             noise=noise, clip_denoised=True, eta=0.0,
                                                             progress=True)
                if TTA_i == 1:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[2])
                elif TTA_i == 2:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[3])
                elif TTA_i == 3:
                    pred_rtdose_one = torch.flip(pred_rtdose_one, dims=[2, 3])
                if n == n_num - 1:
                    pred_rtdose[n * args.bs:, :, :] += pred_rtdose_one[:, 0, :, :].cpu().numpy()
                else:
                    pred_rtdose[n * args.bs: (n + 1) * args.bs, :, :] += pred_rtdose_one[:, 0, :, :].cpu().numpy()

        pred_rtdose = pred_rtdose / TTA_num
        pred_rtdose = (pred_rtdose + 1) * 40
        pred_rtdose = np.clip(pred_rtdose, 0, 80)
        pred_rtdose = pred_rtdose * Mask_possible_dose_mask

        os.makedirs(os.path.join(new_dir, 'predictions', ID))
        NiiDataWrite(os.path.join(new_dir, 'predictions', ID, 'dose.nii.gz'),
                     pred_rtdose, spacing, origin, direction)

Dose_score, Dose_std, DVH_score, DVH_std = get_Dose_score_and_DVH_score(prediction_dir=os.path.join(new_dir, 'predictions'),
                                                     gt_dir=r'preprocessed_data/test-pats')
print('Dose_score: {}'.format(Dose_score))
print('DVH_score: {}'.format(DVH_score))

with open(os.path.join(new_dir, 'score.txt'), 'w') as file:
    file.write('Dose_score: {} {}\nDVH_score: {} {}'.format(Dose_score, Dose_std, DVH_score, DVH_std))