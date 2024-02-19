import os
from dataset import Dataset_PSDM_train, Dataset_PSDM_val
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import argparse
import torch.utils.data as Data
import shutil
from guided_diffusion.unet import UNetModel_MS_Former
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
import torch.distributed as dist
from guided_diffusion.resample import create_named_schedule_sampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default="0", help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--T', type=int, default=1000, help='T')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method="env://")
device = torch.device("cuda")

train_bs = args.bs
val_bs = args.bs
lr_max = 0.0001
img_size = (128, 128)
all_epochs = args.epoch
data_root_train = 'preprocessed_data/NPY/train'
data_root_val = 'preprocessed_data/NPY/validation'
L2 = 0.0001
ture_bs = len(args.gpu.split(',')) * args.bs
val_bs = 2 * args.bs

save_name = 'T{}_bs{}_epoch{}'.format(args.T, ture_bs, args.epoch)

if dist.get_rank() == 0:
    if os.path.exists(os.path.join('trained_models', save_name)):
        shutil.rmtree(os.path.join('trained_models', save_name))
    os.makedirs(os.path.join('trained_models', save_name), exist_ok=True)
    print(save_name)

    train_writer = SummaryWriter(os.path.join('trained_models', save_name, 'log/train'), flush_secs=2)

train_data = Dataset_PSDM_train(data_root=data_root_train)
train_samper = Data.distributed.DistributedSampler(train_data)
val_data = Dataset_PSDM_val(data_root=data_root_val)
val_samper = Data.distributed.DistributedSampler(val_data)
train_dataloader = DataLoader(dataset=train_data, batch_size=train_bs, sampler=train_samper, shuffle=False, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=val_bs, sampler=val_samper, shuffle=False, num_workers=4, pin_memory=True)

if dist.get_rank() == 0:
    print('train_lenth: %i   val_lenth: %i' % (train_data.len, val_data.len))

dis_channels = 20

model = UNetModel_MS_Former(image_size=img_size, in_channels=1, ct_channels=1, dis_channels=dis_channels,
                       model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32),
                       dropout=0,
                       channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
                       use_checkpoint=False,
                       use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
                       use_scale_shift_norm=True,
                       resblock_updown=False, use_new_attention_order=False)

diffusion = SpacedDiffusion(use_timesteps=space_timesteps(args.T, [args.T]),
                            betas=gd.get_named_beta_schedule("linear", args.T),
                            model_mean_type=(gd.ModelMeanType.EPSILON),
                            model_var_type=(gd.ModelVarType.FIXED_LARGE),
                            loss_type=gd.LossType.MSE, rescale_timesteps=False)

diffusion_test = SpacedDiffusion(use_timesteps=space_timesteps(args.T, 'ddim4'),
                                betas=gd.get_named_beta_schedule("linear", args.T),
                                model_mean_type=(gd.ModelMeanType.EPSILON),
                                model_var_type=(gd.ModelVarType.FIXED_LARGE),
                                loss_type=gd.LossType.MSE, rescale_timesteps=False)

model.cuda()
model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,
            bucket_cap_mb=128, find_unused_parameters=False)

schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((7 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
best_MAE = 1000

for epoch in range(all_epochs):
    lr = optimizer.param_groups[0]['lr']
    model.train()
    train_epoch_loss = []
    for i, (ct, dis, rtdose) in enumerate(train_dataloader):
        ct, dis, rtdose = ct.cuda().float(), dis.cuda().float(), rtdose.cuda().float()

        optimizer.zero_grad()
        t, weights = schedule_sampler.sample(rtdose.shape[0], rtdose.device)
        losses = diffusion.training_losses(model=model, x_start=rtdose, t=t, model_kwargs={'ct': ct, 'dis': dis}, noise=None)
        loss = (losses["loss"] * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / dist.get_world_size()

        train_epoch_loss.append(loss.item())
        if dist.get_rank() == 0:
            print('[%d/%d, %d/%d] train_loss: %.3f' %
                  (epoch + 1, all_epochs, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()

    if dist.get_rank() == 0:
        train_epoch_loss = np.mean(train_epoch_loss)
        train_writer.add_scalar('lr', lr, epoch + 1)
        train_writer.add_scalar('train_loss', train_epoch_loss, epoch + 1)

    if (epoch == 0) or (((epoch + 1) % 10) == 0):
        model.eval()
        val_epoch_MAE = []
        image_CT = []
        ture_rtdose = []
        pred_rtdose = []
        with torch.no_grad():
            for i, (ct, dis, rtdose) in enumerate(val_dataloader):
                ct, dis, rtdose = ct.cuda().float(), dis.cuda().float(), rtdose.cuda().float()

                pred = diffusion_test.ddim_sample_loop(
                    model=model, shape=(ct.size(0), 1, img_size[0], img_size[1]), noise=None, clip_denoised=True,
                    denoised_fn=None, cond_fn=None, model_kwargs={'ct': ct, 'dis': dis}, device=None, progress=False, eta=0.0)

                rtdose = (rtdose + 1) * 40
                pred = (pred + 1) * 40
                body_mask = dis[:, 6: 7]
                MAE = (torch.abs(rtdose - pred) * body_mask).sum() / body_mask.sum()
                dist.all_reduce(MAE, op=torch.distributed.ReduceOp.SUM)
                MAE = MAE / dist.get_world_size()

                val_epoch_MAE.append(MAE.item())
                if i in [0, 2, 4, 8]:
                    image_CT.append(ct[0:1, :, :, :].cpu())
                    ture_rtdose.append(rtdose[0:1, :, :, :].cpu())
                    pred_rtdose.append(pred[0:1, :, :, :].cpu())

        if dist.get_rank() == 0:
            val_epoch_MAE = np.mean(val_epoch_MAE)
            train_writer.add_scalar('val_MAE', val_epoch_MAE, epoch + 1)

            image_CT = torch.cat(image_CT, dim=0)
            image_CT = make_grid(image_CT, 2, normalize=True)
            train_writer.add_image('image_CT', image_CT, epoch + 1)
            ture_rtdose = torch.cat(ture_rtdose, dim=0)
            ture_rtdose = make_grid(ture_rtdose, 2, normalize=True)
            train_writer.add_image('ture_rtdose', ture_rtdose, epoch + 1)
            pred_rtdose = torch.cat(pred_rtdose, dim=0)
            pred_rtdose = make_grid(pred_rtdose, 2, normalize=True)
            train_writer.add_image('pred_rtdose', pred_rtdose, epoch + 1)

            torch.save(model.state_dict(),
                       os.path.join('trained_models', save_name, 'model_epoch' + str(epoch + 1) + '.pth'))
            if val_epoch_MAE < best_MAE:
                best_MAE = val_epoch_MAE
                torch.save(model.state_dict(),
                           os.path.join('trained_models', save_name, 'model_best_mae.pth'))

if dist.get_rank() == 0:
    train_writer.close()
    print('saved_model_name:', save_name)