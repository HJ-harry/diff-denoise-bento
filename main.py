from scipy.io import loadmat
from pathlib import Path
from models import utils as mutils
import sampling
import os
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      LangevinCorrectorCS)
from models import ncsnpp
from itertools import islice
from losses import get_optimizer
import datasets
import time
import controllable_generation
from utils import restore_checkpoint, fft2, ifft2, show_samples_gray, get_mask, clear, inv_sigma
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
from scipy.io import savemat
import matplotlib.pyplot as plt

###############################################
# Configurations
###############################################
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
config_name = 'fastmri_knee_320_ncsnpp_continuous'
sde = 'VESDE'
ckpt_num = 95
num_scales = 1000
from configs.ve import fastmri_knee_320_ncsnpp_continuous as configs
ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
config = configs.get_config()

# noise parameter
noise_type = 'gaussian'  # or 'rician'
std = 13.0
noise_std = std / 255.0
FR_T = inv_sigma(noise_std, sigma_max=config.model.sigma_max, sigma_min=config.model.sigma_min)
FR_T_scale = 0.1
N = int(num_scales * FR_T * FR_T_scale)
lamb = 0.03
lamb_schedule = 'linear'

if sde.lower() == 'vesde':
    config.model.num_scales = num_scales
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5

# Define and load model parameters
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
ema.copy_to(score_model.parameters())

load_root_label = Path(f'./samples/knee_320_val')
load_root_input = Path(f'./samples/knee_320_val_noise/{noise_type}/{std}')

subfolders = [f.path for f in os.scandir(load_root_label) if f.is_dir()]

save_root = Path(f'./results/retrospective/{noise_type}/{std}')
for f in subfolders:
    for type in ["input", "recon", "label"]:
        vol_name = f.split('/')[-1]
        save_dir_v = save_root / vol_name / type
        save_dir_v.mkdir(parents=True, exist_ok=True)

label_list = sorted(list(load_root_label.glob('*/*.npy')))
input_list = sorted(list(load_root_input.glob('*/*.npy')))

length = len(label_list)

mask = torch.zeros([1, 1, 320, 320])
mask[..., 81:240, 81:240] = 1
mask = mask.to(config.device)


for idx in tqdm(range(length)):
    fname = label_list[idx]
    flist = str(fname).split('/')
    vol = flist[-2]
    fname = flist[-1][:-4]

    img = torch.from_numpy(np.load(label_list[idx]).astype(np.float32))
    img_n = torch.from_numpy(np.load(input_list[idx]).astype(np.float32))

    img = img.view(1, 1, 320, 320).to(config.device)
    img_n = img_n.view(1, 1, 320, 320).to(config.device)

    k = fft2(img_n)
    uk = k * mask
    uimg = torch.abs(ifft2(uk))

    pc_sampler = controllable_generation.get_pc_denoiser_lf_img(
        sde,
        predictor, corrector,
        inverse_scaler,
        FR_T=FR_T * FR_T_scale,
        snr=0.16,
        n_steps=1,
        probability_flow=False,
        continuous=config.training.continuous,
        denoise=True,
        lamb=lamb,
        lamb_schedule=lamb_schedule,
        save_progress=True,
        save_root=save_root
    )
    # Reverse SDE
    x = pc_sampler(score_model, scaler(img_n), scaler(uimg))

    plt.imsave(str(save_root / f'{vol}' / 'label' / f'{fname}.png'), clear(img), cmap='gray')
    plt.imsave(str(save_root / f'{vol}' / 'input' / f'{fname}.png'), clear(img_n), cmap='gray')
    plt.imsave(str(save_root / f'{vol}' / 'recon' / f'{fname}.png'), clear(x), cmap='gray')