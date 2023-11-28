import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from utils import restore_checkpoint_ema

# configs, checkpoints
from configs.ve import fastmri_knee_320_ncsnpp_continuous as configs

ckpt_num = 95
config_name = 'fastmri_knee_320_ncsnpp_continuous'
ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
config = configs.get_config()

model = mutils.create_model(config)
state = dict(model=model)
state = restore_checkpoint_ema(ckpt_filename, state, config.device, skip_sigma=True)