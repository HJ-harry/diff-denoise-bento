# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on fastmri knee with VE SDE."""

from configs.default_lsun_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training

  # training (regression)
  training.mask_type = 'gaussian2d'
  training.acc_factor = [8, 15]

  # data
  data = config.data
  data.dataset = 'fastmri_knee'
  data.root = '/media/harry/tomo/fastmri'
  data.image_size = 128

  # model
  model = config.model
  model.name = 'unet'
  model.ema_rate = 0.999
  model.in_chans = 1
  model.out_chans = 1
  model.chans = 64
  model.num_pool_layers = 4
  model.use_residual = True

  return config
