from models import utils as mutils
import torch
import torch.nn.functional as F
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
from utils import fft2, ifft2, fft2_m, ifft2_m, clear, clear_color, normalize, normalize_np, down_up_right, \
    down_right, up_right
# from torch_radon import Radon
from utils import show_samples, show_samples_gray, down_up, root_sum_of_squares
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_pc_denoiser(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                    n_steps=1, probability_flow=False, continuous=False,
                    denoise=True, eps=1e-5):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_update_fn(update_fn):
        def inpaint_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return inpaint_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_inpainter(model, data):
        with torch.no_grad():
            x = data
            timesteps = torch.linspace(FR_T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = corrector_update_fn(model, data, x, t)
                x, x_mean = predictor_update_fn(model, data, x, t)

            return inverse_scaler(x_mean if denoise else x)

    return pc_inpainter


def get_pc_denoiser_lf_img(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                           n_steps=1, probability_flow=False, continuous=False,
                           denoise=True, eps=1e-5, lamb=1.0, lamb_schedule='const',
                           measurement_noise=False, save_progress=False, save_root=None):
    """
    lamb: weight that is applied to the conditional low frequency image
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def data_fidelity(x, x_mean, y, vec_t, lamb=0.01):
        if measurement_noise:
            y_mean, std = sde.marginal_prob(y, vec_t)
            y = y_mean + torch.randn_like(x) * std[:, None, None, None]
        else:
            y_mean = y

        x = (1. - lamb) * x + lamb * y
        x_mean = (1. - lamb) * x_mean + lamb * y_mean
        return x, x_mean

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, y, i):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                if lamb_schedule == 'linear':
                    cur_lamb = lamb - i / sde.N * lamb
                elif lamb_schedule == 'const':
                    cur_lamb = lamb
                x, x_mean = data_fidelity(x, x_mean, y, vec_t, lamb=cur_lamb)
                return x, x_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data, y):
        with torch.no_grad():
            x = data
            timesteps = torch.linspace(FR_T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = corrector_update_fn(model, x, t, y, i)
                x, x_mean = predictor_update_fn(model, x, t, y, i)

            return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser


def get_pc_denoiser_lf_img_SR(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                              n_steps=1, probability_flow=False, continuous=False,
                              denoise=True, eps=1e-5,
                              lamb_SR=1.0, scale_factor=2,
                              lamb_LF=0.005, lamb_schedule='const'):
    """
    lamb: weight that is applied to the conditional low frequency image
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def data_fidelity_SR(x, x_mean, y, vec_t, lamb=1.0):
        # y: (Low frequency part) of the LR + noisy image
        y_mean, std = sde.marginal_prob(y, vec_t)
        y = y_mean + torch.randn_like(x) * std[:, None, None, None]
        x = x + lamb * y - lamb * down_up_right(x, scale_factor=scale_factor)
        x_mean = x_mean + lamb * y_mean - lamb * down_up_right(x_mean, scale_factor=scale_factor)
        return x, x_mean

    def data_fidelity_lf_img(x, x_mean, y, vec_t, lamb=0.01):
        # y: (Low frequency part) of the LR + noisy image
        y_mean, std = sde.marginal_prob(y, vec_t)
        y = y_mean + torch.randn_like(x) * std[:, None, None, None]
        x = (1. - lamb) * x + lamb * y
        x_mean = (1. - lamb) * x_mean + lamb * y_mean
        return x, x_mean

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, y_LR, y_LRLF, i):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                if lamb_schedule == 'linear':
                    cur_lamb_LF = lamb_LF - i / sde.N * lamb_LF
                elif lamb_schedule == 'const':
                    cur_lamb_LF = lamb_LF
                x, x_mean = data_fidelity_SR(x, x_mean, y_LR, vec_t, lamb=lamb_SR)
                x, x_mean = data_fidelity_lf_img(x, x_mean, y_LRLF, vec_t, lamb=cur_lamb_LF)
                return x, x_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data, y_LR, y_LRLF):
        with torch.no_grad():
            FR_T_vec = torch.ones(data.shape[0], device=data.device) * FR_T
            mean, std = sde.marginal_prob(data, FR_T_vec)
            x = data + torch.randn_like(data) * std
            timesteps = torch.linspace(FR_T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = corrector_update_fn(model, x, t, y_LR, y_LRLF, i)
                x, x_mean = predictor_update_fn(model, x, t, y_LR, y_LRLF, i)

            return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser


def get_pc_denoiser_SR(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                       n_steps=1, probability_flow=False, continuous=False,
                       denoise=True, eps=1e-5,
                       lamb_SR=1.0, scale_factor=2, save_progress=False, save_root=None):
    """
    lamb: weight that is applied to the conditional low frequency image
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def data_fidelity_SR(x, x_mean, y, vec_t, lamb=1.0):
        y_mean, std = sde.marginal_prob(y, vec_t)
        y = y_mean + torch.randn_like(x) * std[:, None, None, None]
        y = down_up_right(y, scale_factor=scale_factor)
        x = x + lamb * y - lamb * down_up_right(x, scale_factor=scale_factor)
        x_mean = x_mean + lamb * y_mean - lamb * down_up_right(x_mean, scale_factor=scale_factor)
        return x, x_mean

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, y_LR):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                x, x_mean = data_fidelity_SR(x, x_mean, y_LR, vec_t, lamb=lamb_SR)
                return x, x_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data, y):
        with torch.no_grad():
            FR_T_vec = torch.ones(data.shape[0], device=data.device) * FR_T
            mean, std = sde.marginal_prob(data, FR_T_vec)
            x = data + torch.randn_like(data) * std
            timesteps = torch.linspace(FR_T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = corrector_update_fn(model, x, t, y)
                x, x_mean = predictor_update_fn(model, x, t, y)
                if save_progress:
                    if i % 100 == 0:
                        plt.imsave(save_root / 'progress' / f'x_{i}.png', clear(x_mean), cmap='gray')

            return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser


def get_pc_denoiser_MCG(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                        n_steps=1, probability_flow=False, continuous=False,
                        denoise=True, eps=1e-5, lamb=1.0, lamb_schedule='const', conv=None,
                        save_progress=False, save_root=None):
    """
    lamb: weight that is applied to the conditional low frequency image
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, i):
            x = x.requires_grad_()
            if lamb_schedule == 'linear':
                cur_lamb = lamb - i / sde.N * lamb
            elif lamb_schedule == 'const':
                cur_lamb = lamb

            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # n2s mapping
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt ** 2) * score

            # MCG
            norm = torch.linalg.norm(conv(hatx0) - x_next)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0] * cur_lamb

            x_next = x_next - norm_grad
            x_next_mean = x_next_mean - norm_grad

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'hatx0_{i}.png', clear(hatx0), cmap='gray')
                plt.imsave(save_root / 'recon' / 'progress' / f'x_{i}.png', clear(x_next_mean), cmap='gray')

            return x_next, x_next_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data):
        x = data
        timesteps = torch.linspace(FR_T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x, x_mean = corrector_update_fn(model, x, t, i)
            x, x_mean = predictor_update_fn(model, x, t, i)

        return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser


def get_pc_denoiser_MCG_lf(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                           n_steps=1, probability_flow=False, continuous=False,
                           denoise=True, eps=1e-5, lamb=1.0, lamb_schedule='const',
                           save_progress=False, save_root=None):
    """
    lamb: weight that is applied to the conditional low frequency image
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, i):
            x = x.requires_grad_()
            if lamb_schedule == 'linear':
                cur_lamb = lamb - i / sde.N * lamb
            elif lamb_schedule == 'const':
                cur_lamb = lamb

            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # n2s mapping
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt ** 2) * score

            # MCG
            norm = torch.linalg.norm(hatx0 - x_next)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0] * cur_lamb

            x_next = x_next - norm_grad
            x_next_mean = x_next_mean - norm_grad

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'hatx0_{i}.png', clear(hatx0), cmap='gray')
                plt.imsave(save_root / 'recon' / 'progress' / f'x_{i}.png', clear(x_next_mean), cmap='gray')

            return x_next, x_next_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data):
        x = data
        timesteps = torch.linspace(FR_T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x, x_mean = corrector_update_fn(model, x, t, i)
            x, x_mean = predictor_update_fn(model, x, t, i)

        return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser



def get_pc_denoiser_deblur_MCG(sde, predictor, corrector, inverse_scaler, snr, FR_T=0.03,
                               n_steps=1, probability_flow=False, continuous=False,
                               denoise=True, eps=1e-5, lamb=1.0, lamb_schedule='const',
                               save_progress=False, save_root=None, conv=None):
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def get_update_fn(update_fn):
        def denoise_update_fn(model, x, t, i, y=None):
            x = x.requires_grad_()
            if lamb_schedule == 'linear':
                cur_lamb = lamb - i / sde.N * lamb
            elif lamb_schedule == 'const':
                cur_lamb = lamb

            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            # n2s mapping
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt ** 2) * score
            hatx0 = torch.clip(hatx0, 0.0, 1.0)

            # DPS:forward model with blur conv
            norm = torch.linalg.norm(conv(hatx0) - y)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0] * cur_lamb

            x_next = x_next - norm_grad
            x_next_mean = x_next_mean - norm_grad

            x_next = x_next.detach()
            x_next_mean = x_next_mean.detach()

            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'hatx0_{i}.png', clear(hatx0), cmap='gray')
                plt.imsave(save_root / 'recon' / 'progress' / f'x_{i}.png', clear(x_next_mean), cmap='gray')

            return x_next, x_next_mean

        return denoise_update_fn

    predictor_update_fn = get_update_fn(predictor_update_fn)
    corrector_update_fn = get_update_fn(corrector_update_fn)

    def pc_denoiser(model, data):
        x = data
        timesteps = torch.linspace(FR_T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x, x_mean = corrector_update_fn(model, x, t, i, y=data)
            x, x_mean = predictor_update_fn(model, x, t, i, y=data)

        return inverse_scaler(x_mean if denoise else x)

    return pc_denoiser