import torch
import torch.nn.functional as F
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from fastmri_utils import fft2c_new, ifft2c_new
from statistics import mean, stdev
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sporco.metric import gmsd, mse
from scipy.ndimage import gaussian_laplace
from sigpy.mri import poisson
import sigpy as sp
import functools
from resize_right import resize
from torch import nn
import math
import random


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def restore_checkpoint(ckpt_dir, state, device, skip_sigma=False):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.error(f"No checkpoint found at {ckpt_dir}. "
                      f"Returned the same state as input")
        FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
        return state
    else:
        # import ipdb; ipdb.set_trace()
        loaded_state = torch.load(ckpt_dir, map_location=device)
        # state['optimizer'].load_state_dict(loaded_state['optimizer'])
        loaded_model_state = loaded_state['model']
        if skip_sigma:
            try:
                loaded_model_state.pop('module.sigmas')
            except:
                pass

        state['model'].load_state_dict(loaded_model_state, strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        print(f'loaded checkpoint dir from {ckpt_dir}')
        return state
    
    
def restore_checkpoint_ema(ckpt_dir, state, device, skip_sigma=False):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.error(f"No checkpoint found at {ckpt_dir}. "
                      f"Returned the same state as input")
        FileNotFoundError(f'No such checkpoint: {ckpt_dir} found!')
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        loaded_model_state = loaded_state['model']
        state['model'].load_state_dict(loaded_model_state)
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
"""
Helper functions for new types of inverse problems
"""


def fft2(x):
    """ FFT with shifting DC to the center of the image"""
    return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
    """ IFFT with shifting DC to the corner of the image prior to transform"""
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
    """ FFT for multi-coil """
    return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
    """ IFFT for multi-coil """
    return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def crop_center(img, cropx, cropy):
    c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def unnormalize(img, maxv, minv):
    img *= maxv
    img += minv
    return img


def normalize_np(img, verbose=False):
    """ Normalize img in arbitrary range to [0, 1] """
    minv = np.min(img)
    img -= minv
    maxv = np.max(img)
    img /= maxv
    if verbose:
        extra_params = {'maxv': maxv, 'minv': minv}
    else:
        extra_params = {}
    return img, extra_params


def unnormalize_np(img, extra_params):
    """ unormalize img in arbitrary range from [0, 1] to original range """
    img *= extra_params['maxv']
    img += extra_params['minv']
    return img


def pad_img(img, mult=64):
    """ pad given image to the multiples of "mult" """
    h, w = img.shape
    # h
    if h % mult == 0:
        padw_h = 0
    else:
        padw_h = (64 - h % 64) // 2
    # w
    if w % mult == 0:
        padw_w = 0
    else:
        padw_w = (64 - w % 64) // 2
    pad_img = np.pad(img, ((padw_h, padw_h), (padw_w, padw_w)), 'constant')
    # when too small (e.g. 64x64), error occurs. temporarily pad to 128x128
    if pad_img.shape[-1] < 256:
        marg = (256 - pad_img.shape[-1]) // 2
        pad_img = np.pad(img, ((marg, marg), (marg, marg)), 'constant')
    return pad_img


def cut_img(pad_img, orig_h, orig_w):
    """ cut img to return to the original size """
    h, w = pad_img.shape
    marg_h = (h - orig_h) // 2
    if marg_h == 0:
        pass
    else:
        pad_img = pad_img[marg_h:-(marg_h), :]

    marg_w = (w - orig_w) // 2
    if marg_w == 0:
        pass
    else:
        pad_img = pad_img[:, marg_w:-(marg_w)]
    return pad_img


def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    # ang_img = torch.angle(img)
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def image_grid(x, sz=32):
    size = sz
    channels = 3
    img = x.reshape(-1, size, size, channels)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
    return img


def show_samples(x, sz=32):
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x, sz)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def image_grid_gray(x, size=32):
    img = x.reshape(-1, size, size)
    w = int(np.sqrt(img.shape[0]))
    img = img.reshape((w, w, size, size)).transpose((0, 2, 1, 3)).reshape((w * size, w * size))
    return img


def show_samples_gray(x, size=32, save=False, save_fname=None):
    x = x.detach().cpu().numpy()
    img = image_grid_gray(x, size=size)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()
    if save:
        plt.imsave(save_fname, img, cmap='gray')


class lambda_schedule:
    def __init__(self, total=2000):
        self.total = total

    def get_current_lambda(self, i):
        pass


class lambda_schedule_linear(lambda_schedule):
    def __init__(self, start_lamb=1.0, end_lamb=0.0):
        super().__init__()
        self.start_lamb = start_lamb
        self.end_lamb = end_lamb

    def get_current_lambda(self, i):
        return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
    def __init__(self, lamb=1.0):
        super().__init__()
        self.lamb = lamb

    def get_current_lambda(self, i):
        return self.lamb


def clear(x):
    return x.detach().cpu().abs().squeeze().numpy()


def clear_nonorm(x):
    return x.detach().cpu().abs().squeeze().numpy()


def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def down_up(x, scale_factor=4):
    # performs down-sampling, then upsample to original resolution
    sz = x.shape[-1]
    x = F.interpolate(x, size=sz // scale_factor, mode='bicubic')
    return F.interpolate(x, size=sz, mode='bicubic')


def down_up_right(x, scale_factor=4):
    x = resize(x, scale_factors=1 / scale_factor)
    return resize(x, scale_factors=scale_factor)


def up_right(x, scale_factor=4):
    return resize(x, scale_factors=scale_factor)


def down_right(x, scale_factor=4):
    return resize(x, scale_factors=1 / scale_factor)


def get_mask(img, size, batch_size, type='gaussian2d', acc_factor=8, center_fraction=0.04, fix=False):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'gaussian2d':
        mask = torch.zeros_like(img)
        cov_factor = size * (1.5 / 128)
        mean = [size // 2, size // 2]
        cov = [[size * cov_factor, 0], [0, size * cov_factor]]
        if fix:
            samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
        else:
            for i in range(batch_size):
                # sample different masks for batch
                samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, int_samples[:, 0], int_samples[:, 1]] = 1
    elif type == 'uniformrandom2d':
        mask = torch.zeros_like(img)
        if fix:
            mask_vec = torch.zeros([1, size * size])
            samples = np.random.choice(size * size, int(Nsamp))
            mask_vec[:, samples] = 1
            mask_b = mask_vec.view(size, size)
            mask[:, ...] = mask_b
        else:
            for i in range(batch_size):
                # sample different masks for batch
                mask_vec = torch.zeros([1, size * size])
                samples = np.random.choice(size * size, int(Nsamp))
                mask_vec[:, samples] = 1
                mask_b = mask_vec.view(size, size)
                mask[i, ...] = mask_b
    elif type == 'gaussian1d':
        mask = torch.zeros_like(img)
        mean = size // 2
        std = size * (15.0 / 128)
        Nsamp_center = int(size * center_fraction)
        if fix:
            samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
            int_samples = samples.astype(int)
            int_samples = np.clip(int_samples, 0, size - 1)
            mask[..., int_samples] = 1
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                samples = np.random.normal(loc=mean, scale=std, size=int(Nsamp * 1.2))
                int_samples = samples.astype(int)
                int_samples = np.clip(int_samples, 0, size - 1)
                mask[i, :, :, int_samples] = 1
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from + Nsamp_center] = 1
    elif type == 'poisson':
        mask = poisson((size, size), accel=acc_factor).astype(np.float32)
        mask = torch.from_numpy(mask)
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask, t, th, w, wl


def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.

    Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 5D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 5D kspace tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], 2 * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
    """
    Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

    This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
    If the coil dimension is not divisible by 2, the function assumes that the input data is 'real' data,
    and thus pads the imaginary dimension as 0.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4
    s = tensor.shape
    if tensor.shape[1] == 1:
        imag_tensor = torch.zeros(s, device=tensor.device)
        tensor = torch.cat((tensor, imag_tensor), dim=1)
        s = tensor.shape
    tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
    return tensor


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def save_data(fname, arr):
    """ Save data as .npy and .png """
    np.save(fname + '.npy', arr)
    plt.imsave(fname + '.png', arr, cmap='gray')


def mean_std(vals: list):
    return mean(vals), stdev(vals)


def cal_metric(comp, label):
    LoG = functools.partial(gaussian_laplace, sigma=1.5)
    psnr_val = peak_signal_noise_ratio(comp, label)
    ssim_val = structural_similarity(comp, label)
    hfen_val = mse(LoG(comp), LoG(label))
    gmsd_val = gmsd(label, comp)
    return psnr_val, ssim_val, hfen_val, gmsd_val


def pad_array(x, div=64):
    h, w = x.shape
    h_padlen = (h // div + 1) * div
    w_padlen = (w // div + 1) * div

    h_pad = (h_padlen - h) // 2
    w_pad = (w_padlen - w) // 2
    x = np.pad(x, ((h_pad, h_pad), (w_pad, w_pad)))
    return x


def estimate_FR_T(noise_scale):
    """
    For a given noise_scale (estimated), we would like to find the optimal FR_T
    that will be used for reverse diffusion.
    We define our estimation with a linear fuction f, such that
    f(0.004) = 0.002  --> minimum
    f(0.016) = 0.05   --> maximum
    Note that the above values were found heuristically.
    TODO: maybe update this to a geometric function?

    f(x) = 4x - 0.014
    Args:
      noise_scale: the estimated noise scale from noise_estimate.py

    Returns: FR_T value that will be multiplied to form the number of iterations for reverse diffusion
    """
    FR_T = noise_scale * 4 - 0.014
    return truncate(FR_T, 3)


def inv_sigma(std, sigma_max, sigma_min):
    """
    \sigma(t) = \sigma_min (\sigma_max / \sigma_min)^t = std
    we wish to calculate
    t = \sigma^{-1}(std)
    """
    t = np.log(std / sigma_min) / np.log(sigma_max / sigma_min)
    return truncate(t, 3)


def normalize_mvue(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling


def unnormalize_mvue(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=0) / np.sqrt(
        np.sum(np.square(np.abs(s_maps)), axis=0))


def A_np(mvue, sens, mask):
    return mask * sp.fft(sens * mvue, axes=(-1, -2))


import h5py as h5


# Here I am reading one single image from  demoImage.hdf5 for testing demo code
def getTestingData(filename):
    print('Reading the data. Please wait...')
    # filename='demoImage.hdf5' #set the correct path here
    with h5.File(filename, 'r') as f:
        org, csm, mask = f['tstOrg'][:], f['tstCsm'][:], f['tstMask'][:]
    print('Successfully undersampled data!')
    return org, csm, mask


def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)

    with h5.File(path, 'r') as f:
        scan_node(f)


from scipy import signal


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import scipy
import time
class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        nTime = time.time()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(1, 1, self.kernel_size, stride=1, padding=0, bias=False)
        )

        self.weights_init()
        nTime = time.time() - nTime
        print("blurkernel : " , nTime)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        else:
            NotImplementedError(f"Blur type must be gaussian. Got {self.blur_type}")

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


if __name__ == '__main__':
    noise_scale = 20
    print(inv_sigma(noise_scale, 378, 0.01))
