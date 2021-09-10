from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os
from glob import glob
import torch
import numpy as np
from config import config

def get_Y(img, y_only=True):
    '''
    from CutBlur's official code.
    Yoo, Jaejun, Namhyuk Ahn, and Kyung-Ah Sohn.
    "Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
    '''

    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.

    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
            [24.966, 112.0, -18.214]]
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def compare(gt, sr):
    psnr = compare_psnr(gt, sr)
    ssim = compare_ssim(gt, sr)

    return psnr, ssim


def remove_boundary(img, n_pixels):
    return img[n_pixels: -n_pixels, n_pixels: -n_pixels]



