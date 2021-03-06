from skimage.measure import compare_psnr, compare_ssim, compare_mse
import cv2
import os
from glob import glob
import torch

from utils import crop_img, downsample, np_to_torch_tensor, torch_tensor_to_np
from config import config


def validate_img(model, scale_by, img_path, tag):
    lr = cv2.imread(img_path)

    h = lr.shape[0]
    w = lr.shape[1]
    lr_h = h // scale_by
    lr_w = w // scale_by
    hr_h = lr_h * scale_by
    hr_w = lr_w * scale_by

    lr = crop_img(lr, size=(hr_h, hr_w), random=False)
    lr, _ = downsample(lr, scale_by=scale_by)

    lr = np_to_torch_tensor(lr, norm_to=config['in_norm'])
    lr = lr.float().cuda()

    with torch.no_grad():
        sr = model(lr)

    sr = torch_tensor_to_np(sr)
    name = os.path.basename(img_path)[:-4]

    filename = os.path.join(config['path']['validation'], name + '_' + tag + '.png')
    cv2.imwrite(filename=filename, img=sr)


def forward(network, img_dir, out_dir, auto_convert=True):
    network = network.cuda()
    img_list = glob(os.path.join(img_dir, '*.png'))

    for img in img_list:
        _name = os.path.basename(img)
        img = cv2.imread(img)
        if not auto_convert:
            img = np_to_torch_tensor(img)

        with torch.no_grad():
            sr = network(img)

        if not auto_convert:
            sr = torch_tensor_to_np(sr)
        cv2.imwrite(os.path.join(out_dir, _name), sr)


def evaluate(sr, gt):
    if not gt.shape == sr.shape:
        h, w, _ = sr.shape
        gt = gt[:h, :w]

    # MSE in BGR
    mse = compare_mse(gt, sr)

    # in Y channel of YCbCr
    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # PSNR
    psnr = compare_psnr(gt, sr)

    # SSIM
    ssim = compare_ssim(gt, sr)

    return mse, psnr, ssim
