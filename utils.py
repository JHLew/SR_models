import torch
import torchvision
from torch.utils import data
import os
import cv2
import numpy as np
from glob import glob
from random import randint

to_tensor = torchvision.transforms.ToTensor()


# dataset for training
class training_dataset(data.Dataset):
    def __init__(self, dirs, crop_size=96, scale_by=4):
        self.crop_size = crop_size
        self.scale_by = scale_by
        self.img_list = []
        for d in dirs:
            self.img_list = self.img_list + glob(os.path.join(d, '*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        img = crop_img(img, size=(self.crop_size, self.crop_size))
        img = augmentation(img)
        lr_img, gt_img = downsample(img, scale_by=self.scale_by)
        lr_img = normalization(lr_img)
        gt_img = normalization(gt_img)
        lr_img = to_tensor(lr_img)
        gt_img = to_tensor(gt_img)


        return lr_img, gt_img, img_name


# dataset for evaluation
class evaluation_dataset(data.Dataset):
    def __init__(self, dirs, scale_by=4):
        self.scale_by = scale_by
        self.img_list = []
        for d in dirs:
            self.img_list = self.img_list + glob(os.path.join(d, '*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_file = self.img_list[index]
        img = cv2.imread(img_file)
        img_name = os.path.basename(img_file)

        h = img.shape[0]
        w = img.shape[1]
        lr_h = h // self.scale_by
        lr_w = w // self.scale_by
        hr_h = lr_h * self.scale_by
        hr_w = lr_w * self.scale_by

        img = crop_img(img, size=(hr_h, hr_w), random=False)
        lr_img, gt_img = downsample(img, scale_by=self.scale_by)

        lr_img = normalization(lr_img)
        gt_img = normalization(gt_img)

        lr_img = to_tensor(lr_img)
        gt_img = to_tensor(gt_img)

        return lr_img, gt_img, img_name


def getFiles(dir, dataList):
    if os.path.isdir(dir):
        temp_dataList = os.listdir(dir)
        for directory in temp_dataList:
            directory = os.path.join(dir, directory)
            getFiles(directory, dataList)
    elif os.path.isfile(dir):
        if dir.endswith('.png') or dir.endswith('.jpeg') or dir.endswith('.jpg'):
            dataList.append(dir)


# crop a part of image
def crop_img(image, size, random=True):
    if random:
        h = randint(0, image.shape[0] - size[0])
        w = randint(0, image.shape[1] - size[1])
    else:
        h = 0
        w = 0

    cropped_img = image[h: h + size[0], w: w + size[1]]

    return cropped_img


# data augmentation by flipping and rotating
def augmentation(image):
    # flipping
    flip_flag = randint(0, 1)
    if flip_flag == 1:
        image = cv2.flip(image, 1)

    # rotation
    rot = randint(0, 359)
    if rot < 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rot < 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rot < 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


# downsample by blur + bicubic / or / Area based interpolation
def downsample(image, scale_by, blur=False):
    if blur:
        lr_img = cv2.GaussianBlur(image, (7, 7), 1.6, borderType=cv2.BORDER_DEFAULT)
        lr_img = cv2.resize(lr_img, dsize=None, fx=scale_by, fy=scale_by, interpolation=cv2.INTER_CUBIC)
    else:
        lr_img = cv2.resize(image, dsize=None, fx=scale_by, fy=scale_by, interpolation=cv2.INTER_AREA)

    return lr_img, image


# normalization
def normalization(image, _from=(0, 255), _to=(-1, 1)):
    if _from == _to:
        return image

    if _from == (0, 255):
        if _to == (0, 1):
            return image / 255
        if _to == (-1, 1):
            return image / 127.5 - 1

    elif _from == (0, 1):
        if _to == (0, 255):
            return image * 255
        if _to == (-1, 1):
            return image * 2 - 1

    elif _from == (-1, 1):
        if _to == (0, 255):
            return (image + 1) * 127.5
        if _to == (0, 1):
            return (image + 1) / 2

    # else: out of range
    raise ValueError('wrong range input: normalization only suppoerts range of (0, 1), (-1, 1), and (0, 255)')


# computation of Gradient Penalty: for WGAB-GP
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# converting a numpy array to a torch tensor
def np_to_torch_tensor(x, norm_from=(0, 255), norm_to=(-1, 1)):
    to_tensor = torchvision.transforms.ToTensor()
    if x.ndim == 3:
        x = normalization(x, _from=norm_from, _to=norm_to)
        x = to_tensor(x)
        x = x.unsqueeze(0)
    else:
        raise ValueError('number of dimensions should be 3(H, W, C) - this is only for forwarding only a single image.')

    return x


# converting a torch tensor to numpy array
def torch_tensor_to_np(x, norm_from=(-1, 1), norm_to=(0, 255)):
    x = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
    x = np.clip(normalization(x, _from=norm_from, _to=norm_to), 0, 255).astype(np.uint8)[0]

    return x
