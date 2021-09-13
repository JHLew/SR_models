import torch
from torchvision.transforms.functional import to_tensor
from torch.utils import data
import os
from PIL import Image
import numpy as np
from glob import glob
from random import randint


# dataset for training
class dataset(data.Dataset):
    def __init__(self, dirs, patch_size=192, scale=4, is_train=True):
        self.crop_size = patch_size
        self.lr_size = patch_size // scale
        self.scale_by = scale
        self.is_train = is_train
        self.img_list = []
        for d in dirs:
            self.img_list = self.img_list + glob(os.path.join(d, '*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img_name = os.path.basename(img_path)

        if self.is_train:
            img, _ = crop_img(img, size=(self.crop_size, self.crop_size))
            img, _ = augmentation(img)
            lr_size = (self.lr_size, self.lr_size)
        else:
            w, h = img.size
            lr_size = (int(w // self.scale_by), int(h // self.scale_by))

        lr_img = img.resize(lr_size, resample=Image.BICUBIC)

        lr_img = to_tensor(lr_img)
        gt_img = to_tensor(img)

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
def crop_img(img, size, custom=None):
    width, height = size
    if custom is None:
        left = randint(0, img.size[0] - width)
        top = randint(0, img.size[1] - height)
    else:
        left, top = custom

    cropped_img = img.crop((left, top, left + width, top + height))

    return cropped_img, (left, top)


# data augmentation by flipping and rotating
def augmentation(img, custom=None, do_rot=True):
    if custom is None:
        flip_flag = randint(0, 1)
        rot = randint(0, 359)
    else:
        flip_flag, rot = custom
        if rot is None:
            do_rot = False

    # flipping
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # rotation
    if do_rot:
        if rot < 90:
            rot = 45
            img = img.rotate(90)
        elif rot < 180:
            rot = 135
            img = img.rotate(180)
        elif rot < 270:
            rot = 225
            img = img.rotate(270)
        else:
            rot = 315
    else:
        rot = None

    return img, (flip_flag, rot)

