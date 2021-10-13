import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from tqdm import tqdm

from config import config as _config
from utils import dataset
from validation import validation
from models import EDSR as Generator
from Loss import Thresh_Error

# proj_directory = '/project'
# data_directory = '/dataset'


def train(config, epoch_from=0):
    threshold = 5
    threshold = threshold / 127.5
    penalty = 'over'

    print('process before training...')
    train_dataset = dataset(config['path']['dataset']['train'], patch_size=config['train']['patch size'],
                            scale=config['scale'])
    train_data = DataLoader(
        dataset=train_dataset, batch_size=config['train']['batch size'],
        shuffle=True, num_workers=16
    )

    valid_dataset = dataset(config['path']['dataset']['valid'], patch_size=config['train']['patch size'],
                            scale=config['scale'], is_train=False)
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid']['batch size'], num_workers=4)

    # training details - epochs & iterations
    iterations_per_epoch = len(train_dataset) // config['train']['batch size'] + 1
    n_epoch = config['train']['iterations'] // iterations_per_epoch + 1
    print('epochs scheduled: %d , iterations per epoch %d...' % (n_epoch, iterations_per_epoch))

    # define main SR network as generator
    generator = Generator(scale_by=config['scale'], n_blocks=32, n_feats=256, res_scaling=0.1).cuda()
    save_path_G = config['path']['ckpt']

    # optimizer
    learning_rate = config['train']['lr']
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(G_optimizer, config['train']['decay']['every'],
                                             config['train']['decay']['by'])

    # if training from scratch, remove all validation images and logs
    if epoch_from == 0:
        if os.path.exists(config['path']['validation']):
            shutil.rmtree(config['path']['validation'])
        if os.path.exists(config['path']['logs']):
            shutil.rmtree(config['path']['logs'])

    # if training not from scratch, load weights
    else:
        if os.path.exists(save_path_G):
            ckpt = torch.load(save_path_G)
            generator.load_state_dict(ckpt['model'])
            print('reading generator checkpoints...')
            G_optimizer.load_state_dict(ckpt['opt'])
            lr_scheduler.last_epoch = epoch_from * iterations_per_epoch
            del ckpt
        else:
            raise FileNotFoundError('Pretrained weight not found.')

    os.makedirs(config['path']['validation'], exist_ok=True)
    os.makedirs(config['path']['logs'], exist_ok=True)
    writer = SummaryWriter(config['path']['logs'])

    # loss functions
    # loss = None
    squared = False
    if config['loss'] == 'L1':
        # loss = nn.L1Loss()
        squared = False
    elif config['loss'] == 'MSE':
        # loss = nn.MSELoss()
        squared = True
    loss_fn = Thresh_Error(threshold, squared, penalty)
    generator = nn.DataParallel(generator).cuda()

    # validation
    valid = validation(generator, valid_data, writer, config['path']['validation'])

    # training
    print('start training...')
    for epoch in range(epoch_from, n_epoch):
        generator = generator.train()
        epoch_loss = 0
        for i, data in enumerate(tqdm(train_data)):
            lr, gt, _ = data
            lr = lr.cuda()
            gt = gt.cuda()

            # forwarding
            sr = generator(lr)
            g_loss = loss_fn(sr, gt)
            # g_loss = loss(sr, gt)
            # distance = torch.abs(sr - gt)
            # shift = (distance - threshold)  # L1
            # thresholded_loss = relu(shift)  # penalty over t only
            # # thresholded_loss = relu(-shift)  # penalty under t only
            # if config['loss'] == 'MSE':
            #     thresholded_loss = thresholded_loss ** 2
            # g_loss = thresholded_loss.mean()  # penalty over t only
            # # g_loss = -thresholded_loss.mean()  # penalty under t only

            # back propagation
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
            lr_scheduler.step()
            epoch_loss += g_loss.item()

        print('Training loss at {:d} : {:.8f}\n'.format(epoch, epoch_loss))

        # validation
        if (epoch + 1) % config['valid']['every'] == 0:
            is_best = valid.run(epoch + 1)

            # save validation image
            valid.save(tag='latest')
            if is_best:
                ckpt = {'model': generator.module.state_dict(), 'opt': G_optimizer.state_dict()}
                torch.save(ckpt, save_path_G)
            torch.cuda.empty_cache()


    # training process finished.
    # final validation and save checkpoints
    is_best = valid.run(n_epoch)
    valid.save(tag='final')
    writer.close()
    if is_best:
        ckpt = {'model': generator.module.state_dict(), 'opt': G_optimizer.state_dict()}
        torch.save(ckpt, save_path_G)

    print('training finished.')


if __name__ == '__main__':
    train(_config, epoch_from=0)

    # from PIL import Image
    # import numpy as np
    # r = np.zeros([250, 250, 1])
    # g = np.zeros([250, 250, 1])
    # b = np.zeros([250, 250, 1])
    #
    # # e = 50 / np.sqrt(3) + 1
    # e = 30
    # r.fill(100 + e)
    # g.fill(100 + e)
    # b.fill(100 + e)
    #
    # img = np.concatenate([r, g, b], axis=2)
    # # img.fill(120)
    # img = img.astype(np.uint8)
    # Image.fromarray(img).save('./temp3.png')
    #
    # from validation import PSNR
    # psnr = PSNR(max=255)
    # from torchvision.transforms.functional import to_tensor
    # tmp1 = to_tensor(Image.open('temp1.png').convert('RGB')) * 255  # 100 100 100
    # tmp2 = to_tensor(Image.open('temp2.png').convert('RGB')) * 255  # 100 100 150
    # tmp3 = to_tensor(Image.open('temp3.png').convert('RGB')) * 255  # 130 130 130
    # tmp4 = to_tensor(Image.open('temp4.png').convert('RGB')) * 255  # 150 150 150
    #
    # mse = torch.mean((tmp1 - tmp2) ** 2)
    # psnr_12 = psnr(mse)
    # mse = torch.mean((tmp1 - tmp3) ** 2)
    # psnr_13 = psnr(mse)
    # mse = torch.mean((tmp1 - tmp4) ** 2)
    # psnr_14 = psnr(mse)
    # print(psnr_12, psnr_13, psnr_14)
