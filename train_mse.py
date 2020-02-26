import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tensorboardX import SummaryWriter

from utils import training_dataset
from utils import evaluation_dataset as evaluation_dataset
from evaluation import validate_img
from config import config as _config

# from models import SRResNet as Generator
from models import EDSR as Generator

# proj_directory = '/project'
# data_directory = '/dataset'


def train(config, epoch_from=0):
    model = config['model']
    print('process before training...')

    dataset = training_dataset(config['path']['dataset']['train'], in_norm=config['in_norm'])
    train_data = DataLoader(
        dataset=dataset, batch_size=config['train']['batch size'],
        shuffle=True, num_workers=10
    )

    valid_dataset = evaluation_dataset(config['path']['dataset']['valid'], in_norm=config['in_norm'])
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid']['batch size'])
    n_valid = valid_dataset.__len__()

    iterations_per_epoch = dataset.__len__() // config['train']['batch size'] + 1
    n_epoch = config['train']['iterations'] // iterations_per_epoch
    print('epochs scheduled: %d , iterations per epoch %d...' % (n_epoch, iterations_per_epoch))

    n_epoch_decay = config['train']['decay']['every'] // iterations_per_epoch
    print('lr decay every ', n_epoch_decay)

    if not os.path.exists(config['path']['validation']):
        os.makedirs(config['path']['validation'])
    if not os.path.exists(config['path']['ckpt']['dir']):
        os.makedirs(config['path']['ckpt']['dir'])
    if not os.path.exists(config['path']['logs']):
        os.makedirs(config['path']['logs'])
    writer = SummaryWriter(config['path']['logs'])

    # generator = Generator().cuda()
    generator = Generator(scale_by=config['scale_by'], n_blocks=32, n_feats=256, res_scaling=0.1).cuda()
    save_path_G = config['path']['ckpt'][model]

    # if training from scratch, remove all validation images and logs
    if epoch_from == 0:
        if os.path.exists(config['path']['validation']):
            _old = os.listdir(config['path']['validation'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['validation'], f)):
                    os.remove(os.path.join(config['path']['validation'], f))
        if os.path.exists(config['path']['logs']):
            _old = os.listdir(config['path']['logs'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['logs'], f)):
                    os.remove(os.path.join(config['path']['logs'], f))
    # if training not from scratch, load weights
    else:
        if os.path.exists(save_path_G):
            generator.load_state_dict(torch.load(save_path_G))
            print('reading generator checkpoints...')
        else:
            raise FileNotFoundError('Pretrained weight not found.')

    # train Generator based on MSE
    learning_rate = config['train']['lr']
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

    # loss functions predefined
    mse = nn.MSELoss().cuda()
    loss = mse
    if config['loss'] == 'L1':
        loss = nn.L1Loss().cuda()

    # training
    print('start training...')
    for epoch in range(epoch_from, n_epoch):
        generator = generator.train()
        if epoch % n_epoch_decay == 0:
            learning_rate *= config['train']['decay']['by'] ** (epoch // n_epoch_decay)
            G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

        for i, data in enumerate(train_data):
            lr, gt, name = data
            lr = lr.float().cuda()
            gt = gt.float().cuda()

            # forwarding
            G_optimizer.zero_grad()
            sr = generator(lr)
            g_loss = loss(sr, gt)

            # back propagation
            g_loss.backward()
            G_optimizer.step()

        # validation every epoch
        if epoch % config['valid']['every'] == 0:
            generator = generator.eval()
            val_mse_loss = 0
            for _, val_data in enumerate(valid_data):
                lr, gt, img_name = val_data
                lr = lr.float().cuda()
                gt = gt.float().cuda()

                with torch.no_grad():
                    sr = generator(lr)

                val_mse_loss += mse(sr, gt).item()

            val_mse_loss /= n_valid
            print("Validation loss(MSE) at %2d:\t==>\t%.4f" % (epoch, val_mse_loss))
            writer.add_scalar('G Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
            writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))

        # validation with an image
        if epoch % config['valid']['img_every'] == 0:
            validate_img(generator, config['scale_by'], config['path']['dataset']['valid_w_img'], tag=str(epoch))

            # save checkpoints
            torch.save(generator.state_dict(), save_path_G)

    # training process finished.
    # final validation and save checkpoints
    writer.close()
    torch.save(generator.state_dict(), save_path_G)
    generator = generator.eval()
    validate_img(generator, config['scale_by'], config['path']['dataset']['valid_w_img'], tag='final')

    print('training finished.')


if __name__ == '__main__':
    train(_config, epoch_from=0)
