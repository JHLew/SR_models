import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tensorboardX import SummaryWriter

from utils import training_dataset
from utils import evaluation_dataset as evaluation_dataset

from models import SRResNet as Generator
from evaluation import validate_img
from config import config

# proj_directory = '/project'
# data_directory = '/dataset'

def train(config):
    print('process before training...')
    dataset = training_dataset(config['path']['dataset']['train'], in_norm=config['in_norm'])
    train_data = DataLoader(
        dataset=dataset, batch_size=config['train']['batch size'],
        shuffle=True, num_workers=10
    )

    valid_dataset = evaluation_dataset(config['path']['dataset']['valid'])
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid'])

    if not os.path.exists(config['path']['validation']):
        os.makedirs(config['path']['validation'])
    if not os.path.exists(config['path']['ckpt']['dir']):
        os.makedirs(config['path']['ckpt']['dir'])
    if not os.path.exists(config['path']['logs']):
        os.makedirs(config['path']['logs'])
    writer = SummaryWriter(config['path']['logs'])

    iterations_per_epoch = dataset.__len__() // config['batch size'] + 1
    n_epoch = config['train']['iterations'] // iterations_per_epoch
    print('epochs scheduled: %d , iterations per epoch %d...' % (n_epoch, iterations_per_epoch))

    generator = Generator().cuda()
    save_path_G = config['path']['ckpt']['SRResNet']

    if os.path.exists(save_path_G):
        generator.load_state_dict(torch.load(save_path_G))
        print('reading generator checkpoints...')

    # train Generator based on MSE
    G_optimizer = optim.Adam(generator.parameters(), lr=config['train']['lr'])

    # loss functions predefined
    mse = nn.MSELoss().cuda()

    # training
    print('start training...')
    for epoch in range(n_epoch):
        generator = generator.train()
        for i, data in enumerate(train_data):
            lr, gt, name = data
            lr = lr.float().cuda()
            gt = gt.float().cuda()

            # forwarding
            G_optimizer.zero_grad()
            sr = generator(lr)
            g_loss = mse(sr, gt)

            # back propagation
            g_loss.backward()
            G_optimizer.step()

        # validation every epoch
        if epoch % config['valid']['every']:
            generator = generator.eval()
            val_mse_loss = 0
            n_it = 0
            for _, val_data in enumerate(valid_data):
                lr, gt, img_name = val_data
                lr = lr.float().cuda()
                gt = gt.float().cuda()

                with torch.no_grad():
                    sr = generator(lr)

                val_mse_loss += mse(sr, gt).item()
                n_it += 1

            val_mse_loss /= n_it
            print("Validation loss(MSE) at %2d:\t==>\t%.4f" % (epoch, val_mse_loss))
            writer.add_scalar('G Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
            writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))

        # validation with an image
        if epoch % config['valid']['img_every'] == 0:
            validate_img(generator, config['path']['dataset']['valid_w_img'], tag=str(epoch))

            # save checkpoints
            torch.save(generator.state_dict(), save_path_G)

    # training process finished.
    # final validation and save checkpoints
    writer.close()
    torch.save(generator.state_dict(), save_path_G)
    generator = generator.eval()
    validate_img(generator, config['path']['dataset']['valid_w_img'], tag='final')

    print('training finished.')


if __name__ == '__main__':
    train(config)
