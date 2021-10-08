import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from torch.nn.functional import relu

from config import config as _config
from utils import dataset
from validation import validation

from models import EDSR as Generator
from tqdm import tqdm

# proj_directory = '/project'
# data_directory = '/dataset'


def train(config, epoch_from=0):
    threshold = 10
    threshold = threshold / 127.5

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
    loss = None
    if config['loss'] == 'L1':
        loss = nn.L1Loss()
    elif config['loss'] == 'MSE':
        loss = nn.MSELoss()

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
            # g_loss = loss(sr, gt)
            # g_loss = torch.mean(relu(torch.abs(sr - gt) - threshold))  # error below thresh - no penalty
            g_loss = torch.mean(-relu(-(torch.abs(sr - gt) - threshold)))  # error over thresh - no penalty

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


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import os
# from tensorboardX import SummaryWriter
#
# from utils import training_dataset
# from utils import evaluation_dataset as evaluation_dataset
# from evaluation import validate_img
# from config import config as _config
#
# from models import EDSR as Generator
#
# # proj_directory = '/project'
# # data_directory = '/dataset'
#
#
# def train(config, epoch_from=0):
#     model = config['model']
#     print('process before training...')
#
#     dataset = training_dataset(config['path']['dataset']['train'], in_norm=config['in_norm'])
#     train_data = DataLoader(
#         dataset=dataset, batch_size=config['train']['batch size'],
#         shuffle=True, num_workers=10
#     )
#
#     valid_dataset = evaluation_dataset(config['path']['dataset']['valid'], in_norm=config['in_norm'])
#     valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid']['batch size'])
#     n_valid = valid_dataset.__len__()
#
#     iterations_per_epoch = dataset.__len__() // config['train']['batch size'] + 1
#     n_epoch = config['train']['iterations'] // iterations_per_epoch
#     print('epochs scheduled: %d , iterations per epoch %d...' % (n_epoch, iterations_per_epoch))
#
#     n_epoch_decay = config['train']['decay']['every'] // iterations_per_epoch
#     print('lr decay every ', n_epoch_decay)
#
#     if not os.path.exists(config['path']['validation']):
#         os.makedirs(config['path']['validation'])
#     if not os.path.exists(config['path']['ckpt']['dir']):
#         os.makedirs(config['path']['ckpt']['dir'])
#     if not os.path.exists(config['path']['logs']):
#         os.makedirs(config['path']['logs'])
#     writer = SummaryWriter(config['path']['logs'])
#
#     # generator = Generator().cuda()
#     generator = Generator(scale_by=config['scale_by'], n_blocks=32, n_feats=256, res_scaling=0.1).cuda()
#     save_path_G = config['path']['ckpt'][model]
#
#     # if training from scratch, remove all validation images and logs
#     if epoch_from == 0:
#         if os.path.exists(config['path']['validation']):
#             _old = os.listdir(config['path']['validation'])
#             for f in _old:
#                 if os.path.isfile(os.path.join(config['path']['validation'], f)):
#                     os.remove(os.path.join(config['path']['validation'], f))
#         if os.path.exists(config['path']['logs']):
#             _old = os.listdir(config['path']['logs'])
#             for f in _old:
#                 if os.path.isfile(os.path.join(config['path']['logs'], f)):
#                     os.remove(os.path.join(config['path']['logs'], f))
#     # if training not from scratch, load weights
#     else:
#         if os.path.exists(save_path_G):
#             generator.load_state_dict(torch.load(save_path_G))
#             print('reading generator checkpoints...')
#         else:
#             raise FileNotFoundError('Pretrained weight not found.')
#
#     # train Generator based on MSE
#     learning_rate = config['train']['lr']
#     G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
#
#     # loss functions predefined
#     mse = nn.MSELoss().cuda()
#     loss = mse
#     if config['loss'] == 'L1':
#         loss = nn.L1Loss().cuda()
#
#     # training
#     print('start training...')
#     for epoch in range(epoch_from, n_epoch):
#         generator = generator.train()
#         if epoch % n_epoch_decay == 0:
#             learning_rate *= config['train']['decay']['by'] ** (epoch // n_epoch_decay)
#             G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
#
#         for i, data in enumerate(train_data):
#             lr, gt, name = data
#             lr = lr.float().cuda()
#             gt = gt.float().cuda()
#
#             # forwarding
#             G_optimizer.zero_grad()
#             sr = generator(lr)
#             g_loss = loss(sr, gt)
#
#             # back propagation
#             g_loss.backward()
#             G_optimizer.step()
#
#         # validation every epoch
#         if epoch % config['valid']['every'] == 0:
#             generator = generator.eval()
#             val_mse_loss = 0
#             for _, val_data in enumerate(valid_data):
#                 lr, gt, img_name = val_data
#                 lr = lr.float().cuda()
#                 gt = gt.float().cuda()
#
#                 with torch.no_grad():
#                     sr = generator(lr)
#
#                 val_mse_loss += mse(sr, gt).item()
#
#             val_mse_loss /= n_valid
#             print("Validation loss(MSE) at %2d:\t==>\t%.4f" % (epoch, val_mse_loss))
#             writer.add_scalar('G Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
#             writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))
#
#         # validation with an image
#         if epoch % config['valid']['img_every'] == 0:
#             validate_img(generator, config['scale_by'], config['path']['dataset']['valid_w_img'], tag=str(epoch))
#
#             # save checkpoints
#             torch.save(generator.state_dict(), save_path_G)
#
#     # training process finished.
#     # final validation and save checkpoints
#     writer.close()
#     torch.save(generator.state_dict(), save_path_G)
#     generator = generator.eval()
#     validate_img(generator, config['scale_by'], config['path']['dataset']['valid_w_img'], tag='final')
#
#     print('training finished.')
#
#
# if __name__ == '__main__':
#     train(_config, epoch_from=0)
