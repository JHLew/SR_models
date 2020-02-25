import torch.nn as nn
import torch
from utils import np_to_torch_tensor, torch_tensor_to_np


# EDSR architecture
class EDSR(nn.Module):
    def __init__(self, scale_by, n_blocks, n_feats, res_scaling, n_colors=3):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(in_channels=n_colors, out_channels=n_feats, kernel_size=3, padding=1)

        res_blocks = [ResidualBlock(n_feats, res_scaling=res_scaling) for _ in range(n_blocks)]
        res_blocks.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*res_blocks)

        self.tail = nn.Sequential(
            Upsample(n_feats, scale_by=scale_by),
            nn.Conv2d(n_feats, n_colors, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        _result = self.tail(res + x)

        return _result


# SRResNet(SRGAN) architecture
class SRResNet(nn.Module):
    def __init__(self, n_colors=3):
        super(SRResNet, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=n_colors, out_channels=64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        res_blocks = [ResidualBlock(64, bn=True, act='prelu') for _ in range(16)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.block17 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            Upsample(64, 2, act='prelu'),
            Upsample(64, 2, act='prelu')
        )

        self.final = nn.Conv2d(64, n_colors, 9, padding=4)

    def forward(self, x):
        block0 = self.block0(x)
        res_blocks = self.res_blocks(block0)
        block17 = self.block17(res_blocks)
        upsampled = self.upsample(block17 + block0)
        _result = self.final(upsampled)

        return _result


# Discriminator from SRGAN - Instance Norm is used instead of Batch Norm: for WGAN-GP
class Discriminator(nn.Module):
    def __init__(self, n_colors=3):
        super(Discriminator, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(n_colors, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.mid_blocks = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        x = self.block0(x)
        x = self.mid_blocks(x)
        _result = self.final_block(x)

        return _result


# default Residual Block used
class ResidualBlock(nn.Module):
    def __init__(self, n_channels, bn=False, act='relu', res_scaling=1):
        super(ResidualBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(n_channels, n_channels, 3, padding=1))
        if bn:
            block.append(nn.BatchNorm2d(n_channels))

        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())

        block.append(nn.Conv2d(n_channels, n_channels, 3, padding=1))
        if bn:
            block.append(nn.BatchNorm2d(n_channels))

        self.block = nn.Sequential(*block)
        self.res_scale = res_scaling

    def forward(self, x):
        res = self.block(x) * self.res_scale

        return x + res


# default Upsampling block: uses Pixel Shuffling
class Upsample(nn.Module):
    def __init__(self, in_channels, scale_by, act='relu'):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_by ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_by)
        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'prelu':
            self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)

        return x


# module for simple forwarding
class SR_network(nn.Module):
    def __init__(self, model, eval_mode=True, dtype='np_array', w_path=None, in_norm=(-1, 1), out_norm=(-1, 1)):
        super().__init__()
        """
        configure model for forwarding.
        can be set to evaluation mode, with path to weights set
        data type of input and output can be configured as
        torch tensor(B, C, H, W) or a numpy array(H, W, C)
        """

        """
        EDSR - in: (-1, 1), out: (-1, 1)
        SRResNet - in: (0, 1), out: (-1, 1)
        """

        self.generator = model.cuda()
        self.mode = eval_mode
        self.in_norm = in_norm
        self.out_norm = out_norm

        if eval_mode:
            self.generator.eval()
            if w_path is None:
                raise SyntaxError('To use the evaluation mode, the path to weight must be specified.')

        if dtype == 'torch_tensor' or dtype == 'np_array':
            self.dtype = dtype
        else:
            raise ValueError('dtype must be either "torch_tensor" or "np_array".')

        if w_path is not None:
            self.generator.load_state_dict(torch.load(w_path))

    def forward(self, x):
        """
        forwarding done based on the initialization of the class.
        normalization and renormalization done automatically if it is a numpy array.
        for torch tensor, none is done but forwarding.
        :param x: input
        :return:
        output is either a torch tensor (B, C, H, W) or a numpy array(H, W, C)
        """
        if self.dtype == 'np_array':
            x = np_to_torch_tensor(x, norm_from=(0, 255), norm_to=self.in_norm)

        x = x.float().cuda()

        if self.mode:
            with torch.no_grad():
                _output = self.generator(x)
        else:
            _output = self.generator(x)

        if self.dtype == 'np_array':
            _output = torch_tensor_to_np(_output, norm_from=self.out_norm, norm_to=(0, 255))

        return _output
