import torch
import torch.nn as nn
import torch.nn.functional as F

bceloss = nn.BCEWithLogitsLoss().cuda()


class GAN_Loss:
    def __init__(self, Discriminator):
        self.D = Discriminator

    # WGAN-GP
    def D_WGAN_GP(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        gradient_penalty = self.compute_gradient_penalty(real, fake)
        d_loss = fake_logit - real_logit + 10. * gradient_penalty

        return d_loss

    def G_WGAN_GP(self, fake):
        return self.G_WGAN(fake)

    # WGAN
    def D_WGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        d_loss = fake_logit - real_logit

        return d_loss

    def G_WGAN(self, fake):
        fake_logit = self.D(fake)

        return -fake_logit

    # computation of Gradient Penalty: for WGAN-GP
    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.randn(real_samples.size(0), 1, 1, 1)
        if torch.cuda.is_available():
            alpha = alpha.cuda()

        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
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
        # gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    # LSGAN
    def D_LSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        d_loss = fake_logit ** 2 + (real_logit - 1) ** 2

        return d_loss

    def G_LSGAN(self, fake):
        fake_logit = self.D(fake)

        return (fake_logit - 1) ** 2

    # relativistic GAN
    def D_RSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return -F.logsigmoid(real_logit - fake_logit)

    def G_RSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return -F.logsigmoid(fake_logit - real_logit)

    # relativistic average GAN
    def D_RaGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        zero = torch.zeros(fake_logit.size()).cuda()
        one = torch.ones(real_logit.size()).cuda()

        pred_real = bceloss(real_logit - fake_logit.mean(), one)
        pred_fake = bceloss(fake_logit - real_logit.mean(), zero)

        return -(pred_real + pred_fake)

    def G_RaGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        zero = torch.zeros(real_logit.size()).cuda()
        one = torch.ones(fake_logit.size()).cuda()

        pred_fake = bceloss(fake_logit - real_logit.mean(), one)
        pred_real = bceloss(real_logit - fake_logit.mean(), zero)

        return -(pred_fake + pred_real)

    # relativistic LSGAN
    def D_RLSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return (real_logit - fake_logit - 1) ** 2 + (fake_logit - real_logit) ** 2

    def G_RLSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return (fake_logit - real_logit - 1) ** 2 + (real_logit - fake_logit) ** 2

    # relativistic average LSGAN
    def D_RaLSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return (real_logit - fake_logit.mean() - 1) ** 2 + (fake_logit - real_logit.mean()) ** 2

    def G_RaLSGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        return (fake_logit - real_logit.mean() - 1) ** 2 + (real_logit - fake_logit.mean()) ** 2

    # Vanilla GAN
    def D_MMGAN(self, real, fake):
        fake_logit = self.D(fake)
        real_logit = self.D(real)

        zero = torch.zeros(fake_logit.size()).cuda()
        one = torch.ones(real_logit.size()).cuda()

        fake_prob = bceloss(fake_logit, zero)
        real_prob = bceloss(real_logit, one)

        return (real_prob + fake_prob) / 2

    def G_MMGAN(self, fake):
        fake_logit = self.D(fake)
        one = torch.ones(fake_logit.size()).cuda()

        return bceloss(fake_logit, one)

    # Vanilla Non-saturating GAN
    def G_NSGAN(self, fake):
        fake_logit = self.D(fake)

        return -F.logsigmoid(fake_logit)
