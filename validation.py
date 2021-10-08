import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import os

class PSNR:
    def __init__(self, max):
        super(PSNR, self).__init__()
        self.max = max

    def __call__(self, mse):
        return 10 * torch.log(self.max / mse)


class validation:
    mse = nn.MSELoss()
    psnr = PSNR(max=1.)

    def __init__(self, network, loader, writer, save_path):
        self.generator = network
        self.loader = loader
        self.writer = writer
        self.n = len(loader.dataset)
        self.save_path = save_path
        self.best = 0

    def run(self, epoch):
        generator = self.generator.eval()
        val_mse_loss = 0
        val_psnr = 0

        self.valid_outputs = []
        self.img_names = []

        for _, val_data in enumerate(self.loader):
            torch.cuda.empty_cache()
            lr, gt, img_name = val_data
            lr = lr.cuda()
            gt = gt.cuda()

            with torch.no_grad():
                sr = generator(lr)

                self.valid_outputs.append(sr[0].cpu())
                self.img_names.append(img_name[0])

                mse = self.mse(sr, gt)
                val_mse_loss += mse.item()
                val_psnr += self.psnr(mse).item()

        val_mse_loss /= self.n
        val_psnr /= self.n

        print(f"Validation loss(MSE) at {epoch:2d}:\t==>\tPSNR: {val_psnr:.2f}\tMSE: {val_mse_loss:.6f}")
        self.writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))
        self.writer.add_scalar('G Loss/PSNR', val_psnr, (epoch + 1))
        self.generator.train()
        if self.best <= val_psnr:
            self.best = val_psnr
            return True
        else:
            return False

    def save(self, tag):
        save_dir = os.path.join(self.save_path, str(tag))
        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.n):
            img = self.valid_outputs[i]
            name = self.img_names[i]

            F.to_pil_image(img).save(os.path.join(save_dir, name))
