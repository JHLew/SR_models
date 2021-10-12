import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import os
from glob import glob
from config import config
from models import EDSR as Generator
from PIL import Image

if __name__ == '__main__':
    val_data_list = 'C:/Users/JH/Datasets/DIV2K/train_HR/*'
    name = 'penalty_under_10'
    ckpt = f'./{name}.pth'
    scale = 4
    save_path = './eval'

    generator = Generator(scale_by=config['scale'], n_blocks=32, n_feats=256, res_scaling=0.1).cuda()
    generator.load_state_dict(torch.load(ckpt)['model'])
    generator = generator.eval()

    val_data_list = glob(val_data_list)
    os.makedirs(os.path.join(save_path, name), exist_ok=True)

    for img in val_data_list:
        torch.cuda.empty_cache()
        img_name = os.path.basename(img)
        img = Image.open(img).convert('RGB')
        w, h = img.size
        lr_w, lr_h = int(w // scale), int(h // scale)
        lr = img.resize((lr_w, lr_h), resample=Image.BICUBIC)
        lr = F.to_tensor(lr).unsqueeze(0)
        lr = lr.cuda()

        with torch.no_grad():
            sr = generator(lr)
            F.to_pil_image(sr[0].cpu()).save(f'./{save_path}/{name}/{img_name}')
