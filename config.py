config = {
  'model': 'SRResNet',
  'train': {
    'batch size': 16,
    'iterations': 1000000,
    'lr': 1e-4,
    'decay':{
      'at': None,
      'every': None
    }
  },

  'valid':{
    'batch size': 1,
    'every': 1,
    'img_every': 100
  },

  'in_norm': (-1, 1),
  'out_norm': (-1, 1),

  'path': {
    'project': '/project',
    'ckpt': {
      'dir': '/project/ckpt',
      'EDSR': '/project/ckpt/EDSR.pth',
      'SRResNet': '/project/ckpt/SRResNet.pth',
      'SRGAN': '/project/ckpt/SRGAN.pth'
    },

    'dataset': {
      'dir': '/dataset',
      'train': ['/dataset/DIV2K/train_HR',
                '/dataset/Flickr2K/Flickr2K_HR'],
      'valid': ['/dataset/DIV2K/valid_HR'],
      'valid_w_img': '/dataset/Set14/comic.png'
    },

    'validation': '/project/validation',
    'logs': '/project/logs',

  }
}