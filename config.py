<<<<<<< HEAD
name = 'baseline'
config = {
  'train': {
    'batch size': 16,
    'patch size': 192,
    'iterations': 300000,
    'lr': 1e-4,
    'decay': {
=======
config = {
  'model': 'EDSR',
  'train': {
    'batch size': 16,
    'iterations': 1000000,
    'lr': 1e-4,
    'decay':{
>>>>>>> c763628aee6f7b00b358442739a3ec261832fe33
      'every': 2e+5,
      'by': 0.5
    }
  },
<<<<<<< HEAD
  'valid': {
=======

  'valid':{
>>>>>>> c763628aee6f7b00b358442739a3ec261832fe33
    'batch size': 1,
    'every': 1,
    'img_every': 100
  },

<<<<<<< HEAD
  'scale': 4,
=======
  'in_norm': (-1, 1),
  'out_norm': (-1, 1),
  'scale_by': 4,
>>>>>>> c763628aee6f7b00b358442739a3ec261832fe33
  'loss': 'L1',

  'path': {
    'project': '/project',
<<<<<<< HEAD
    'ckpt': f'/project/{name}.pth',

    'dataset': {
      'train': ['/dataset/DIV2K/train_HR',
                '/dataset/Flickr2K/Flickr2K_HR'],
      'valid': ['/dataset/DIV2K/valid_HR'],
    },
    'validation': f'/project/validation/{name}',
    'logs': f'/project/logs/{name}'
  }
}
=======
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
    'logs': '/project/logs'
  }

}
>>>>>>> c763628aee6f7b00b358442739a3ec261832fe33
