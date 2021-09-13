name = 'baseline'
config = {
  'train': {
    'batch size': 16,
    'patch size': 192,
    'iterations': 300000,
    'lr': 1e-4,
    'decay': {
      'every': 2e+5,
      'by': 0.5
    }
  },
  'valid': {
    'batch size': 1,
    'every': 1,
    'img_every': 100
  },

  'scale': 4,
  'loss': 'L1',

  'path': {
    'project': '/project',
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
