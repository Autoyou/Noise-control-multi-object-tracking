import argparse

data = {
    'MOT16': {
        'val':[
            'MOT16-01',
        ],
        'test':[
            'MOT16-01',
        ]
    },
    'MOT20': {
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--dataset',
            type=str,
            default='MOT16',
            help='MOT16 or MOT20',
        )
        self.parser.add_argument(
            '--mode',
            default='val',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--dir_save',
            default='/home/kmust/下载/open/GBRC/tmp'
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.sequences = data[opt.dataset][opt.mode]
        return opt

opt = opts().parse()