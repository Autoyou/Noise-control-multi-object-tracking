# Copyright (c) .Yunnan Key Laboratory of Computer Technologies Application
# Kunming University of Science and Technology, Kunming 650500, China and its affiliates. All Rights Reserved
# Modified from StrongSORT（https://github.com/dyhBUPT/StrongSORT）
# Original author: Du Yunhao. and its affiliates. All Rights Reserved
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from GBI import GBInterpolation

if __name__ == '__main__':

    for i, seq in enumerate(opt.sequences, start=1):
        print('processing the {}th video {}...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')
        """
        change the path into your path of interresult
        """
        path_out = join('/home/kmust/下载/open/GBRC/interresult', seq + '.txt')
        GBInterpolation(
                path_in=path_save,
                path_out=path_out,
                interval=20,
                tau=10.5
        )




