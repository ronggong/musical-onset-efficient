import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from bock_no_rnn_basecode import run_training_process
import sys

if __name__ == '__main__':

    part = int(sys.argv[1]) # 0: folds 0-3, 1: folds 4-7
    low_cv_bound = int(sys.argv[2]) # part0: 0, part1:4
    high_cv_bound = int(sys.argv[3]) # part0: 4, part1:8
    deep = sys.argv[4]
    deep_bool = True if deep == 'deep' else False
    # part = 0
    # low_cv_bound = 0
    # high_cv_bound = 2

    for ii in range(low_cv_bound, high_cv_bound):
        run_training_process(part, ii, deep_bool)