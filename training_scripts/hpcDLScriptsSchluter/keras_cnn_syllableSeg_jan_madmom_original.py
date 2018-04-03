import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from syllableSeg_jan_madmom_original_basecode import syllableSeg_jan_madmom_original_basecode
import sys

if __name__ == '__main__':

    part = int(sys.argv[1]) # 0: folds 0-3, 1: folds 4-7
    low_cv_bound = int(sys.argv[2]) # part0: 0, part1:4
    high_cv_bound = int(sys.argv[3]) # part0: 4, part1:8
    # part = 0
    # low_cv_bound = 0
    # high_cv_bound = 2

    for ii in range(low_cv_bound, high_cv_bound):
        syllableSeg_jan_madmom_original_basecode(part, ii, False, False)