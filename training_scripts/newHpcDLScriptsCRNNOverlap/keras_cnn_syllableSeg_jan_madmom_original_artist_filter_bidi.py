import sys

from syllableSeg_jan_madmom_original_basecode import syllableSeg_jan_madmom_original_basecode

if __name__ == '__main__':

    low_cv_bound = int(sys.argv[1]) # part0: 0, part1:4
    high_cv_bound = int(sys.argv[2]) # part0: 4, part1:8
    # part = 0
    # low_cv_bound = 0
    # high_cv_bound = 2

    for ii in range(low_cv_bound, high_cv_bound):
        syllableSeg_jan_madmom_original_basecode(ii, 'artist_filter', bidi=True)