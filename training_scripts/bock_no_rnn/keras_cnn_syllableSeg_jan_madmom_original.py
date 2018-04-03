import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

from bock_no_rnn_basecode import run_training_process

if __name__ == '__main__':

    for ii in range(low_cv_bound, high_cv_bound):
        run_training_process(part, ii, False, False)