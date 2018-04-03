import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[3])

# from keras import backend as K
#
# if 'tensorflow' == K.backend():
#     import tensorflow as tf
#     from keras.backend.tensorflow_backend import set_session
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.visible_device_list = str(sys.argv[3])
#     set_session(tf.Session(config=config))

from syllableSeg_jan_madmom_original_basecode_batch import syllableSeg_jan_madmom_original_basecode

if __name__ == '__main__':

    low_cv_bound = int(sys.argv[1]) # part0: 0, part1:4
    high_cv_bound = int(sys.argv[2]) # part0: 4, part1:8


    for ii in range(low_cv_bound, high_cv_bound):
        syllableSeg_jan_madmom_original_basecode(ii, 'ismir')