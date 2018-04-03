import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_jingju_phrase import getTrainingFilenames
from data_preparation_CRNN import featureLabelSampleWeightsLoad, featureLabelSampleWeightsPad
from data_preparation_CRNN import createInputTensor
from data_preparation_CRNN import writeValLossCsv
from models_CRNN import jan_original
from sklearn.metrics import log_loss
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from file_path_jingju_rnn import *

def syllableSeg_jan_madmom_original_basecode(ii, dataset_str='ismir'):

    file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/'+dataset_str+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(ii) + '.h5'
    file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/'+dataset_str+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(ii) + '.csv'

    if dataset_str == 'ismir':
        jingju_feature_data_scratch_path = '/scratch/rgongcnnSyllableSeg_jan_phrase_jingju/ismir'
    else:
        jingju_feature_data_scratch_path = '/scratch/rgongcnnSyllableSeg_jan_phrase_jingju/artist_filter'

    # file_path_model = '../../temp/'+dataset_str+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(
    #     ii) + '.h5'
    # file_path_log = '../../temp/'+dataset_str+'_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(
    #     ii) + '.csv'

    # if dataset_str == 'ismir':
    #     jingju_feature_data_scratch_path = ismir_feature_data_path
    # else:
    #     jingju_feature_data_scratch_path = artist_filter_feature_data_path

    if dataset_str == 'ismir':
        scaler = pickle.load(open(scaler_ismir_phrase_model_path, 'r'))
    else:
        scaler = pickle.load(open(scaler_artist_filter_phrase_model_path, 'r'))

    train_validation_fns = getTrainingFilenames(jingju_feature_data_scratch_path)

    print(train_validation_fns)

    # split the training set to train and validation sets
    train_fns, validation_fns = None, None
    rs = ShuffleSplit(n_splits=1, test_size=.1)
    for train_idx, validation_idx in rs.split(train_validation_fns):
        train_fns = [train_validation_fns[ti] for ti in train_idx]
        validation_fns = [train_validation_fns[vi] for vi in validation_idx]

    nb_epochs = 500
    best_val_loss = 1.0 # initialize the val_loss
    counter = 0
    patience = 15   # early stopping patience

    # initialize the model
    model = jan_original(filter_density=1,
                       dropout=0.5,
                       input_shape=(1, len_seq, 1, 80, 15),
                       batchNorm=False,
                       dense_activation='sigmoid',
                       channel=1,
                         stateful=True)

    for ii_epoch in range(nb_epochs):

        # training
        for tfn in train_fns:

            mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(jingju_feature_data_scratch_path,
                                                                             tfn,
                                                                             scaler)

            mfcc_line_pad, label_pad, sample_weights_pad, _ = \
                featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, len_seq)

            for ii_iter in range(len(mfcc_line_pad)/len_seq):

                mfcc_line_tensor, label_tensor, sample_weights_tensor = \
                    createInputTensor(mfcc_line_pad, label_pad, sample_weights_pad, len_seq, ii_iter)

                model.train_on_batch(mfcc_line_tensor,
                                   label_tensor,
                                   sample_weight=sample_weights_tensor)
            model.reset_states()


        # validation
        y_pred_val_all = np.array([])
        label_val_all = np.array([])

        for vfn in validation_fns:

            mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(jingju_feature_data_scratch_path,
                                                                             vfn,
                                                                             scaler)

            mfcc_line_pad, label_pad, sample_weights_pad, len_padded = \
                featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, len_seq)

            iter_time = len(mfcc_line_pad)/len_seq
            for ii_iter in range(iter_time):

                mfcc_line_tensor, label_tensor, _ = \
                    createInputTensor(mfcc_line_pad, label_pad, sample_weights_pad, len_seq, ii_iter)

                y_pred = model.predict_on_batch(mfcc_line_tensor)

                # remove the padded samples
                if ii_iter == iter_time-1 and len_padded > 0:
                    y_pred = y_pred[:, :len_seq-len_padded,:]
                    label_tensor = label_tensor[:, :len_seq-len_padded, :]

                # reset the states at the end of each sequence
                if ii_iter == iter_time-1:
                    model.reset_states()

                # reduce the label dimension
                y_pred = y_pred.reshape((y_pred.shape[1], ))
                label_tensor = label_tensor.reshape((label_tensor.shape[1], ))

                y_pred_val_all = np.append(y_pred_val_all, y_pred)
                label_val_all = np.append(label_val_all, label_tensor)

        val_loss = log_loss(label_val_all, y_pred_val_all)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model.save_weights(file_path_model)
        else:
            counter += 1

        # write validation loss to csv
        writeValLossCsv(file_path_log, ii_epoch, val_loss)

        # early stopping
        if counter >= patience:
            break

        random.shuffle(train_fns)

if __name__ == '__main__':
    syllableSeg_jan_madmom_original_basecode(0, 'ismir')