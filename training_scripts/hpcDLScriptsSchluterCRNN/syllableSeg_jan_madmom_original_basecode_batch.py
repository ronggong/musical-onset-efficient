import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_schluter import getTrainingFilenames
from data_preparation_CRNN import featureLabelSampleWeightsLoad, featureLabelSampleWeightsPad, featureLabelSampleWeightsPad2Length
from data_preparation_CRNN import writeValLossCsv
from data_preparation_CRNN import createInputTensor
from data_preparation_CRNN import reinitialize_train_variables, reinitialize_train_tensor
from models_CRNN import jan_original
from sklearn.metrics import log_loss
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from file_path_bock import *

def syllableSeg_jan_madmom_original_basecode(ii):

    file_path_model = '/homedtic/rgong/cnnSyllableSeg/out/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(ii) + '.h5'
    file_path_log = '/homedtic/rgong/cnnSyllableSeg/out/log/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(ii) + '.csv'
    schluter_feature_data_scratch_path = '/scratch/rgongcnnSyllableSeg_jan_phrase_schluter/syllableSeg/'

    # file_path_model = '../../temp/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(
    #     ii) + '.h5'
    # file_path_log = '../../temp/schulter_jan_madmom_simpleSampleWeighting_early_stopping_adam_cv_phrase' + str(
    #     ii) + '.csv'
    # schluter_feature_data_scratch_path = schluter_feature_data_path_madmom_simpleSampleWeighting_phrase

    test_cv_filename = os.path.join(bock_cv_path, '8-fold_cv_random_' + str(ii) + '.fold')
    train_validation_fns = getTrainingFilenames(bock_annotations_path, test_cv_filename)

    # split the training set to train and validation sets
    train_fns, validation_fns = None, None
    rs = ShuffleSplit(n_splits=1, test_size=.1)
    for train_idx, validation_idx in rs.split(train_validation_fns):
        train_fns = [train_validation_fns[ti] for ti in train_idx]
        validation_fns = [train_validation_fns[vi] for vi in validation_idx]

    scaler = pickle.load(open(scaler_bock_phrase_model_path, 'r'))

    nb_epochs = 500
    best_val_loss = 1.0 # initialize the val_loss
    counter = 0
    patience = 15   # early stopping patience

    input_shape = (batch_size, len_seq, 1, 80, 15)

    # initialize the model
    model = jan_original(filter_density=1,
                         dropout=0.5,
                         input_shape=input_shape,
                         batchNorm=False,
                         dense_activation='sigmoid',
                         channel=1,
                         stateful=True)

    input_shape_val = (1, len_seq, 1, 80, 15)

    # initialize the model
    model_val = jan_original(filter_density=1,
                             dropout=0.5,
                             input_shape=input_shape_val,
                             batchNorm=False,
                             dense_activation='sigmoid',
                             channel=1,
                             stateful=True)

    for ii_epoch in range(nb_epochs):

        # training
        list_mfcc_line, list_label, list_sample_weights, len_max_train, batch_counter \
            = reinitialize_train_variables()

        for ii_tfn, tfn in enumerate(train_fns):

            batch_counter += 1

            mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(schluter_feature_data_scratch_path,
                                                                             tfn,
                                                                             scaler)

            list_mfcc_line.append(mfcc_line)
            list_label.append(label)
            list_sample_weights.append(sample_weights)

            # the max length
            len_max_train = len(mfcc_line) if len(mfcc_line) > len_max_train else len_max_train

            if batch_counter >= batch_size - 1 or ii_tfn >= len(train_fns) - 1:
                # the padded length
                padded_len = int(len_seq * np.ceil(len_max_train / float(len_seq)))

                # pad them into the maximum padded length
                for ii_batch in range(len(list_mfcc_line)):
                    list_mfcc_line[ii_batch], list_label[ii_batch], list_sample_weights[ii_batch], _ = \
                        featureLabelSampleWeightsPad2Length(list_mfcc_line[ii_batch],
                                                            list_label[ii_batch],
                                                            list_sample_weights[ii_batch],
                                                            padded_len)

                # time loop
                for ii_iter in range(padded_len / len_seq):

                    mfcc_line_tensor, label_tensor, sample_weights_tensor = \
                        reinitialize_train_tensor(input_shape, batch_size, len_seq)

                    idx_start = len_seq * ii_iter
                    idx_end = idx_start + len_seq

                    # batch loop
                    for ii_batch in range(len(list_mfcc_line)):
                        mfcc_line_tensor[ii_batch, :, 0, :, :] = list_mfcc_line[ii_batch][idx_start:idx_end]
                        label_tensor[ii_batch, :, 0] = list_label[ii_batch][idx_start:idx_end]
                        sample_weights_tensor[ii_batch, :] = list_sample_weights[ii_batch][idx_start:idx_end]

                    model.train_on_batch(mfcc_line_tensor,
                                         label_tensor,
                                         sample_weight=sample_weights_tensor)

                list_mfcc_line, list_label, list_sample_weights, len_max_train, batch_counter \
                    = reinitialize_train_variables()

                model.reset_states()

        weights_trained = model.get_weights()
        model_val.set_weights(weights_trained)

        # validation non-overlap
        y_pred_val_all = np.array([], dtype='float32')
        label_val_all = np.array([], dtype='int')

        for vfn in validation_fns:

            mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(schluter_feature_data_scratch_path,
                                                                             vfn,
                                                                             scaler)

            mfcc_line_pad, label_pad, sample_weights_pad, len_padded = \
                featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, len_seq)

            iter_time = len(mfcc_line_pad) / len_seq
            for ii_iter in range(iter_time):

                mfcc_line_tensor, label_tensor, _ = \
                    createInputTensor(mfcc_line_pad, label_pad, sample_weights_pad, len_seq, ii_iter)

                y_pred = model_val.predict_on_batch(mfcc_line_tensor)

                # remove the padded samples
                if ii_iter == iter_time - 1 and len_padded > 0:
                    y_pred = y_pred[:, :len_seq - len_padded, :]
                    label_tensor = label_tensor[:, :len_seq - len_padded, :]

                # reduce the label dimension
                y_pred = y_pred.reshape((y_pred.shape[1],))
                label_tensor = label_tensor.reshape((label_tensor.shape[1],))

                y_pred_val_all = np.append(y_pred_val_all, y_pred)
                label_val_all = np.append(label_val_all, label_tensor)

            model_val.reset_states()

        val_loss = log_loss(label_val_all, y_pred_val_all)

        # # validation
        # y_pred_val_all = np.array([])
        # label_val_all = np.array([])
        # sample_weights_val_all = np.array([])
        #
        # list_mfcc_line, list_label, list_sample_weights, len_max_train, batch_counter \
        #     = reinitialize_train_variables()
        # for ii_vfn, vfn in enumerate(validation_fns):
        #
        #     batch_counter += 1
        #
        #     mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(schluter_feature_data_scratch_path,
        #                                                                      vfn,
        #                                                                      scaler)
        #
        #     list_mfcc_line.append(mfcc_line)
        #     list_label.append(label)
        #     list_sample_weights.append(sample_weights)
        #
        #     # the max length
        #     len_max_train = len(mfcc_line) if len(mfcc_line) > len_max_train else len_max_train
        #
        #     if batch_counter >= batch_size - 1 or ii_vfn >= len(validation_fns) - 1:
        #         # the padded length
        #         padded_len = int(len_seq * np.ceil(len_max_train / float(len_seq)))
        #
        #         # pad them into the maximum padded length
        #         for ii_batch in range(len(list_mfcc_line)):
        #             list_mfcc_line[ii_batch], list_label[ii_batch], list_sample_weights[ii_batch], _ = \
        #                 featureLabelSampleWeightsPad2Length(list_mfcc_line[ii_batch],
        #                                                     list_label[ii_batch],
        #                                                     list_sample_weights[ii_batch],
        #                                                     padded_len)
        #
        #         for ii_iter in range(padded_len / len_seq):
        #
        #             mfcc_line_tensor, label_tensor, sample_weights_tensor = \
        #                 reinitialize_train_tensor(input_shape, batch_size, len_seq)
        #
        #             idx_start = len_seq * ii_iter
        #             idx_end = idx_start + len_seq
        #
        #             for ii_batch in range(len(list_mfcc_line)):
        #                 mfcc_line_tensor[ii_batch, :, 0, :, :] = list_mfcc_line[ii_batch][idx_start:idx_end]
        #                 label_tensor[ii_batch, :, 0] = list_label[ii_batch][idx_start:idx_end]
        #                 sample_weights_tensor[ii_batch, :] = list_sample_weights[ii_batch][idx_start:idx_end]
        #
        #             y_pred = model.predict_on_batch(mfcc_line_tensor)
        #
        #             y_pred = y_pred.reshape((y_pred.shape[1] * batch_size,))
        #             label_tensor = label_tensor.reshape((label_tensor.shape[1] * batch_size,))
        #             sample_weights_tensor = sample_weights_tensor.reshape(
        #                 (sample_weights_tensor.shape[1] * batch_size,))
        #
        #             y_pred_val_all = np.append(y_pred_val_all, y_pred)
        #             label_val_all = np.append(label_val_all, label_tensor)
        #             sample_weights_val_all = np.append(sample_weights_val_all, sample_weights_tensor)
        #
        #         list_mfcc_line, list_label, list_sample_weights, len_max_train, batch_counter \
        #             = reinitialize_train_variables()
        #
        #         model.reset_states()
        #
        # idx_valid = np.nonzero(sample_weights_val_all)
        # y_pred_val_all = y_pred_val_all[idx_valid]
        # label_val_all = label_val_all[idx_valid]

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
    syllableSeg_jan_madmom_original_basecode(0)