import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation_jingju_phrase import getTrainingFilenames
from data_preparation_CRNN import featureLabelSampleWeightsLoad
from data_preparation_CRNN import featureLabelSampleWeightsPad
from data_preparation_CRNN import createInputTensor
from data_preparation_CRNN import writeValLossCsv
from models_CRNN import jan_original
from sklearn.model_selection import ShuffleSplit
import pickle
import numpy as np
import random

from keras.losses import binary_crossentropy
from keras import backend as K

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


def loss_cal(fns, data_path, scaler, model, len_seq):
    """
    Calculate loss
    :param fns:
    :param data_path:
    :param scaler:
    :param model:
    :return:
    """
    y_pred_val_all = np.array([], dtype='float32')
    label_val_all = np.array([], dtype='int')

    for fn in fns:

        mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(data_path,
                                                                         fn,
                                                                         scaler)

        # pad sequence
        mfcc_line_pad, label_pad, sample_weights_pad, len_padded = \
            featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, len_seq)

        iter_time = len(mfcc_line_pad) / len_seq
        for ii_iter in range(iter_time):

            # create tensor from the padded line
            mfcc_line_tensor, label_tensor, _ = \
                createInputTensor(mfcc_line_pad, label_pad, sample_weights_pad, len_seq, ii_iter)

            y_pred = model.predict_on_batch(mfcc_line_tensor)

            # remove the padded samples
            if ii_iter == iter_time - 1 and len_padded > 0:
                y_pred = y_pred[:, :len_seq - len_padded, :]
                label_tensor = label_tensor[:, :len_seq - len_padded, :]

            # reduce the label dimension
            y_pred = y_pred.reshape((y_pred.shape[1],))
            label_tensor = label_tensor.reshape((label_tensor.shape[1],))

            y_pred_val_all = np.append(y_pred_val_all, y_pred)
            label_val_all = np.append(label_val_all, label_tensor)

    y_true = K.variable(label_val_all)
    y_pred = K.variable(y_pred_val_all)

    loss = K.eval(binary_crossentropy(y_true, y_pred))

    return loss


def run_training_process(path_input, path_output, ii, len_seq, bidi=True):

    file_path_model = os.path.join(path_output, 'bidi_lstms_'+ str(len_seq) + str(ii) + '.h5')
    file_path_log = os.path.join(path_output, 'bidi_lstms_'+ str(len_seq) + str(ii) + '.csv')

    jingju_feature_data_scratch_path = os.path.join(path_input, 'jingju_phrase')

    scaler = pickle.load(open(os.path.join(path_input, 'scaler_jingju_phrase.pkl'), 'r'))

    train_validation_fns = getTrainingFilenames(jingju_feature_data_scratch_path)

    # split the training set to train and validation sets
    train_fns, validation_fns = None, None
    rs = ShuffleSplit(n_splits=1, test_size=.1)
    for train_idx, validation_idx in rs.split(train_validation_fns):
        train_fns = [train_validation_fns[ti] for ti in train_idx]
        validation_fns = [train_validation_fns[vi] for vi in validation_idx]

    batch_size = 256
    nb_epochs = 500
    best_val_loss = 1.0  # initialize the val_loss
    counter = 0
    patience = 15   # early stopping patience
    overlap = 10  # overlap frames

    input_shape = (batch_size, len_seq, 1, 80, 15)

    # initialize the model
    model = jan_original(filter_density=1,
                         dropout=0.5,
                         input_shape=input_shape,
                         batchNorm=False,
                         dense_activation='sigmoid',
                         channel=1,
                         stateful=False,
                         bidi=bidi)

    model.summary()

    input_shape_val = (1, len_seq, 1, 80, 15)

    # initialize the model
    model_val = jan_original(filter_density=1,
                             dropout=0.5,
                             input_shape=input_shape_val,
                             batchNorm=False,
                             dense_activation='sigmoid',
                             channel=1,
                             stateful=False,
                             bidi=bidi)

    for ii_epoch in range(nb_epochs):

        batch_counter = 0

        # initialize the tensors
        mfcc_line_tensor = np.zeros(input_shape, dtype='float32')
        label_tensor = np.zeros((batch_size, len_seq, 1), dtype='int')
        sample_weights_tensor = np.zeros((batch_size, len_seq))

        # training
        for tfn in train_fns:

            mfcc_line, label, sample_weights = featureLabelSampleWeightsLoad(jingju_feature_data_scratch_path,
                                                                             tfn,
                                                                             scaler)

            mfcc_line_pad, label_pad, sample_weights_pad, _ = \
                featureLabelSampleWeightsPad(mfcc_line, label, sample_weights, overlap)

            for ii in range((len(mfcc_line_pad)-len_seq)/overlap+1):

                idx_start = ii*overlap
                idx_end = idx_start + len_seq

                mfcc_seg = mfcc_line_pad[idx_start:idx_end]
                label_seg = label_pad[idx_start:idx_end]
                sample_weights_seg = sample_weights_pad[idx_start:idx_end]

                # feed the tensor
                mfcc_line_tensor[batch_counter, :, 0, :, :] = mfcc_seg
                label_tensor[batch_counter, :, 0] = label_seg
                sample_weights_tensor[batch_counter, :] = sample_weights_seg

                if batch_counter >= batch_size - 1:
                    train_loss_batch, train_acc_batch = model.train_on_batch(mfcc_line_tensor,
                                                                             label_tensor,
                                                                             sample_weight=sample_weights_tensor)
                    batch_counter = 0
                else:
                    batch_counter += 1

        weights_trained = model.get_weights()
        model_val.set_weights(weights_trained)

        # calculate losses
        train_loss = loss_cal(train_fns, jingju_feature_data_scratch_path, scaler, model_val, len_seq)
        val_loss = loss_cal(validation_fns, jingju_feature_data_scratch_path, scaler, model_val, len_seq)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model.save_weights(file_path_model)
        else:
            counter += 1

        # write validation loss to csv
        writeValLossCsv(file_path_log, ii_epoch, val_loss, train_loss)

        # early stopping
        if counter >= patience:
            break

        random.shuffle(train_fns)

if __name__ == '__main__':
    run_training_process(0, 'ismir', bidi=True)