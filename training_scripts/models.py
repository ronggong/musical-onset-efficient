from os import remove
from os.path import basename

import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Flatten, ELU, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

from feature_generator import generator
from data_preparation import load_data_bock
from data_preparation import load_data_jingju


def front_end_a(model_1,
                filter_density,
                reshape_dim,
                channel_order,
                dropout):
    model_1.add(Conv2D(int(10 * filter_density),
                       (3, 7),
                       padding="valid",
                       input_shape=reshape_dim,
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    model_1.add(Conv2D(int(20 * filter_density),
                       (3, 3),
                       padding="valid",
                       data_format=channel_order,
                       activation='relu'))
    model_1.add(MaxPooling2D(pool_size=(3, 1),
                             padding='valid',
                             data_format=channel_order))

    if dropout:
        model_1.add(Dropout(dropout))

    return model_1


def back_end_a(model_1, dense_activation, dropout):
    model_1.add(Dense(units=256, activation=dense_activation))

    if dropout:
        model_1.add(Dropout(dropout))
    return model_1


def back_end_c(model_1, filter_density, padding, channel_order, dropout):
    # replacement of the dense layer
    model_1.add(Conv2D(int(40 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(40 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(40 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(80 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(80 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(80 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(135 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Flatten())

    if dropout:
        model_1.add(Dropout(dropout))

    return model_1


def back_end_d(model_1, filter_density, padding, channel_order, dropout):
    # replacement of the dense layer
    model_1.add(Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Conv2D(int(60 * filter_density), (3, 3), padding=padding,
                       data_format=channel_order, activation='relu'))
    model_1.add(BatchNormalization(axis=1))

    model_1.add(Flatten())

    if dropout:
        model_1.add(Dropout(dropout))

    return model_1


def model_tail(model_1):
    model_1.add(Dense(1, activation='sigmoid'))

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    print(model_1.summary())
    return model_1


def jan_original(filter_density,
                 dropout,
                 input_shape,
                 batchNorm=False,
                 dense_activation='relu',
                 channel=1,
                 dense=True):
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1 = front_end_a(model_1=model_1,
                          filter_density=filter_density,
                          reshape_dim=reshape_dim,
                          channel_order=channel_order,
                          dropout=dropout)

    model_1.add(Flatten())

    if dense:
        model_1 = back_end_a(model_1=model_1,
                             dense_activation=dense_activation,
                             dropout=dropout)

    model_1 = model_tail(model_1=model_1)

    return model_1


def jan_original_9_layers_cnn(filter_density,
                              dropout,
                              input_shape,
                              batchNorm=False,
                              dense_activation='relu',
                              channel=1):
    """
    this model overfits too much
    :param filter_density:
    :param dropout:
    :param input_shape:
    :param batchNorm:
    :param dense_activation:
    :param channel:
    :return:
    """
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    padding = "same"

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1 = front_end_a(model_1=model_1,
                          filter_density=filter_density,
                          reshape_dim=reshape_dim,
                          channel_order=channel_order,
                          dropout=dropout)

    model_1 = back_end_c(model_1=model_1,
                         filter_density=filter_density,
                         padding=padding,
                         channel_order=channel_order,
                         dropout=dropout)

    model_1 = model_tail(model_1=model_1)

    return model_1


def jan_original_5_layers_cnn(filter_density,
                              dropout,
                              input_shape,
                              batchNorm=False,
                              dense_activation='relu',
                              channel=1):
    "less deep architecture"
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    padding = "same"

    model_1 = Sequential()

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1 = front_end_a(model_1=model_1,
                          filter_density=filter_density,
                          reshape_dim=reshape_dim,
                          channel_order=channel_order,
                          dropout=dropout)

    model_1 = back_end_d(model_1=model_1,
                         filter_density=filter_density,
                         padding=padding,
                         channel_order=channel_order,
                         dropout=dropout)

    model_1 = model_tail(model_1=model_1)

    return model_1


def jan_original_feature_extractor(model_pretrained,
                                   filter_density,
                                   dropout,
                                   input_shape,
                                   type='a',
                                   dense_activation='relu',
                                   channel=1):
    "less deep architecture"
    if channel == 1:
        reshape_dim = (1, input_shape[0], input_shape[1])
        channel_order = 'channels_first'
    else:
        reshape_dim = input_shape
        channel_order = 'channels_last'

    padding = "same"

    # print(model_pretrained.summary())

    # convert to functional model
    model_pretrained_func = model_pretrained.model

    # freeze pretrained model
    for layer in model_pretrained_func.layers:
        layer.trainable = False

    input = Input(shape=reshape_dim, name='input_new_1')

    x = Conv2D(int(10 * filter_density),
               (3, 7),
               padding="valid",
               data_format=channel_order,
               activation='relu',
               name='conv_1')(input)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order,
                     name='maxp_1')(x)

    x = Conv2D(int(20 * filter_density),
               (3, 3),
               padding="valid",
               data_format=channel_order,
               activation='relu',
               name='conv_2')(x)
    x = MaxPooling2D(pool_size=(3, 1),
                     padding='valid',
                     data_format=channel_order,
                     name='maxp_2')(x)

    if dropout:
        x = Dropout(dropout, name='drop_1')(x)

    if type == 'a':
        # feature extraction shallow
        x = concatenate([x, model_pretrained_func.get_layer('dropout_1').output], axis=1)

    # to maintain the filter number
    if type == 'a':
        filter_number_conv_3 = 54
        filter_number_conv_4 = 56
        filter_number_conv_5 = 56
    else:
        filter_number_conv_3 = 58
        filter_number_conv_4 = 58
        filter_number_conv_5 = 58

    # replacement of the dense layer
    x = Conv2D(int(filter_number_conv_3 * filter_density),
               (3, 3),
               padding=padding,
               data_format=channel_order,
               activation='relu',
               name='conv_3')(x)
    x = BatchNormalization(axis=1, name='bn_3')(x)

    x = Conv2D(int(filter_number_conv_4 * filter_density),
               (3, 3),
               padding=padding,
               data_format=channel_order,
               activation='relu',
               name='conv_4')(x)
    x = BatchNormalization(axis=1, name='bn_4')(x)

    x = Conv2D(int(filter_number_conv_5 * filter_density),
               (3, 3),
               padding=padding,
               data_format=channel_order,
               activation='relu',
               name='conv_5')(x)
    x = BatchNormalization(axis=1, name='bn_5')(x)

    x = Flatten(name='flatten_new_1')(x)

    if dropout:
        x = Dropout(dropout, name='drop_2')(x)

    if type == 'b':
        x = concatenate([x, model_pretrained_func.get_layer('dropout_2').output])

    outputs = Dense(1, activation='sigmoid', name='dense_new_1')(x)

    model_1 = Model(inputs=[input, model_pretrained_func.input], outputs=outputs)

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    print(model_1.summary())

    return model_1


def createModel_schluter(input,
                         num_filter,
                         height_filter,
                         width_filter,
                         filter_density,
                         pool_n_row,
                         pool_n_col,
                         dropout):
    """
    original Schluter relu activation, no dropout
    :param input:
    :param num_filter:
    :param height_filter:
    :param width_filter:
    :param filter_density:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :return:
    """

    x = Conv2D(int(num_filter * filter_density), (height_filter, width_filter), padding="same",
                       data_format="channels_first",
                       activation='relu')(input)

    output_shape = K.int_shape(x)

    if pool_n_row == 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], output_shape[3]), padding='same', data_format="channels_first")(x)
    elif pool_n_row == 'all' and pool_n_col != 'all':
        x = MaxPooling2D(pool_size=(output_shape[2], pool_n_col), padding='same', data_format="channels_first")(x)
    elif pool_n_row != 'all' and pool_n_col == 'all':
        x = MaxPooling2D(pool_size=(pool_n_row, output_shape[3]), padding='same', data_format="channels_first")(x)
    else:
        x = MaxPooling2D(pool_size=(pool_n_row, pool_n_col), padding='same', data_format="channels_first")(x)
    #    model.add(Flatten())
    return x


def temporal_layer_schluter(filter_density_layer1, pool_n_row, pool_n_col, dropout, input_dim):
    reshape_dim = (1, input_dim[0], input_dim[1])

    input = Input(shape=reshape_dim)

    x_1 = createModel_schluter(input, 12, 1, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_2 = createModel_schluter(input, 6, 3, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_3 = createModel_schluter(input, 3, 5, 7, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_4 = createModel_schluter(input, 12, 1, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_5 = createModel_schluter(input, 6, 3, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    x_6 = createModel_schluter(input, 3, 5, 12, filter_density_layer1, pool_n_row, pool_n_col,
                          dropout)

    merged = concatenate([x_1, x_2, x_3, x_4, x_5, x_6], axis=1)

    return input, merged


def model_layer2_schluter(input, merged, filter_density_layer2, dropout):
    """
    original bock layer
    :param input:
    :param merged:
    :param filter_density_layer2:
    :param dropout:
    :return:
    """

    # conv 2 layers
    merged = Conv2D(int(20 * filter_density_layer2),
                    (3, 3),
                    padding="valid",
                    data_format="channels_first",
                    activation='relu')(merged)

    merged = MaxPooling2D(pool_size=(3, 1),
                          padding='valid',
                          data_format="channels_first")(merged)

    merged = Dropout(dropout)(merged)
    merged = Flatten()(merged)

    # dense
    merged = Dense(units=256, activation='sigmoid')(merged)
    merged = Dropout(dropout)(merged)

    merged = Dense(1, activation='sigmoid')(merged)
    model_merged = Model(inputs=input, outputs=merged)

    optimizer = Adam()

    model_merged.compile(loss='binary_crossentropy',
                         optimizer= optimizer,
                         metrics=['accuracy'])

    print(model_merged.summary())

    return model_merged


def jordi_model_schluter(filter_density_1,
                         filter_density_2,
                         pool_n_row,
                         pool_n_col,
                         dropout,
                         input_shape,
                         dim='timbral'):
    """
    Schluter model configuration
    :param filter_density_1:
    :param filter_density_2:
    :param pool_n_row:
    :param pool_n_col:
    :param dropout:
    :param input_shape:
    :param dim:
    :return:
    """

    inputs, merged = temporal_layer_schluter(filter_density_layer1=filter_density_1,
                                             pool_n_row=pool_n_row,
                                             pool_n_col=pool_n_col,
                                             dropout=dropout,
                                             input_dim=input_shape)

    model = model_layer2_schluter(input=inputs,
                                  merged=merged,
                                  filter_density_layer2=filter_density_2,
                                  dropout=dropout)

    return model


def model_train(model_0, batch_size, patience, input_shape,
                path_feature_data,
                indices_train, Y_train, sample_weights_train,
                indices_validation, Y_validation, sample_weights_validation,
                indices_all, Y_train_validation, sample_weights, class_weights,
                file_path_model, filename_log):

    """
    train the model with validation early stopping and retrain the model with whole training dataset
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param sample_weights_train:
    :param indices_validation:
    :param Y_validation:
    :param sample_weights_validation:
    :param indices_all:
    :param Y_train_validation:
    :param sample_weights:
    :param class_weights:
    :param file_path_model:
    :param filename_log:
    :return:
    """

    model_0.save_weights(basename(file_path_model))

    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]
    print("start training...")

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels=Y_train,
                                sample_weights=sample_weights_train,
                                multi_inputs=False)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels=Y_validation,
                              sample_weights=sample_weights_validation,
                              multi_inputs=False)

    history = model_0.fit_generator(generator=generator_train,
                                    steps_per_epoch=steps_per_epoch_train,
                                    epochs=500,
                                    validation_data=generator_val,
                                    validation_steps=steps_per_epoch_val,
                                    class_weight=class_weights,
                                    callbacks=callbacks,
                                    verbose=2)

    model_0.load_weights(basename(file_path_model))

    # train again use all train and validation set
    epochs_final = len(history.history['val_loss'])
    # epochs_final = 100

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    input_shape=input_shape,
                                    labels=Y_train_validation,
                                    sample_weights=sample_weights,
                                    multi_inputs=False)

    model_0.fit_generator(generator=generator_train_val,
                          steps_per_epoch=steps_per_epoch_train_val,
                          epochs=epochs_final,
                          class_weight=class_weights,
                          verbose=2)

    model_0.save(file_path_model)
    remove(basename(file_path_model))


# class MomentumScheduler(Callback):
#     '''Momentum scheduler.
#     # Arguments
#         schedule: a function that takes an epoch index as input
#             (integer, indexed from 0) and returns a new
#             momentum as output (float).
#     '''
#     def __init__(self, schedule):
#         super(MomentumScheduler, self).__init__()
#         self.schedule = schedule
#
#     def on_epoch_begin(self, epoch, logs={}):
#         assert hasattr(self.model.optimizer, 'momentum'), \
#             'Optimizer must have a "momentum" attribute.'
#         mmtm = self.schedule(epoch)
#         assert type(mmtm) == float, 'The output of the "schedule" function should be float.'
#         K.set_value(self.model.optimizer.momentum, mmtm)


def momentumIncrease(epoch):
    """
    increase momentum linearly from 0.45 to 0.9 between epoch 10 and 20
    :param epoch:
    :return:
    """
    if epoch <= 9:
        mmtm = 0.45
    elif epoch > 9 and epoch < 20:
        mmtm = 0.45 + 0.045*(epoch-9)
    else:
        mmtm = 0.9

    # print('epoch:', epoch, 'momentum:', mmtm)

    return mmtm


def lrDecrease(epoch):
    """
    decrease learning rate each epoch by a coefficient 0.995
    :param epoch:
    :return:
    """
    lr = np.power(0.995, epoch)
    # print('epoch:', epoch, 'Learning rate:', lr)
    return lr


def model_train_schluter(model_0,
                         batch_size,
                         input_shape,
                         path_feature_data,
                         indices_all,
                         Y_train_validation,
                         sample_weights,
                         class_weights,
                         file_path_model,
                         filename_log,
                         channel):

    # mmtm = MomentumScheduler(momentumIncrease)
    # lrSchedule = LearningRateScheduler(lrDecrease)
    # callbacks = [mmtm, lrSchedule, CSVLogger(filename=filename_log, separator=';')]
    callbacks = [CSVLogger(filename=filename_log, separator=';')]
    print("start training...")

    # train again use all train and validation set
    epochs_final = 100

    steps_per_epoch_train_val = int(np.ceil(len(indices_all) / batch_size))

    generator_train_val = generator(path_feature_data=path_feature_data,
                                    indices=indices_all,
                                    number_of_batches=steps_per_epoch_train_val,
                                    file_size=batch_size,
                                    input_shape=input_shape,
                                    labels=Y_train_validation,
                                    sample_weights=sample_weights,
                                    multi_inputs=False,
                                    channel=channel)

    model_0.fit_generator(generator=generator_train_val,
                          steps_per_epoch=steps_per_epoch_train_val,
                          epochs=epochs_final,
                          callbacks=callbacks,
                          # class_weight=class_weights,
                          verbose=2)

    model_0.save(file_path_model)
    # remove(basename(file_path_model))


def model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           path_feature_data,
                           indices_train,
                           Y_train,
                           sample_weights_train,
                           indices_validation,
                           Y_validation,
                           sample_weights_validation,
                           class_weights,
                           file_path_model,
                           filename_log,
                           channel,
                           multi_inputs=False):

    """
    train the model with validation early stopping and retrain the model with whole training dataset
    :param model_0:
    :param batch_size:
    :param patience:
    :param input_shape:
    :param path_feature_data:
    :param indices_train:
    :param Y_train:
    :param sample_weights_train:
    :param indices_validation:
    :param Y_validation:
    :param sample_weights_validation:
    :param indices_all:
    :param Y_train_validation:
    :param sample_weights:
    :param class_weights:
    :param file_path_model:
    :param filename_log:
    :return:
    """

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]
    print("start training with validation...")

    steps_per_epoch_train = int(np.ceil(len(indices_train) / batch_size))
    steps_per_epoch_val = int(np.ceil(len(indices_validation) / batch_size))

    generator_train = generator(path_feature_data=path_feature_data,
                                indices=indices_train,
                                number_of_batches=steps_per_epoch_train,
                                file_size=batch_size,
                                input_shape=input_shape,
                                labels=Y_train,
                                sample_weights=sample_weights_train,
                                multi_inputs=multi_inputs,
                                channel=channel)
    generator_val = generator(path_feature_data=path_feature_data,
                              indices=indices_validation,
                              number_of_batches=steps_per_epoch_val,
                              file_size=batch_size,
                              input_shape=input_shape,
                              labels=Y_validation,
                              sample_weights=sample_weights_validation,
                              multi_inputs=multi_inputs,
                              channel=channel)

    model_0.fit_generator(generator=generator_train,
                          steps_per_epoch=steps_per_epoch_train,
                          epochs=500,
                          validation_data=generator_val,
                          validation_steps=steps_per_epoch_val,
                          callbacks=callbacks,
                          verbose=2)


def model_switcher(model_name,
                   filter_density,
                   dropout,
                   input_shape,
                   channel,
                   deep,
                   dense,
                   activation_dense,
                   model_pretrained=None):

    if model_name == 'jan_original':
        if deep == '5_layers_cnn':
            model_0 = jan_original_5_layers_cnn(filter_density=filter_density,
                                                dropout=dropout,
                                                input_shape=input_shape,
                                                batchNorm=False,
                                                dense_activation=activation_dense,
                                                channel=channel)
        elif deep == '9_layers_cnn':
            model_0 = jan_original_9_layers_cnn(filter_density=filter_density,
                                                dropout=dropout,
                                                input_shape=input_shape,
                                                batchNorm=False,
                                                dense_activation=activation_dense,
                                                channel=channel)
        else:
            model_0 = jan_original(filter_density=filter_density,
                                   dropout=dropout,
                                   input_shape=input_shape,
                                   batchNorm=False,
                                   dense_activation=activation_dense,
                                   channel=channel,
                                   dense=dense)
    elif model_name == 'feature_extractor_a':
        model_0 = jan_original_feature_extractor(model_pretrained=model_pretrained,
                                                 filter_density=filter_density,
                                                 dropout=dropout,
                                                 input_shape=input_shape,
                                                 type='a',
                                                 dense_activation=activation_dense,
                                                 channel=channel)
    elif model_name == 'feature_extractor_b':
        model_0 = jan_original_feature_extractor(model_pretrained=model_pretrained,
                                                 filter_density=filter_density,
                                                 dropout=dropout,
                                                 input_shape=input_shape,
                                                 type='b',
                                                 dense_activation=activation_dense,
                                                 channel=channel)
    elif model_name == 'jordi_temporal_schluter':
        model_0 = jordi_model_schluter(filter_density_1=2,
                                       filter_density_2=filter_density,
                                       pool_n_row=5,  # old 3
                                       pool_n_col=1,  # old 5
                                       dropout=dropout,
                                       input_shape=input_shape,
                                       dim='temporal')
    else:
        model_0 = None

    return model_0


def train_model(filename_train_validation_set,
                filename_labels_train_validation_set,
                filename_sample_weights,
                filter_density,
                dropout,
                input_shape,
                file_path_model,
                filename_log,
                model_name='jan_original',
                channel=1):
    """
    train final model save to model path
    """

    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data_bock(filename_labels_train_validation_set,
                       filename_sample_weights)

    model_0 = model_switcher(model_name,filter_density,dropout,input_shape,channel)

    batch_size = 256

    # print(model_0.count_params())

    model_train_schluter(model_0,
                         batch_size,
                         input_shape,
                         filename_train_validation_set,
                         filenames_features,
                         Y_train_validation,
                         sample_weights,
                         class_weights,
                         file_path_model,
                         filename_log,
                         channel)


def train_model_validation(filename_train_validation_set,
                           filename_labels_train_validation_set,
                           filename_sample_weights,
                           filter_density,
                           dropout,
                           input_shape,
                           file_path_model,
                           filename_log,
                           model_name='jan_original',
                           deep='nodeep',
                           activation_dense='sigmoid',
                           dense=True,
                           channel=1):
    """
    train model with validation
    """

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data_jingju(filename_labels_train_validation_set,
                         filename_sample_weights)

    model_0 = model_switcher(model_name=model_name,
                             filter_density=filter_density,
                             dropout=dropout,
                             input_shape=input_shape,
                             channel=channel,
                             activation_dense=activation_dense,
                             deep=deep,
                             dense=dense)

    batch_size = 256
    patience = 15

    model_train_validation(model_0,
                           batch_size,
                           patience,
                           input_shape,
                           filename_train_validation_set,
                           filenames_train, Y_train, sample_weights_train,
                           filenames_validation, Y_validation, sample_weights_validation,
                           class_weights,
                           file_path_model,
                           filename_log,
                           channel)


def finetune_model_validation(filename_train_validation_set,
                              filename_labels_train_validation_set,
                              filename_sample_weights,
                              filter_density,
                              dropout,
                              input_shape,
                              file_path_model,
                              filename_log,
                              model_name,
                              path_model,
                              deep='5_layers_cnn',
                              dense=True,
                              channel=1):
    """
    train model with validation
    """

    filenames_train, Y_train, sample_weights_train, \
    filenames_validation, Y_validation, sample_weights_validation, \
    filenames_features, Y_train_validation, sample_weights, class_weights = \
        load_data_jingju(filename_labels_train_validation_set,
                         filename_sample_weights)

    # load pretrained model
    model_pretrained = load_model(filepath=path_model)

    if model_name == 'retrained':
        model = model_pretrained
        multi_inputs = False
    else:
        model = model_switcher(model_name=model_name,
                               filter_density=filter_density,
                               dropout=dropout,
                               input_shape=input_shape,
                               channel=channel,
                               deep=deep,
                               dense=dense,
                               model_pretrained=model_pretrained,
                               activation_dense='sigmoid')
        multi_inputs = True

    batch_size = 256
    patience = 15

    model_train_validation(model,
                           batch_size,
                           patience,
                           input_shape,
                           filename_train_validation_set,
                           filenames_train, Y_train, sample_weights_train,
                           filenames_validation, Y_validation, sample_weights_validation,
                           class_weights,
                           file_path_model,
                           filename_log,
                           channel,
                           multi_inputs=multi_inputs)

