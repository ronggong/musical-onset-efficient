from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, TimeDistributed
from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.python.client import device_lib


def jan_original(filter_density,
                 dropout,
                 input_shape,
                 batchNorm=False,
                 dense_activation='relu',
                 channel=1,
                 stateful=False,
                 training=True,
                 bidi=False):

    reshape_dim = input_shape
    channel_order = 'channels_first'

    model_1 = Sequential()

    device_type = device_lib.list_local_devices()[0].device_type

    if batchNorm:
        model_1.add(BatchNormalization(axis=1, input_shape=reshape_dim))

    model_1.add(TimeDistributed(Conv2D(int(10 * filter_density),
                                       (3, 7),
                                       padding="valid",
                                       data_format=channel_order,
                                       activation='relu'),
                                       batch_input_shape=reshape_dim))
    model_1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 1),
                                             padding='valid',
                                             data_format=channel_order)))

    model_1.add(TimeDistributed(Conv2D(int(20 * filter_density),
                                       (3, 3),
                                       padding="valid",
                                       data_format=channel_order,
                                       activation='relu')))
    model_1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 1),
                                             padding='valid',
                                             data_format=channel_order)))

    if dropout:
        model_1.add(Dropout(dropout))

    model_1.add(TimeDistributed(Flatten()))

    if bidi:
        if device_type == 'CPU':
            from keras.layers import LSTM
            model_1.add(Bidirectional(LSTM(30, stateful=stateful, return_sequences=True)))
        else:
            from keras.layers import CuDNNLSTM
            model_1.add(Bidirectional(CuDNNLSTM(30, stateful=stateful, return_sequences=True)))
    else:
        if device_type == 'CPU':
            from keras.layers import LSTM
            model_1.add(LSTM(60, stateful=stateful, return_sequences=True))
        else:
            from keras.layers import CuDNNLSTM
            model_1.add(CuDNNLSTM(60, stateful=stateful, return_sequences=True))

    if dropout:
        model_1.add(Dropout(dropout))

    model_1.add(TimeDistributed(Dense(1, activation='sigmoid')))

    optimizer = Adam()

    model_1.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'],
                    sample_weight_mode='temporal')

    # print(model_1.summary())

    return model_1

if __name__ == '__main__':
    jan_original(filter_density=1,
                 dropout=0.5,
                 input_shape=(1, 400, 1, 80, 15),
                 batchNorm=False,
                 dense_activation='relu',
                 channel=1,
                 stateful=False,
                 training=False,
                 bidi=True)