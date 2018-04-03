import numpy as np
import h5py


def shuffleFilenamesLabelsInUnison(filenames, labels, sample_weights):
    assert len(filenames) == len(labels)
    assert len(filenames) == len(sample_weights)
    p=np.random.permutation(len(filenames))
    return filenames[p], labels[p], sample_weights[p]


def generator(path_feature_data,
              indices,
              number_of_batches,
              file_size,
              input_shape,
              labels=None,
              sample_weights=None,
              shuffle=True,
              multi_inputs=False,
              channel=1):

    f = h5py.File(path_feature_data, 'r')
    indices_copy = np.array(indices[:], np.int64)

    if labels is not None:
        labels_copy = np.copy(labels)
        # labels_copy = to_categorical(labels_copy)
    else:
        labels_copy = np.zeros((len(indices_copy), ))

    if sample_weights is not None:
        sample_weights_copy = np.copy(sample_weights)
    else:
        sample_weights_copy = np.ones((len(indices_copy), ))

    counter = 0

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        X_batch = []

        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_tensor = labels_copy[idx_start:idx_end][index_sort]
        sample_weights_batch_tensor = sample_weights_copy[idx_start:idx_end][index_sort]

        if channel == 1:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort],:,:]
        else:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)

        counter += 1

        if sample_weights is not None:
            if multi_inputs:
                yield [X_batch_tensor,X_batch_tensor], y_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor, sample_weights_batch_tensor
        else:
            if multi_inputs:
                yield [X_batch_tensor,X_batch_tensor], y_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                indices_copy, labels_copy, sample_weights_copy = shuffleFilenamesLabelsInUnison(indices_copy, labels_copy, sample_weights_copy)