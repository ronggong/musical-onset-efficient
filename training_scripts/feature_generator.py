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

    # print(len(filenames))
    # print(path_feature_data)
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
    # print(filenames)

    # test shuffle
    # filenames_copy, labels_copy = shuffleFilenamesLabelsInUnison(filenames_copy, labels_copy)
    # print(filenames_copy)
    # print(labels_copy)

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        X_batch = []
        # print(idx_start)
        # print(idx_end)
        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_tensor = labels_copy[idx_start:idx_end][index_sort]
        sample_weights_batch_tensor = sample_weights_copy[idx_start:idx_end][index_sort]

        # batch_size = len(y_batch_tensor)
        # X_batch_tensor = np.zeros((batch_size, 1, input_shape[0], input_shape[1]), dtype='float32')
        # print(batch_indices)
        # print(index_sort)
        # print(batch_indices[index_sort])
        if channel == 1:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort],:,:]
        else:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)


        # for ii, fn in enumerate(filenames_copy[idx_start:idx_end]):

            # print(fn)

            # path_feature_fn = os.path.join(path_feature, fn + '.pkl')
            # with gzip.open(fn, 'rb') as f:
            #     feature = cPickle.load(f)
            # feature = pickle.load(open(fn, "r"))
            # labels_block = labels[idx_start:idx_end, :]
            # preprocessing
            # feature = scaler.transform(feature)

            # print(feature.shape, y)

            # number of segments
            # seg = feature.shape[0] / float(input_shape[0])
            #
            # if seg > 1:
            #     feature_list, y = featureOnlineSegmentation(feature, y, input_shape)
            #
            #     for ii_f, f in enumerate(feature_list):
            #         X_batch.append(f)
            #         y_batch.append(y[ii_f])
            # elif seg == 1:
            # X_batch.append(feature)
            # y_batch.append(y)
            # X_batch_tensor[ii,0,:,:] = feature
            # y_batch_tensor = y_batch
            # we don't consider the case if feature dims is less than the batch dims

        # for ii in xrange(len(X_batch)):
        #     X_batch_tensor[ii] = np.expand_dims(X_batch[ii], axis=2)
        #     y_batch_tensor[ii, :] = y_batch[ii]

        # print(counter, X_batch_tensor.shape)
        counter += 1

        if sample_weights is not None:
            if multi_inputs:
                yield [X_batch_tensor,X_batch_tensor], y_batch_tensor

                # yield [X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor], y_batch_tensor, sample_weights_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor, sample_weights_batch_tensor
        else:
            if multi_inputs:
                yield [X_batch_tensor,X_batch_tensor], y_batch_tensor

                # yield [X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor], y_batch_tensor
            else:
                yield X_batch_tensor, y_batch_tensor

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                indices_copy, labels_copy, sample_weights_copy = shuffleFilenamesLabelsInUnison(indices_copy, labels_copy, sample_weights_copy)