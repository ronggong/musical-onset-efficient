import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utilFunctions import getRecordings


def getTrainingFilenames(data_path):
    """
    training filenames
    :param data_path:
    :return:
    """
    fns = getRecordings(data_path)
    train_fns = [fn.replace('feature_', '') for fn in fns if 'feature' in fn]
    # train_fns = [fn.replace(fn.split('_')[0]+'_', '') for fn in fns]

    return train_fns
