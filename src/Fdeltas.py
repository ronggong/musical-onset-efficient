# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import lfilter
# import matplotlib.pyplot as plt

def Fdeltas(x,w=9):
    '''
    # D = deltas(X,W)  Calculate the deltas (derivatives) of a sequence
    input feature*frame
    '''
    # nr feature, nc frame
    nr,nc = x.shape

    # Define window shape
    hlen = np.floor(w/2)
    w = 2*hlen + 1
    win = np.arange(hlen,-hlen-1,-1)

    # normalisation
    win = win/np.sum(np.abs(win))

    # pad data by repeating first and last columns
    xx = np.hstack((np.tile(x[:,0], (hlen,1)).transpose(), x, np.tile(x[:,-1], (hlen,1)).transpose()))
    #
    # plt.figure()
    # plt.pcolormesh(xx)
    # plt.show()

    # Apply the delta filter
    d = lfilter(win, 1, xx, 1)  # filter along dim 1 (rows)

    # Trim edges
    d = d[:,2*hlen:2*hlen+nc]

    return d

