# -*- coding: utf-8 -*-


import numpy as np
# import matplotlib.pyplot as plt

def Fprev_sub(x,w=2):
    '''
    # D = prev_sub(X,W) calculate the shifted x, with shifting frames 2
    input feature*frame
    '''
    # pad data by repeating first and last columns
    if w > 0:
        # shift to right
        xx = np.hstack((np.tile(x[:,0], (w,1)).transpose(), x[:,:-w]))
    if w < 0:
        # shift to left
        xx = np.hstack((x[:,-w:], np.tile(x[:,-1], (-w,1)).transpose()))
    if w==0:
        raise ValueError("shifting frame coef can't be 0.")

    # plt.figure()
    # plt.pcolormesh(xx)
    # plt.show()

    return xx

