# -*- coding: utf-8 -*-

'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentation
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''

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

