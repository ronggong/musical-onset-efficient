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

