#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tool.py
@Time    :   2023/12/04 09:49:00
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''

import os
import numpy as np
import pandas as pd


import sys; sys.path.append('/home/jiahaoran/workspace')


def train_test_split_basic(data_path='data/data.feather', split_date='2021-06-30'):
    data = pd.read_feather(data_path)
    data.reset_index(inplace=True)
    
    data_train = data[data['date'] <= split_date]
    data_train.set_index(['date', 'code'], inplace=True)
    x_train = data_train.loc[:, 'x1':'x160']
    y_train = data_train['y6']
    
    data_test = data[data['date'] > split_date]
    data_test.set_index(['date', 'code'], inplace=True)
    x_test = data_test.loc[:, 'x1':'x160']
    y_test = data_test['y6']
    
    return (data_train, x_train, y_train), (data_test, x_test, y_test)




if __name__ == "__main__":
    
    os.chdir('/home/jiahaoran/workspace')

