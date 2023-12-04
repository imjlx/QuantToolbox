#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summary.py
@Time    :   2023/11/27 17:47:00
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''
import pandas as pd
import numpy as np


def equal_capital(target: pd.DataFrame):
    """等资金交易所有target不为0的品种，无论target是正负

    Args:
        target (_type_): 输入target
    """
    
    new_target = pd.DataFrame(np.zeros_like(target), index=target.index, columns=target.columns)
    new_target[target != 0] = 1
    new_target = new_target.div(new_target.sum(axis=1), axis=0)
    
    return new_target


def section_clip(target: pd.DataFrame, pct_range: tuple = (0.9, 1)):
    """截面交易分位数在pct_range中的品种

    Args:
        target (pd.DataFrame): _description_
        pct_range (tuple): _description_
    """
    pct = target.rank(axis=1, pct=True, ascending=True)
    new_target = target.copy()
    new_target[pct <= pct_range[0]] = 0
    new_target[pct > pct_range[1]] = 0
    
    return new_target.fillna(0)