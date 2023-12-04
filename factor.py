#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summary.py
@Time    :   2023/11/30 15:50:00
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(corr: pd.DataFrame):
    
    plt.figure(figsize=(10, 8))
    is_annot = True if len(corr) < 15 else False
    sns.heatmap(
        corr, cmap='coolwarm', vmin=-1, vmax=1, 
        annot=is_annot, fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    

def corr_analyse(data: pd.DataFrame, thresh=0.7, isPlotCorr=True, isPlotVIF=True):
    """对data中每一列值之间的相关性进行分析

    Args:
        data (pd.DataFrame): _description_
    """
    
    # 相关性分析
    corr = np.corrcoef(data.values, rowvar=False)   # 使用numpy的corrcoef计算，速度远高于pandas
    corr = pd.DataFrame(corr, index=data.columns, columns=data.columns)
    plot_correlation(corr)
    # 输出相关性过大的参数对
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) & (abs(corr) > thresh)
    row_inds, col_inds = np.where(mask)
    corr_results = pd.DataFrame({
        'Variable 1': corr.columns[row_inds],
        'Variable 2': corr.columns[col_inds],
        'Correlation': corr.values[row_inds, col_inds]
    })
    
    # VIF分析
    vif = pd.Series(np.linalg.inv(corr.to_numpy()).diagonal(), index=corr.columns, name='VIF')
    
    plt.figure(figsize=(5, 4))
    vif.hist(bins=50)
    plt.title('VIF Hist')
    
    return corr_results, vif