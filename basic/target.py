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
import matplotlib.pyplot as plt
import seaborn as sns

# ================================= 因子处理 =================================

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
    # 用None代替0，不参与排序
    pct = target.replace(0, None).rank(axis=1, pct=True, ascending=True)
    new_target = target.copy()
    new_target[pct <= pct_range[0]] = 0
    new_target[pct > pct_range[1]] = 0
    
    return new_target.fillna(0)


# ================================= 因子分析 =================================


def plot_correlation(corr: pd.DataFrame):
    
    plt.figure(figsize=(10, 8))
    is_annot = True if len(corr) < 15 else False
    sns.heatmap(
        corr, cmap='coolwarm', vmin=-1, vmax=1, 
        annot=is_annot, fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    

def corr_analyse(data: pd.DataFrame, isPlotCorr=True, isPlotVIF=True) -> (pd.DataFrame, pd.Series):
    """对data中每一列值之间的相关性进行分析

    Args:
        data (pd.DataFrame): _description_
    """
    
    # 相关性分析
    corr = np.corrcoef(data.values, rowvar=False)   # 使用numpy的corrcoef计算，速度远高于pandas
    corr = pd.DataFrame(corr, index=data.columns, columns=data.columns)
    if isPlotCorr:
        plot_correlation(corr)
    # 输出相关性过大的参数对
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) # & (abs(corr) > thresh)
    row_inds, col_inds = np.where(mask)
    corr_results = pd.DataFrame({
        'var1': corr.columns[row_inds],
        'var2': corr.columns[col_inds],
        'corr': corr.values[row_inds, col_inds]
    })
    
    # VIF分析
    vif = pd.Series(np.linalg.inv(corr.to_numpy()).diagonal(), index=corr.columns, name='VIF')
    
    if isPlotVIF:
        plt.figure(figsize=(5, 4))
        vif.hist(bins=50)
        plt.title('VIF Hist')
    
    return corr_results, vif



# ================================= 因子比较 =================================

def intersection(targets: pd.DataFrame, pct_range=(0.9, 1)):
    df = df.groupby('date').rank(pct=True)
    df[df < 0.9] = 0
    df[df >= 0.9] = 1
    
    


