#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summary.py
@Time    :   2023/09/28 16:40:32
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import datetime
import time
import itertools

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator

import multiprocessing as mp

import os; os.chdir('/home/jiahaoran/workspace')
import sys; sys.path.append('/home/jiahaoran/workspace')

import importlib
import QuantToolbox as qb


# ============================== 回测函数 ==============================

def basic(target: pd.DataFrame, price: pd.DataFrame = None, ret: pd.DataFrame = None, 
          limit: pd.DataFrame = None, fee_bps: float = 5):
    """
    矩阵运算实现的简单回测模块（wideform），回测结果不区分标的*
    输入数据为wideform，即每行一个日期，每列一个标的
    (*对于股票截面策略来说，确实不需要关注单标的回测情况；对于期货可能要另行考虑了)
    
    Args:
        target (pd.DataFrame): 仓位
        price (pd.DataFrame): 价格
        ret (pd.DataFrame): 对应的收益率，与价格二选一
        limit (pd.DataFrame): 交易限制，0代表正常，1代表涨停无法买入，-1代表跌停无法卖出
        fee_bps (float, optional): 双边交易费. Defaults to 5%%.
    Returns:
        return, turnover, fee
    """
    # 处理price与return
    ret = ret if ret is not None else price.pct_change()
    assert ret is not None, "One of price and ret must be specified"
    
    # 处理limit
    if limit is None:
        position = target   # 没有限制, 仓位就是target
    else:
        position = target.fillna(0)
        # 涨停后无法开仓，要把target置为0
        position[limit == 1] = 0
        # 特殊情况：昨天开了仓，第二天还是开仓但涨停，此时不需新开仓，虽然涨停但可延续昨日仓位
        position[(position.shift() > 0) & (target > 0) & (limit == 1)] = np.nan
        position = position.ffill()
        
        # 跌停后无法平仓，要延续前一天的信号（0保持0，1保持开仓）
        position[(position.shift() > 0) & (position == 0) & (limit == -1)] = np.nan
        position = position.ffill()
        
    # pnl with no fee
    pnl = (position * ret).sum(axis=1)
    
    # turnover
    position = position.fillna(0)
    trade = position.diff().abs()
    turnover = trade.sum(axis=1) / (position + position.shift()).sum(axis=1) * 2
    
    # fee
    fee = turnover * fee_bps * 1e-4
    
    return pnl - fee, turnover, fee


# ============================== 特殊结构回测函数 ==============================

def cross_section(target: pd.DataFrame, price: pd.DataFrame = None, ret: pd.DataFrame = None, 
                  limit: pd.DataFrame = None, fee_bps: float = 5, pct_range=(0.9, 1)):
    """截面分层回测

    Args:
        target (pd.DataFrame): 仓位
        price (pd.DataFrame, optional): 价格. Defaults to None.
        ret (pd.DataFrame, optional): 收益率. Defaults to None.
        limit (pd.DataFrame, optional): 涨跌停限制. Defaults to None.
        fee_bps (float, optional): 费率. Defaults to 5.
        pct_range (tuple, optional): 分层分位位置. Defaults to (0.9, 1).
    Returns:
        pnl, to, fee, target
    """
    # 根据输入的因子（纯因子、预测收益率等）计算仓位
    target = qb.target.section_clip(target, pct_range=pct_range)    # 裁出每个截面指定分位数的品种
    target = qb.target.equal_capital(target)    # 设为等权重
    
    # 回测
    pnl, to, fee = qb.backtest.basic(target=target, price=price, ret=ret, limit=limit, fee_bps=fee_bps)
    
    return pnl, to, fee


def cross_section_lf(target: str|pd.Series, price: str|pd.Series=None, ret: str|pd.Series = None, 
                     limit: str|pd.Series = None, data: pd.DataFrame = None, fee_bps=5, pct_range=(0.9, 1), 
                     base: str|pd.Series = None, ofs_time=None, isPlot=True):
    """截面分层回测（长格式)，要求数据以‘date’、‘code’为index

    Args:
        target (str | pd.Series): 仓位
        ret (str | pd.Series, optional): 收益率. Defaults to 'y1'.
        limit (str | pd.Series, optional): 涨跌停限制. Defaults to 'ud_limit_h3'.
        base (str | pd.Series, optional): _description_. Defaults to None.
        data (pd.DataFrame, optional): 以一个Dataframe传入多个信息. Defaults to None.
        ofs_time (_type_, optional): . Defaults to None.
        pct_range (tuple, optional): 分层分位位置. Defaults to (0.9, 1).
        fee_bps (int, optional): 费率. Defaults to 5.

    Returns:
        pnl, to, fee, target
    """
    # long-form 转换为 wide-form
    # target
    if isinstance(target, pd.Series):
        target = target.reset_index().pivot_table(index='date', columns='code', values=target.name)
    else:
        target = data.pivot_table(index='date', columns='code', values=target)
    # price
    if isinstance(price, pd.Series):
        price = price.reset_index().pivot_table(index='date', columns='code', values=price.name)
    else:
        price = data.pivot_table(index='date', columns='code', values=price) if price is not None else None
    # return
    if isinstance(ret, pd.Series):
        ret = ret.reset_index().pivot_table(index='date', columns='code', values=ret.name)
    else:
        ret = data.pivot_table(index='date', columns='code', values=ret) if ret is not None else None
    # limit
    if isinstance(limit, pd.Series):
        limit = limit.reset_index().pivot_table(index='date', columns='code', values=limit.name)
    else:
        limit = data.pivot_table(index='date', columns='code', values=limit) if limit is not None else None
    
    assert (price is not None) | (ret is not None)
    return cross_section(target=target, price=price, ret=ret, limit=limit, fee_bps=fee_bps, pct_range=pct_range)

if __name__ == "__main__":
    pass
    importlib.reload(qb)
    # TimeSeriesCrossValidation用例
    
    print('读取数据...', end='\t'); start_time = time.time()
    data = pd.read_feather('data/data.feather')
    base = pd.read_feather('data/base.feather')
    print(f'{time.time() - start_time: .2f} s')
    
    # summarys = []; fpath = f'data/TimeWeight/0.feather'
    # for k, t in itertools.product(np.arange(0.1, 4, 0.1), np.arange(0.1, 4, 0.1)):
    #     k = np.round(k, 1); t = np.round(t, 1)
    #     print(f"\nk={k} t={t}")
    #     pipeline = WeightedRidge(alpha=3E5, method=WeightedRidge.w_time_decay, args=(k, t))
    #     p = TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], pipeline=pipeline)
    #     p.train_test_split(m_train=36, m_test=1)
    #     p.Predict_()
    #     summary_out = p.Backtest(ret=data['y1'], limit=data['ud_limit_h3'], pct_range=(0.9, 1), label=f"k={k} t={t}", onlyInfo=True)
    #     summary_out = summary_out.to_frame().T.set_index(pd.MultiIndex.from_tuples(((k, t), ), names=['k', 't']))
    #     summarys.append(summary_out)
    #     pd.concat(summarys, axis=0).drop(columns='mdd_date').to_feather(f'data/TimeWeight/100.feather')
    
