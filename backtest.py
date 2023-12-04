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

import QuantToolbox as qb


# ============================== 回测函数 ==============================

def basic(target: pd.DataFrame, price: pd.DataFrame = None, ret: pd.DataFrame = None, 
          limit: pd.DataFrame = None, fee_bps: float = 5):
    """矩阵运算实现的简单回测模块（wideform），回测结果不区分标的，返回一个向量
    Args:
        target (pd.DataFrame): 各标的的持仓
            Positive for buy and negative for sell. Abslute value for how much money to use(ratio).
        price (pd.DataFrame): 标的价格
        ret (pd.DataFrame): 标的日收益率，与价格二选一
        limit (pd.DataFrame): 交易限制，0代表正常，1代表涨停无法买入，-1代表跌停无法卖出
        fee_bps (float, optional): 双边交易费. Defaults to 5%%.
    Returns:
        _type_: _description_
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
    
    # return calculateInfoFromVec(pnl - fee, position, turnover, fee)
    return pnl - fee, turnover, fee


# ============================== 功能型回测函数 ==============================

def cs_lf(data: pd.DataFrame, target: str, ret: str = 'y1', 
          limit: str = 'ud_limit_h3', base: str = None, ofs_time=None,
          pct_range=(0.95, 1), fee_bps=5, isPlot=True, title=None):
    """对long form格式的数据进行回测，要求数据以‘date’、‘code’为index

    Args:
        data (pd.DataFrame): _description_
    """
    # 提取出需要的列
    data_ = data[[target, ret]].copy()
    if limit is not None:
        limit_file = limit
        data_[limit] = data[limit]
    if base is not None:
        data_[base] = data[base]
    data_ = data_.reset_index()
    
    # 转换为wide-form
    target = data_.pivot_table(index='date', columns='code', values=target)
    ret = data_.pivot_table(index='date', columns='code', values=ret)
    limit = data_.pivot_table(index='date', columns='code', values=limit) if limit is not None else None
    # 计算仓位
    target = qb.target.section_clip(target, pct_range=pct_range)
    target = qb.target.equal_capital(target)
    # 回测
    pnl, to, fee = qb.backtest.basic(target=target, ret=ret, fee_bps=fee_bps, limit=limit)
    
    base_dict = None
    if base is not None:
        # 如果有bench_mark数据，也进行回测
        base = data_.pivot_table(index='date', columns='code', values=base)
        base = qb.target.section_clip(base, pct_range=pct_range)
        base = qb.target.equal_capital(base)
        pnl_base, to_base, fee_base = qb.backtest.basic(target=base, ret=ret, fee_bps=fee_bps, limit=limit)
        base_dict = {'pnl': pnl_base, "turnover": to_base, 'fee': fee_base, 'position': base}
    
    if isPlot:
        btinfo = {'limit': limit_file, 'fee_bps': fee_bps, 'pct_range': pct_range}
        qb.summary.plot(pnl, to, fee, target, ofs_time=ofs_time, base_dict=base_dict, title=title, btinfo=btinfo)
    
    return pnl, to, fee, target

    
    
    





