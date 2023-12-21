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
from QuantToolbox.OLS import WeightedRidge


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


# ============================== 特殊结构回测函数 ==============================

def cs_lf(target: str|pd.Series, ret: str|pd.Series = 'y1', limit: str|pd.Series = 'ud_limit_h3', 
          base: str|pd.Series = None, data: pd.DataFrame = None, 
          ofs_time=None, pct_range=(0.9, 1), fee_bps=5, isPlot=True):
    
    """对long form格式的数据进行回测，要求数据以‘date’、‘code’为index
    """
    # 读取数据，转换为wide-form
    if isinstance(target, pd.Series):
        target = target.reset_index().pivot_table(index='date', columns='code', values=target.name)
    else:
        target = data.pivot_table(index='date', columns='code', values=target) 
    if isinstance(ret, pd.Series):
        ret = ret.reset_index().pivot_table(index='date', columns='code', values=ret.name)
    else:
        ret = data.pivot_table(index='date', columns='code', values=ret)
    if isinstance(limit, pd.Series):
        limit = limit.reset_index().pivot_table(index='date', columns='code', values=limit.name)
    else:
        limit = data.pivot_table(index='date', columns='code', values=limit) if limit is not None else None
    if isinstance(base, pd.Series):
        base = base.reset_index().pivot_table(index='date', columns='code', values=base.name)
    else:
        base = data.pivot_table(index='date', columns='code', values=base) if base is not None else None
    
    # 计算仓位
    target = qb.target.section_clip(target, pct_range=pct_range)
    target = qb.target.equal_capital(target)
    # 回测
    pnl, to, fee = qb.backtest.basic(target=target, ret=ret, fee_bps=fee_bps, limit=limit)
    
    if base is not None:
        # 如果有bench_mark数据，也进行回测
        base = qb.target.section_clip(base, pct_range=pct_range)
        base = qb.target.equal_capital(base)
        pnl_base, to_base, fee_base = qb.backtest.basic(target=base, ret=ret, fee_bps=fee_bps, limit=limit)
    
    if isPlot:
        plotter = qb.summary.Plotter(ofs_time=ofs_time)
        plotter.plot_pnl(pnl, to, fee, target, label='label_1')
        if base is not None:
            plotter.plot_pnl_base(pnl_base, to_base, fee_base, base)
        plotter.show()
        
    
    return pnl, to, fee, target


class TimeSeriesCrossValidation(object):
    def __init__(self, X, y, pipeline: Pipeline) -> None:
        self.X = X
        self.y = y
        self.pipeline = pipeline

    # 计算时序交叉验证的时间表
    def train_test_split(self, m_train=6, m_test=6):
        
        time_table = self.y.index.get_level_values(0).drop_duplicates()
        t_start, t_end = time_table.min(), time_table.max()

        d_test = datetime.timedelta(days=int(m_test * 30))
        d_train = datetime.timedelta(days=int(m_train * 30))
        
        # 计算测试集的个数
        nb_test = int(np.around((t_end - t_start - d_train).days / d_test.days))

        # 生成时间列表
        train_start = pd.date_range(start=t_start, end=t_end, freq=d_test)[:nb_test]
        time_table = [[t, t+d_train, t+d_train+d_test] for t in train_start]
        time_table[-1][-1] = t_end
        
        self.time_table = time_table
        
        return self.time_table
    
    def Predict(self):
        y_preds = []
        for t0, t1, t2 in tqdm(self.time_table):
            date_list = self.y.index.get_level_values(0)
            X_train = self.X[(date_list >= t0) & (date_list <= t1)]
            y_train = self.y[(date_list >= t0) & (date_list <= t1)]
            X_test = self.X[(date_list > t1) & (date_list <=t2)]
            
            self.pipeline.fit(X_train, y_train)
            
            if len(y_preds) == 0:
                y_pred = self.pipeline.predict(X_train)
                y_preds.append(pd.Series(y_pred, index=X_train.index, name='y_pred'))
            
            y_pred = self.pipeline.predict(X_test)
            y_preds.append(pd.Series(y_pred, index=X_test.index, name='y_pred'))
            
        self.y_pred = pd.concat(y_preds)
        return self.y_pred
    
    @staticmethod
    def pipeline_wrapper(pipeline: Pipeline, X_train, y_train, X_test, y_preds, i):
        
        pipeline.fit(X_train, y_train)
        y_preds[i] = pd.Series(pipeline.predict(X_test), index=X_test.index, name='y_pred')
        
    def Predict_(self):
        
        with mp.Manager() as manager:
            processes = []; y_preds = manager.dict()
            for i, (t0, t1, t2) in tqdm(enumerate(self.time_table), desc="Starting Multi-processes", total=len(self.time_table)):
                date_list = self.y.index.get_level_values(0)
                X_train = self.X[(date_list >= t0) & (date_list <= t1)]
                y_train = self.y[(date_list >= t0) & (date_list <= t1)]
                X_test = self.X[(date_list > t1) & (date_list <=t2)]
                
                if i == 0:
                    p = mp.Process(target=self.pipeline_wrapper, args=(self.pipeline, X_train, y_train, X_train, y_preds, -1))
                    p.start(); processes.append(p)
                
                p = mp.Process(target=self.pipeline_wrapper, args=(self.pipeline, X_train, y_train, X_test, y_preds, i))
                p.start(); processes.append(p)
            
            for p in tqdm(processes, desc="Waiting Multi-processes"):
                p.join()
            
            self.y_pred = [v for v in y_preds.values()]
            
        self.y_pred = pd.concat(self.y_pred).sort_index()
        return self.y_pred
    
    def Backtest(self, ret:pd.Series, limit:pd.Series=None, base:pd.Series=None, pct_range=(0.9, 1), label='pnl', onlyInfo=False):

        pnl1, to1, fee1, target1 = qb.backtest.cs_lf(
            target=self.y_pred, ret=ret, limit=limit, pct_range=pct_range, isPlot=False)
        
        if base is not None:
            pnl2, to2, fee2, target2 = qb.backtest.cs_lf(
                target=base, ret=ret, limit=limit, pct_range=pct_range, isPlot=False)
        
        if onlyInfo:
            ofs_time = self.time_table[0][1]
            _, summary_out = qb.summary.info(
                pnl1.loc[ofs_time:], to1.loc[ofs_time:], fee1.loc[ofs_time:], target1.loc[ofs_time:])
        else:
            summary1 = qb.summary.infos(pnl1, to1, fee1, target1, self.time_table)
            summary2 = qb.summary.infos(pnl2, to2, fee2, target2, self.time_table)
            
            plotter = qb.summary.Plotter(ofs_time=self.time_table[0][1])
            plotter.plot_pnl(pnl1, to1, fee1, target1, label=label)
            if base is not None:
                plotter.plot_pnl_base(pnl2, to2, fee2, target2)
            summary_out = plotter.plot_info(stat_result=qb.summary.stat(summary1, summary2))
            plotter.show(title=f"pct_range: {pct_range}, base: {base.name}")
        
        return summary_out
        

if __name__ == "__main__":
    pass
    importlib.reload(qb)
    # TimeSeriesCrossValidation用例
    
    print('读取数据...', end='\t'); start_time = time.time()
    data = pd.read_feather('data/data.feather')
    base = pd.read_feather('data/base.feather')
    print(f'{time.time() - start_time: .2f} s')
    
    summarys = []
    for k, t in itertools.product(np.arange(0.4, 4, 0.4), np.arange(2, 4, 0.4)):
        k = np.round(k, 2); t = np.round(t, 2)
        print(f"\nk={k} t={t}")
        pipeline = WeightedRidge(alpha=3E5, method=WeightedRidge.w_time_decay, args=(k, t))
        p = TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], pipeline=pipeline)
        p.train_test_split(m_train=36, m_test=1)
        p.Predict_()
        summary_out = p.Backtest(ret=data['y1'], limit=data['ud_limit_h3'], pct_range=(0.9, 1), label=f"k={k} t={t}", onlyInfo=True)
        summary_out = summary_out.to_frame().T.set_index(pd.MultiIndex.from_tuples(((k, t), ), names=['k', 't']))
        summarys.append(summary_out)
        
    pd.concat(summarys, axis=0).drop(columns='mdd_date').to_feather(f'data/TimeWeight/1.feather')
    
    # alpha = 3E5
    # m_train = 36
    # m_test = 1
    # k = 0
    # t = 0
    # pipeline = WeightedRidge(
    #         alpha=alpha, method=WeightedRidge.w_time_decay, args=(k, t)
    #         )
    # p = TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], pipeline=pipeline)
    # p.train_test_split(m_train=m_train, m_test=m_test)
    # p.Predict_()
    # summary_out = p.Backtest(ret=data['y1'], limit=data['ud_limit_h3'], base=base[f'base_{m_train}_{m_test}'], 
    #                          pct_range=(0.9, 1), label=f"alpha={alpha: .1e}")