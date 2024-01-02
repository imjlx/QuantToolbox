
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


class TimeSeriesCrossValidation(object):
    """时序交叉验证（滚动验证）, 对sklearn接口的模型进行分析
    sklearn.model_selection.TimeSeriesSplit（要求每行日期不同，且固定时间间隔，因而只能处理单标的的数据）
    本实现可以按照绝对时间间隔（如月、年）进行滚动
    """
    def __init__(self, X, y, model: Pipeline) -> None:
        self.X = X
        self.y = y
        self.model = model
        
        self.ofs_time = None
        self.y_pred = None
        
    def train_test_split(self, m_train=36, m_test=3):
        """计算时序交叉验证的时间表

        Args:
            m_train (int, optional): 训练数据长度（月）. Defaults to 36.
            m_test (int, optional): 测试数据长度、滚动长度（月）. Defaults to 3.

        Returns:
            list: [..., [训练集i开始时间, 分界i时间, 测试集i结束时间], ...]
        """
        time_table = self.y.index.get_level_values(0).drop_duplicates()
        t_start, t_end = time_table.min(), time_table.max()

        d_test = datetime.timedelta(days=int(m_test * 30))
        d_train = datetime.timedelta(days=int(m_train * 30))
        
        # 计算测试集的个数
        nb_test = int(np.around((t_end - t_start - d_train).days / d_test.days))

        # 生成时间列表
        train_start = pd.date_range(start=t_start, end=t_end, freq=d_test)[:nb_test]
        self.time_table = [[t, t+d_train, t+d_train+d_test] for t in train_start]
        self.time_table[-1][-1] = t_end
        
        self.ofs_time = self.time_table[0][1]   # 样本外时间是第一个训练集的结束时间
        
        return self.time_table
    
    def _predict_wrapper(self, timepoints, y_preds, i):
        """利用多进程进行调用时的函数
        """
        
        t0, t1, t2 = timepoints
        date_list = self.y.index.get_level_values(0)
        X_train = self.X[(date_list >= t0) & (date_list <= t1)]
        y_train = self.y[(date_list >= t0) & (date_list <= t1)]
        X_test = self.X[(date_list > t1) & (date_list <=t2)]
        
        self.model.fit(X_train, y_train)
        if i == 0:
            y_preds[-1] = pd.Series(self.model.predict(X_train), index=X_train.index, name='y_pred')
        y_preds[i] = pd.Series(self.model.predict(X_test), index=X_test.index, name='y_pred')
        
    def predict(self, isMulti=True, time_wait=0, verbose=True):
        """在model上滚动训练数据

        Args:
            isMulti (bool, optional): 是否启用多进程. Defaults to True.

        Returns:
            pd.Series: y_pred，包含全部训练集、测试集的预测
        """
        if isMulti: # 手动创建多进程
            with mp.Manager() as manager:
                processes = []; y_preds = manager.dict()
                loop = tqdm(enumerate(self.time_table), total=len(self.time_table), 
                            desc="starting Multi-processes") if verbose else enumerate(self.time_table)
                for i, timepoints in loop:
                    p = mp.Process(target=self._predict_wrapper, args=(timepoints, y_preds, i))
                    p.start(); processes.append(p)
                    time.sleep(time_wait)   # 创建进程后等待一下，防止进程太多爆内存
                
                loop = tqdm(processes, desc="Waiting Multi-processes") if verbose else processes
                for p in loop:
                    p.join()
                
                self.y_pred = [v for v in y_preds.values()]
                
            self.y_pred = pd.concat(self.y_pred).sort_index()
            
        else:   # 直接循环
            y_preds = []
            loop = tqdm(self.time_table) if verbose else self.time_table
            for t0, t1, t2 in tqdm(self.time_table):
                # 切割本周期的数据
                date_list = self.y.index.get_level_values(0)
                X_train = self.X[(date_list >= t0) & (date_list <= t1)]
                y_train = self.y[(date_list >= t0) & (date_list <= t1)]
                X_test = self.X[(date_list > t1) & (date_list <=t2)]
                
                # 训练模型
                self.model.fit(X_train, y_train)
                
                # 第一轮训练后，对训练集内数据进行预测，除此以外全部为样本外
                if len(y_preds) == 0:   
                    y_pred = self.model.predict(X_train)
                    y_preds.append(pd.Series(y_pred, index=X_train.index, name='y_pred'))
                # 预测测试集
                y_pred = self.model.predict(X_test)
                y_preds.append(pd.Series(y_pred, index=X_test.index, name='y_pred'))
            self.y_pred = pd.concat(y_preds)
            
        return self.y_pred
    
    def Backtest(self, price=None, ret='y1', limit='ud_limit_h3', base=None, 
                 data=None, pct_range=(0.9, 1), fee_bps=5, label='pnl', isPlot=True, title=None):
        # 回测
        pnl1, to1, fee1 = qb.backtest.cross_section_lf(
            target=self.y_pred, price=price, ret=ret, limit=limit, data=data, pct_range=pct_range, fee_bps=fee_bps)
        
        ofs_time = self.time_table[0][1]
        
        # 不画图，只计算样本外效果
        if not isPlot:
            summary_out = qb.summary.info(pnl1.loc[ofs_time:], to1.loc[ofs_time:], fee1.loc[ofs_time:])
        else:   # 画图
            plotter = qb.summary.Plotter(ofs_time=ofs_time)
            plotter.plot_pnl(pnl1, to1, fee1, label=label)
            
            stat_result = None
            if base is not None:    # 如果有基础数据，则画图对比
                pnl2, to2, fee2 = qb.backtest.cross_section_lf(
                    target=base, price=price, ret=ret, limit=limit, data=data, pct_range=pct_range, fee_bps=fee_bps)
                plotter.plot_pnl(pnl2, to2, fee2, isBase=True)
                
                summary1 = qb.summary.infos(pnl1, to1, fee1, self.time_table)
                summary2 = qb.summary.infos(pnl2, to2, fee2, self.time_table)
                stat_result = qb.summary.stat(summary1, summary2)
                
            summary_out = plotter.plot_info(stat_result=stat_result)
            plotter.show(title=f"pct_range: {pct_range}, base: {base.name}", suptitle=title)
        
        return summary_out


if __name__ == "__main__":

    data = pd.read_feather('data/data.feather')
    base = pd.read_feather('data/base.feather')
    print("完成读取数据")
    
    importlib.reload(qb)
    model = qb.OLS.WeightedRidge(alpha=3E5, method='time_decay', args=(0.1, 2))
    p = qb.validation.TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], model=model)
    p.train_test_split(m_train=36, m_test=1)
    p.predict(time_wait=0)
    summary_out = p.Backtest(data=data, base=base[f'base_36_1'], pct_range=(0.9, 1))


