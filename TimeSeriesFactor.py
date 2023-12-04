#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tool.py
@Time    :   2023/08/22 11:17:59
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''

import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List

from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import itertools

import random
from simanneal import Annealer
from scipy.optimize import dual_annealing


# ========================================== 因子回测框架 ==========================================

class FactorTesterBase(object):
    """
    因子调试框架。类中函数只共享功能参数，不共享待调参数和其他变量。
    因子核心定义在GetTarget中，初始化中定义功能选项参数，Execute进行最终的调用。
    """
    
    def __init__(self, freq=None, future=None, stdate=None, eddate=None, **kwargs):
        """
        :param int|str freq: 基础数据频率, defaults to 'd'
        :param int stdate: 回测起始日期, defaults to 数据起始日期
        :param int eddate: 回测结束日期, defaults to 最新日期
        :param str|list future: 回测标的物, defaults to 'if'
        """
        self.docker = None
        self.sigData = None
        self.main_line = None
        self.up_bound = {}
        self.dn_bound = {}
        self.sub1 = {}
        self.sub2 = {}
        
        # 确定基础品种、数据频率
        self.future = ['if'] if future is None else [future] if isinstance(future, str) else future
        self.freq = 'd' if freq is None else freq
        
        # 默认回测日期
        self.stdate = self.GetData('trading_day').dropna(how='all').iloc[1].dropna()[0] if stdate is None else stdate
        self.eddate = self.GetData('trading_day').dropna(how='all').iloc[-1].dropna()[0] if eddate is None else eddate
        
        # 是否画图
        self.isPlotRet = False if 'isPlotRet' not in kwargs else kwargs['isPlotRet']
        self.isPlotSig = False if 'isPlotSig' not in kwargs else kwargs['isPlotSig']
        
        self.side = 0 if 'side' not in kwargs else kwargs['side']
        assert self.side in [-1, 0, 1]
        
        self.isCutPnl = False if 'isCutPnl' not in kwargs else kwargs['isCutPnl']
        self.isPlotSig = False if 'isPlotSig' not in kwargs else kwargs['isPlotSig']
    
    def GetData(self, field: Union[str, list], future: str = None) -> List[pd.DataFrame]:
        # 生成数据docker
        self.docker = qs.data.quick(comrange='all', UseCom=self.future, hotname='cushot', layer=4) if self.docker is None else self.docker
        datas = []
        for f in field if isinstance(field, list) else [field]:
            data = self.docker[self.freq, f]
            data = data[future] if future is not None else data
            datas.append(data)
        return datas[0] if isinstance(field, str) else datas
        
    def GetTarget(self):
        pass
    
    def Backtest(self, target):
        open, trading_day, limit = self.GetData(['open', 'trading_day', 'limit'])
        
        target = qs.target.equal_capital(target)
        pnlall, toall, mvall = qs.stat.quick(target.shift(1), open, trading_day, limit=limit)

        if self.isPlotRet:
            qs.summary.plot(pnlall[self.future], toall[self.future], mvall[self.future], target=target[self.future], 
                            # fp=open.pct_change(1).shift(-1)[self.future], 
                            ploteddate=self.eddate, plotstdate=self.stdate)

        return pnlall, toall, mvall
            
    def PlotSignal(self, isPlot=False, future=None, main_line: dict = None, trade_point=None, 
                   up_bound={}, dn_bound={}, sub1 = {}, sub2 = {}):
        if not self.isPlotSig:
            return None
        
        future = future if future is not None else self.future[0]
        open, close, high, low, volume, trading_day = self.GetData(['open', 'close', 'high', 'low', 'volume', 'trading_day'], future)
        
        if self.sigData is None:
            # 输入基础数据
            self.sigData = pd.concat([close, high, low, open, trading_day, volume], axis=1)
            self.sigData.columns = ['close', 'high', 'low', 'open', 'trading_day', 'volume']
            self.sigData = self.sigData.dropna().shift(-1)
        
        # 输入指定数据
        if trade_point is not None:
            self.sigData['trade_point'] = trade_point[future].reset_index(drop=True).shift(1)
        if main_line is not None:
            for key in main_line.keys():
                self.sigData[key] = main_line[key][future].reset_index(drop=True)
        
        for key in up_bound.keys():
            self.sigData[key] = up_bound[key][future].reset_index(drop=True).shift(1)
        for key in dn_bound.keys():
            self.sigData[key] = dn_bound[key][future].reset_index(drop=True).shift(1)
        
        for key in sub1.keys():
            self.sigData[key] = sub1[key][future].reset_index(drop=True).shift(1)
        for key in sub2.keys():
            self.sigData[key] = sub2[key][future].reset_index(drop=True).shift(1)
            

        self.main_line = main_line if self.main_line is None else self.main_line
        self.up_bound = up_bound if up_bound != {} else self.up_bound
        self.dn_bound = dn_bound if dn_bound != {} else self.dn_bound
        self.sub1 = sub1 if sub1 != {} else self.sub1
        self.sub2 = sub2 if sub2 != {} else self.sub2
        
        if isPlot:  # 不画图时只传入数据。
            # 填充空缺项为默认值
            cols = self.sigData.columns
            self.sigData['trade_point'] = None if 'trade_point' not in cols else self.sigData['trade_point']
            self.sigData = self.sigData[(self.sigData['trading_day'] > self.stdate) * (self.sigData['trading_day'] < self.eddate)]
            
            if 'pnl' in self.sigData.columns:   # 归零pnl
                self.sigData['pnl'] = (self.sigData['pnl'].cumsum() - self.sigData['pnl'].cumsum().iloc[0]).shift(0) * 100

            qs.plot.signal(
                self.sigData, main_line=list(self.main_line.keys()), trade_point=['trade_point'],
                up_bound=list(self.up_bound.keys()), dn_bound=list(self.dn_bound.keys()), 
                sub1=list(self.sub1.keys()), sub2=list(self.sub2.keys()))
    
    def Execute(self, **kwargs):
        if 'isPlotRet' in kwargs:
            self.isPlotRet = kwargs['isPlotRet']
        if 'isPlotSig' in kwargs:
            self.isPlotSig = kwargs['isPlotSig']
        if 'plotstdate' in kwargs:
            self.stdate = kwargs['plotstdate']
        if 'ploteddate' in kwargs:
            self.eddate = kwargs['ploteddate']
            
        if 'side' in kwargs:
            self.side = kwargs['side']
        if 'plotfuture' in kwargs:
            self.plotfuture = kwargs['plotfuture']
        if 'isCutPnl' in kwargs:
            self.isCutPnl = kwargs['isCutPnl']
    
    @staticmethod
    def CalculateInfo(pnlall: pd.DataFrame, toall: pd.DataFrame, mvall: pd.DataFrame, useQSim=True):
        pnl = pnlall.sum(axis=1)
        
        if useQSim:
            Sharpe, MaxDD, calmar, annual_ret = qs.summary.info(pnl.values)
        else:
            # 夏普
            Sharpe = (pnl.mean() - 0) / pnl.std() * np.sqrt(250)
            Sharpe = np.around(Sharpe, 2)
            # 年化收益率
            annual_ret = pnl.mean() * 252
            annual_ret = np.around(annual_ret * 100, 2)
            # 最大回撤
            ret_cum = (pnl + 1).cumprod(); ret_max = ret_cum.cummax()
            drawdown = (ret_cum - ret_max) / ret_max
            MaxDD = np.around(-drawdown.min(), 2)
            # Calmar 
            calmar = np.around(np.abs(annual_ret / MaxDD), 2)
               
        # 最大回撤
        ret_cum = (pnl + 1).cumprod()
        ret_max = ret_cum.cummax()
        drawdown = (ret_cum - ret_max) / ret_max
        dd_ed = drawdown.argmin()
        if len(ret_max[: dd_ed]) == 0:
            print(dd_ed = datetime.datetime.strptime(str(pnl.index[dd_ed]), "%Y%m%d"))
        dd_st = ret_max[: dd_ed].argmax()
        dd_ed = datetime.datetime.strptime(str(pnl.index[dd_ed]), "%Y%m%d")
        dd_st = datetime.datetime.strptime(str(pnl.index[dd_st]), "%Y%m%d")
        dd_duration = (dd_ed - dd_st).days
            
        # 资金利用/杠杆
        leverage = mvall.abs().sum().sum() / len(mvall['if'].dropna())
        leverage = np.around(leverage, 2)
        # 换手率
        turnover = toall.abs().sum().sum() / len(toall['if'].dropna())
        turnover = np.around(turnover, 2)
        
        return Sharpe, MaxDD, calmar, annual_ret, leverage, turnover, dd_st, dd_ed, dd_duration
        
# ========================================== 调参工具 ==========================================

def search_param(func: callable, args: list, args_name=None, desc="No Desc", **kwargs):
    """
    遍历调参工具，利用multiprocessing加速。

    :param callable func: 回测函数，要求返回前三个参数为pnlall, toall, mvall
    :param list args: 传入func的参数值列表，可为单一值或范围，不同参数间枚举组合。
    :param list args_name: 传入func的参数描述, defaults to None
    """
    # 将回测的函数与分析结果的函数复合，便于并行
    func = partial(func, **kwargs)
    def func_combine(args_):
        pnlall, toall, mvall, _ = func(*args_)
        return FactorTesterBaseDaily.CalculateInfo(pnlall, toall, mvall)
    
    args_name = ["x"+str(i+1) for i in range(len(args))]
    
    isIter = [] # 处理输入的待遍历参数
    for i in range(len(args)):
        if type(args[i]) == int or type(args[i]) == float:
            args[i] = [args[i]]
            isIter.append(False)
        else:
            isIter.append(True)
            
    # 根据迭代参数个数的不同，进行不同处理
    if sum(isIter) == 0:    # 全部参数为单一值，直接绘图输出
        args = [arg[0] for arg in args]
        func(*args)
        return None
    else:   # 需要对参数进行遍历
        # 并行回测
        args_iter = np.array(list(itertools.product(*args)))
        pool = Pool(os.cpu_count()-8)
        results_raw = list(tqdm(pool.imap(func_combine, tuple(args_iter)), total=len(args_iter), desc=desc))
        results_raw = pd.DataFrame(
            results_raw, index=pd.MultiIndex.from_product(args, names=args_name), 
            columns=['SR', "MaxDD", "Calmar", 'Ret', 'Leg', 'TO', 'DD_st', 'DD_ed', 'DDD'])

        if sum(isIter) == 1:    # 若只有一个优化参数，直接画效果曲线图
            results = results_raw.droplevel(level=[idx for idx, value in enumerate(isIter) if not value])
            idx_max = results['SR'].nlargest().index
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), layout='tight', sharex='all')
            for col, ax in zip(['SR', "MaxDD", "Calmar", 'Ret'], axs):
                ax.plot(results.index, results[col])
                for i in range(0, 3):
                    ax.axvline(idx_max[i], c='r', ls='--', label=f"No.{i+1}: {idx_max[i]: .2f}, {results[col][idx_max[i]]}")
                # ax.set_xlabel(args_name[isIter.index(True)]) if args_name is not None else None
                ax.grid(); ax.legend(); ax.set_ylabel(col)
        
            param_info = {name: arg for arg, name in zip(args, args_name)} if args_name is not None else args
            fig.suptitle(param_info)
            fig.show()
        elif sum(isIter) == 2:  # 若有2个优化参数，利用QSimPy画三维图
            idx_plot = [idx for idx, value in enumerate(isIter) if value]
            name_plot = [args_name[idx] for idx in idx_plot]
            
            fig = plt.figure(figsize=(10, 10), dpi=240)
            for col, pos in zip(['SR', "MaxDD", "Calmar", 'Ret'], [221, 222, 223, 224]):
                ax = fig.add_subplot(pos, projection='3d')
                X1, X2 = np.meshgrid(results_raw.index.levels[idx_plot[0]], results_raw.index.levels[idx_plot[1]])
                Y = results_raw.reset_index().pivot(columns=name_plot[0], index=name_plot[1], values=col)
                ax.plot_surface(X1, X2, Y, cmap='viridis')
                ax.set_xlabel(name_plot[0])
                ax.set_ylabel(name_plot[1])
                ax.set_zlabel(col)
                
            fig.suptitle({name: arg for arg, name in zip(args, args_name)})
            plt.show()
            print(results_raw.loc[results_raw['SR'].nlargest(10).index])
        else:
            print(results_raw.loc[results_raw['SR'].nlargest(10).index])
        
        return results_raw


class DAParamOptimizer(Annealer):
    Tmax = 10000
    steps = 500
    
    def __init__(self, func, initial_state=None):
        self.func = func
        super().__init__(initial_state)
    
    def move(self):
        self.state[0] = random.randint(1, 24)
        self.state[1] = random.randint(1, 30)
        self.state[2] = random.randint(50, 100)
        
        return super().move()
    
    def energy(self):
        pnlall, toall, mvall, target = self.func(self.state[0], self.state[1], self.state[2])
        SR, MaxDD, Calmar, Ret = qs.summary.info(pnlall.sum(axis=1))
        return -SR


def my_dual_annealing(func: callable, **kwargs):
    kwargs["isPlotRet"] = False
    kwargs["isPlotSig"] = False
    func = partial(func, x4=0, **kwargs)
    
    def DA_func(x):
        pnlall, toall, mvall, target = func(x[0], x[1], x[2])
        SR, MaxDD, Calmar, Ret = qs.summary.info(pnlall.sum(axis=1))
        return -SR
    
    print("start")
    result = dual_annealing(DA_func, bounds=[[10, 30], [10, 30], [50, 100]], 
                            maxiter=100, 
                            no_local_search=True, 
                            restart_temp_ratio=0.9, 
                            initial_temp=5E4, 
                            visit=2.9
                            )
    
    print(result['x'])
    print('end')


# ========================================== 分析工具 ==========================================

def plot_pnls(pnls: list[pd.DataFrame], labels: list[str]=None, forms: list[dict]=None):
    """
    绘制回测pnl、分品种pnl

    :param list[pd.DataFrame] pnls: 待绘制的pnl列表（任意数量）
    :param list[str] labels: 对应的标签, defaults to None
    :param list[dict] forms: 对应的绘图参数，传入plt.plot(), defaults to None
    :return plt.Axes, list[plt.Axes]: Axes对象
    """
    # 先把所有pnlall的index改为datetime
    pnls = pnls.copy()
    labels = labels.copy()
    nb_line = len(pnls)
    labels = ['pnl'+str(i) for i in range(nb_line)] if labels is None else labels
    forms = [dict() for i in range(nb_line)] if forms is None else forms
    
    for i in range(len(pnls)):
        pnls[i].index = pd.to_datetime(pnls[i].index, format='%Y%m%d')
    
    # 创建fig
    fig = plt.figure(figsize=(12, 12), layout='constrained')
    subfig1, subfig2 = fig.subfigures(2, 1, hspace=0.01)
    
    # 主图
    ax0: plt.Axes = subfig1.subplots(1, 1)
    for i, (pnl, label) in enumerate(zip(pnls, labels)):
        ax0.plot(pnl.index, pnl.sum(axis=1).cumsum(), label=f'{label}', **forms[i])
        info = ", ".join(
            [name + f"{summary: .2f}" for name, summary in zip(
                ['SR', "MaxDD", "Calmar", 'Ret'], qs.summary.info(pnl.sum(axis=1))
            )])
        ax0.text(0.6, 1.02+(nb_line-1-i)*0.03, label+": "+info, transform=ax0.transAxes, ha='left', va='center')
        
    ax0.legend(loc='upper left')
    ax0.set_ylabel("Pnl")
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0%}'))
    ax0.grid()
    ax0.set_title("All", pad=10)
    
    # 分品种图
    axs = subfig2.subplots(2, 2)
    for future, ax in zip(['if', 'ic', 'ih', 'im'], axs.flat):
        for i, (pnl, label) in enumerate(zip(pnls, labels)):
            ax.plot(pnl.index, pnl[future].cumsum(), label=f'{label}', **forms[i])
            info = ", ".join(
                [name + f"{summary: .2f}" for name, summary in zip(
                ['SR', "MaxDD", "Calmar", 'Ret'], qs.summary.info(pnl[future].fillna(0))
                )])
            ax.text(0.1, 1.04+(nb_line-1-i)*0.06, label+": "+info, transform=ax.transAxes, ha='left', va='center')
        
        ax.set_title(future, loc='left')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0%}'))
        ax.grid(); ax.set_xlabel(None)
    
    return ax0, axs


def plot_LS_pnl(pnlall: pd.DataFrame, target: pd.DataFrame):
    pnlall_l = pnlall.copy()
    pnlall_s = pnlall.copy()
    
    target.index = pnlall.index
    
    pnlall_l[target.shift(2) < 0] = 0
    pnlall_s[target.shift(2) > 0] = 0

    ax0, axs = plot_pnls(
        pnls=[pnlall, pnlall_l, pnlall_s], labels=['ALL', 'Long', 'Short']
        )
    
    plt.gcf().suptitle("Long Short Pnl Compare")
    plt.show()
    

def compare_result(pnlall1: pd.DataFrame, pnlall2: pd.DataFrame, target1: pd.DataFrame, target2: pd.DataFrame, 
                   isPlotDiff=False, isPlotPnl=True, isPlotLSPnl=True, labels=None):
    """
    比较两因子的效果差异

    :param pd.DataFrame pnlall1: 因子1的pnlall
    :param pd.DataFrame pnlall2: 因子2的pnlall
    :param pd.DataFrame target1: 因子1的targett
    :param pd.DataFrame target2: 因子2的targett
    :param bool isPlotDiff: 是否绘制差异散点图, defaults to False
    :param bool isPlotPnl: 是否绘制pnl比较图, defaults to True
    :param bool isPlotLSPnl: 是否绘制分多空pnl比较图, defaults to True
    :param list labels: 两因子的标签, defaults to None
    """
    labels = ["pnl1", "pnl2"] if labels is None else labels
    
    idx_standard = pnlall1.index
    
    target1.index = idx_standard
    target2.index = idx_standard
    target1 = target1.replace(0, None)
    target2 = target2.replace(0, None)
    
    # 基本信息
    info1 = qs.summary.info(pnlall1.sum(axis=1))
    info2 = qs.summary.info(pnlall2.sum(axis=1))
    df_info = pd.DataFrame([info1, info2], index=[labels[0], labels[1]], columns=['SR', "MaxDD", "Calmar", 'Ret'])
    df_info.loc['diff'] = df_info.loc[labels[1]] - df_info.loc[labels[0]]

    # 产生区别的位置
    idx_diff = target1.replace(np.nan, 5) != target2.replace(np.nan, 5)
    df_diff = []
    years = range(int(pnlall1.index[0] / 10000), 2024, 1)
    for year in years:
        idx_year = (idx_standard > year * 10000) * (idx_standard < (year+1) * 10000)
        target2_ = target2[idx_year]
        idx_diff_ = idx_diff[idx_year]
        pnlall1_ = pnlall1[idx_year]; pnlall2_ = pnlall2[idx_year]
        
        df = pd.DataFrame(columns=pd.MultiIndex.from_product((['if', 'ic', 'ih', 'im'], ["trig"])))
        for future in ['if', 'ic', 'ih', 'im']:
            df.loc[year, (future, "trig")] = None if target2_[future].count() == 0 else idx_diff_[future].sum()
            df.loc[year, (future, "freq")] = None if target2_[future].count() == 0 else idx_diff_[future].sum() / target2_[future].count()
            df.loc[year, (future, "ret1")] = None if target2_[future].count() == 0 else pnlall1_[future].values[idx_diff_[future].shift(2).fillna(False)].sum()
            df.loc[year, (future, "ret2")] = None if target2_[future].count() == 0 else pnlall2_[future].values[idx_diff_[future].shift(2).fillna(False)].sum()
        df_diff.append(df)    
    df_diff = pd.concat(df_diff)
    for future in ['if', 'ic', 'ih', 'im']:
        df_diff.loc['all', (future, "trig")] = None if target2[future].count() == 0 else idx_diff[future].sum()
        df_diff.loc['all', (future, "freq")] = None if target2[future].count() == 0 else idx_diff[future].sum() / target2[future].count()
        df_diff.loc['all', (future, "ret1")] = None if target2[future].count() == 0 else pnlall1[future].values[idx_diff[future].shift(2).fillna(False)].sum()
        df_diff.loc['all', (future, "ret2")] = None if target2[future].count() == 0 else pnlall2[future].values[idx_diff[future].shift(2).fillna(False)].sum()
        
        df_diff[(future, "ret_gained")] = df_diff[(future, "ret2")] - df_diff[(future, "ret1")]
    
    if isPlotDiff:
        plt.figure(figsize=(10, 6))
        pl = pnlall1.copy()
        pl[~idx_diff.shift(2).fillna(False)] = None
        pl.index = pd.to_datetime(pl.index, format="%Y%m%d")
        for future in ['if', 'ic', 'ih', 'im']:
            plt.scatter(pl.index, pl[future].values, label=future)
        plt.grid()
        plt.legend()
        plt.title(f"Changed Pnl")
        plt.show()
    
    if isPlotLSPnl:
        pnlall1_l = pnlall1.copy(); pnlall1_s = pnlall1.copy()
        pnlall1_l[target1.shift(2) < 0] = 0
        pnlall1_s[target1.shift(2) > 0] = 0
        pnlall2_l = pnlall2.copy(); pnlall2_s = pnlall2.copy()
        pnlall2_l[target2.shift(2) < 0] = 0
        pnlall2_s[target2.shift(2) > 0] = 0
        
        ax0, axs = plot_pnls(
            pnls=[pnlall1_l, pnlall2_l, pnlall1_s, pnlall2_s], 
            labels=[label+', '+tag for label, tag in zip(labels+labels, ['long', 'long', 'short', 'short'])],
            forms=[{'ls': '-', 'c': 'C0'}, {'ls': '-', 'c': 'C1'}, {'ls': ':', 'c': 'C0'}, {'ls': ':', 'c': 'C1'}])
        
        plt.gcf().suptitle("Long Short Pnl Compare")
        plt.show()
    
    if isPlotPnl:
        ax0, axs = plot_pnls(pnls=[pnlall1, pnlall2], labels=labels)
        ax0_ = ax0.twinx()
        (pnlall2.sum(axis=1).cumsum() - pnlall1.sum(axis=1).cumsum()).plot(label='diff', ax=ax0_, c='r', ls='--')
        ax0_.legend(loc='lower right')
        ax0_.set_ylabel("Pnl Difference")
        ax0_.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0%}'))
        
        for future, ax in zip(['if', 'ic', 'ih', 'im'], axs.flat):
            ax_ = ax.twinx()
            (pnlall2[future].cumsum() - pnlall1[future].cumsum()).plot(label='diff', ax=ax_, c='r', ls='--')
            ax_.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0%}'))
        
        plt.gcf().suptitle("Pnl Compare and Difference")
        plt.show()
    
    return df_info, df_diff


if __name__ == "__main__":
    
    # target = pd.DataFrame([1, 0.5, -0.5, 0, -1, -1, 1, -1, 0, 1], columns=['if'])
    # price = pd.DataFrame([100, 101, 102, 103,104, 100, 103, 105, 107, 100], columns=['if'])
    # trading_day = pd.DataFrame([20000102, 20000102, 20000103, 20000103, 20000105, 20000105, 20000107, 20000108, 20000109, 20000110, ], columns=['if'])
    
    # Backtest.basic(target, price, trading_day)
    
    pass