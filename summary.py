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
from scipy import stats

import matplotlib.pyplot as plt



# ============================== 计算因子信息 ==============================

def ic(data: pd.DataFrame = None, ret: pd.DataFrame | str = None, factor: pd.DataFrame | str = None, 
       method: str = 'pearson'):
    """计算截面因子的IC、IR

    Args:
        data (pd.DataFrame): long-form格式数据，以date、code为index
        ret (pd.DataFrame | str): wide-form真实收益率，或long-form数据的对应列名
        factor (pd.DataFrame | str): wide-form因子值，或long-form数据的对应列名
    """
    # 优先读取long-form数据
    if data is not None:
        assert isinstance(ret, str) and isinstance(factor, str)
        data = data[[ret, factor]].reset_index()
        ret = data.pivot_table(index='date', columns='code', values=ret)
        factor = data.pivot_table(index='date', columns='code', values=factor)
    else:
        assert isinstance(ret, pd.DataFrame) and isinstance(factor, pd.DataFrame)
    
    # 计算IC
    ic = ret.corrwith(factor, axis=1, method=method)
    ic_mean = ic.mean(); ic_std = ic.std()
    ir = ic_mean / ic_std * pow(250, 0.5)
    
    print(ic_mean, ir)
    
    # 计算RankIC
    ret_rank = ret.rank(pct=True, ascending=True, axis=1)
    ret = ret[ret_rank > 0.5]
    factor = factor[ret_rank > 0.5]
    rankic = ret_rank.corrwith(factor, axis=1, method=method)
    rankic_mean = rankic.mean(); rankic_std = rankic.std()
    rankir = rankic_mean / rankic_std * pow(250, 0.5)
    
    print(rankic_mean, rankir)
    

# ============================== 计算回测信息 ==============================

def info(pnl: pd.Series, turnover: pd.Series, fee: pd.Series, position: pd.Series):
    """利用Vec输出计算回测指标

    Args:
        pnl (pd.Series): 日频收益率
        position (pd.Series): 每日仓位
        turnover (pd.Series): 每日换手率
        fee (pd.DataFrame): 每日手续费

    Returns:
        _type_: _description_
    """
    # Alpha & sigma
    alpha = pnl.mean() * 1e4
    sigma = pnl.std() * 1e4
    # 夏普
    Sharpe = np.around(alpha / sigma * np.sqrt(252), 2)
    # 年化收益率
    annual_ret = pnl.mean() * 252
    annual_ret = np.around(annual_ret * 100, 2)
    # 日胜率
    win_rate = (pnl > 0).sum() / (pnl.notna().sum())
    # 换手率
    turnover = turnover.mean()
    # 平均交易股票数
    nb_stock = int(position.fillna(0).apply(lambda x: (x!=0).sum(), axis=1).mean())
    
    df = pd.DataFrame(pnl, columns=['pnl'])
    df['pnl_cum'] = df['pnl'].cumsum()
    df['fee'] = fee
    df['drawdowm'] = df['pnl_cum'] - df['pnl_cum'].cummax()
    
    # 最大回撤
    mdd = np.around(df['drawdowm'].min() * 100, 2)
    mdd_date = df['drawdowm'].idxmin()
    
    # 数值信息保存在info series中
    summary = pd.Series(
        data=[alpha, sigma, annual_ret, Sharpe, turnover, nb_stock, win_rate, mdd, mdd_date],
        index=['alpha', 'sigma', 'annual_ret', 'Sharpe', 'turnover', 'nb_stock', 'win_rate', 'mdd', 'mdd_date'])
    
    return df, summary


def infos(pnl: pd.Series, turnover: pd.Series, fee: pd.Series, position: pd.Series, time_table: list):
    summary = []
    for i in range(len(time_table)):
        # 切片分段的数据
        start = time_table[i][1]; end = time_table[i][2]
        pnl_ = pnl.loc[start: end]
        turnover_ = turnover.loc[start: end]
        fee_ = fee.loc[start: end]
        position_ = position.loc[start: end]
        # 计算info
        _, s = info(pnl_, turnover_, fee_, position_)
        summary.append(s)
    
    return pd.concat(summary, axis=1)

def stat(summary1, summary2):

    res = pd.DataFrame(
        index=['alpha', 'sigma', 'annual_ret', 'Sharpe', 'turnover', 'nb_stock', 'win_rate'],
        columns=['t_stat', 'p_value'])
    for index in res.index:
        res.loc[index, 't_stat'], res.loc[index, 'p_value'] = stats.ttest_rel(
            summary1.loc[index], summary2.loc[index])
        
    return res
    
    
# ============================== 回测绘图 ==============================
class Plotter(object):
    c = ['steelblue', 'palevioletred', 'y', 'seagreen', 
         'chocolate', 'indigo', 'darkred', 'gold', 
         'slategrey', 'palegreen']
    
    def __init__(self, ofs_time=None) -> None:
        """
        初始化画布
        """
        self.ofs_time = ofs_time
        
        self.fig = plt.figure(figsize=(10, 8), layout='constrained')
        gs = self.fig.add_gridspec(4, 10)
        self.ax1 = self.fig.add_subplot(gs[:-1, :])
        self.ax2 = self.fig.add_subplot(gs[-1, 0:2])
        self.ax3 = self.fig.add_subplot(gs[-1, 2:4])
        if ofs_time is not None:
            self.ax4 = self.fig.add_subplot(gs[-1, 4:6])
            self.ax5 = self.fig.add_subplot(gs[-1, 6:8])
        
        self.ax6 = self.fig.add_subplot(gs[-1, 8:]); self.ax6.axis('off')
        
        self.pnls, self.summarys, self.handles, self.labels = [], [], [], []
        
    def plot_pnl(self, pnl, turnover, fee, position, **kwargs):
        
        ofs_time = self.ofs_time
        
        label = f'pnl_{len(self.handles)+1}' if 'label' not in kwargs else kwargs['label']
        if 'c' not in kwargs and 'color' not in kwargs:
            c = self.c[len(self.handles)]
        else:
            c = kwargs['c'] if 'c' in kwargs else kwargs['color']
                    
        pnldf, summary = info(pnl, turnover, fee, position)
        # self.pnls.append(pnldf); self.summarys.append(summary)
        self.main_pnldf = pnldf
        
        if ofs_time is None:
            line, = self.ax1.plot(pnldf['pnl_cum'].index, pnldf['pnl_cum'], c=c)
            self.handles.append(line); self.labels.append(label); self.summarys.append([summary, ])
        else:
            # 样本内
            line, = self.ax1.plot(pnldf['pnl_cum'].loc[:ofs_time].index, pnldf['pnl_cum'].loc[:ofs_time], c=c)
            self.handles.append(line); self.labels.append(label)
            _, summary_in = info(pnl.loc[:ofs_time], turnover.loc[:ofs_time], fee.loc[:ofs_time], position.loc[:ofs_time])
            # 样本外
            line, = self.ax1.plot(pnldf['pnl_cum'].loc[ofs_time:].index, pnldf['pnl_cum'].loc[ofs_time:], c=c, ls='--', alpha=0.7)
            _, summary_out = info(pnl.loc[ofs_time:], turnover.loc[ofs_time:], fee.loc[ofs_time:], position.loc[ofs_time:])
            
            self.summarys.append([summary, summary_in, summary_out])
        
        # 只给第一个pnl画回撤图
        # if len(self.handles) == 1:
        #     self.ax2.bar(pnldf['drawdowm'].index, pnldf['drawdowm'], color='grey')
        #     self.ax2.xaxis.set_visible(False)
        
        return pnldf, summary
            
    def plot_pnl_base(self, pnl, turnover, fee, position, **kwargs):
        label = 'base' if 'label' not in kwargs else kwargs['label']
        ofs_time = self.ofs_time
        
        pnldf, summary = info(pnl, turnover, fee, position)
        # self.summarys.append(summary)
        
        if self.ofs_time is None:
            line, = self.ax1.plot(pnldf['pnl_cum'].index, pnldf['pnl_cum'], c='lightgrey')
            self.handles.append(line); self.labels.append(label); self.summarys.append([summary, ])
        else:
            # 样本内
            line, = self.ax1.plot(pnldf['pnl_cum'].loc[:ofs_time].index, pnldf['pnl_cum'].loc[:ofs_time], c='lightgrey')
            self.handles.append(line); self.labels.append(label)
            _, summary_in = info(pnl.loc[:ofs_time], turnover.loc[:ofs_time], fee.loc[:ofs_time], position.loc[:ofs_time])
            # 样本外
            line, = self.ax1.plot(pnldf['pnl_cum'].loc[ofs_time:].index, pnldf['pnl_cum'].loc[ofs_time:], c='lightgrey', ls='--', alpha=0.9)
            _, summary_out = info(pnl.loc[ofs_time:], turnover.loc[ofs_time:], fee.loc[ofs_time:], position.loc[ofs_time:])
            
            self.summarys.append([summary, summary_in, summary_out])
        
        if len(self.handles) == 2:
            ax1_twin = self.ax1.twinx()
            line, = ax1_twin.plot(pnldf['pnl_cum'].index, self.main_pnldf['pnl_cum']-pnldf['pnl_cum'], 'r--', lw=0.5)
            self.handles.append(line); self.labels.append('diff')

    @staticmethod
    def restructure_summary(summary):
        'alpha', 'sigma', 'annual_ret', 'Sharpe', 'turnover', 'nb_stock', 'win_rate', 'mdd', 'mdd_date'
        structed = summary.copy()
        structed['alpha'] = f"{structed['alpha']: .2f}"
        structed['sigma'] = f"{structed['sigma']: .2f}"
        structed['annual_ret'] = f"{structed['annual_ret']: .2f}%"
        structed['Sharpe'] = f"{structed['Sharpe']: .2f}"
        structed['turnover'] = f"{structed['turnover']: .2f}"
        structed['nb_stock'] = f"{structed['nb_stock']: d}"
        structed['win_rate'] = f"{structed['win_rate']: .2f}"
        structed['mdd'] = f"{structed['mdd']: .2f}%"
        structed['mdd_date'] = f"{structed['mdd_date'].strftime('%Y-%m-%d')}"
        # structed['mdd_date'] = f"{structed['mdd_date']}"
        return structed
    
    @staticmethod
    def plot_info_tabel(ax, summary, summary_base=None, title=None):
        text = np.expand_dims(Plotter.restructure_summary(summary).to_numpy(), axis=0).T
        cc = [[('steelblue', 0.5)]]* len(summary)
        if summary_base is not None:    # 列出基准信息
            text = np.concatenate(
                [text, np.expand_dims(Plotter.restructure_summary(summary_base).to_numpy(), axis=0).T], 
                axis=1)
            cc = [[('steelblue', 0.5), 'lightgrey']] * len(summary)
            
        ax.table(cellText=text, cellColours=cc, loc='best')
        ax.axis('off')
        ax.set_title(title)
    
    def plot_stat(self, ax, stat_result=None):
        """
        对分段效果进行统计检验
        """
        cc = [['yellow', 'yellow'] if p < 0.05 else ['white', 'white'] for p in stat_result['p_value']]
        ax.table(cellText=np.around(stat_result.to_numpy().astype(float), 2), loc='best', cellColours=cc)
        ax.axis('off')
        ax.set_title('Paired t-test')
    
    def plot_info(self, stat_result=None):
        # 项目
        labels = np.expand_dims(self.summarys[0][0].index.to_numpy(), axis=0).T
        self.ax2.table(cellText=labels, loc='best', fontsize=20, colWidths=[0.6, ])
        self.ax2.axis('off')
        
        # 全区间信息
        summary_base = self.summarys[1][1] if len(self.summarys) > 1 else None
        self.plot_info_tabel(ax=self.ax3, summary=self.summarys[0][0], summary_base=summary_base, title='All Time')
        ofs_time = self.ofs_time
        
        if ofs_time is not None:
            # 样本内信息
            summary_base = self.summarys[1][1] if len(self.summarys) > 1 else None
            self.plot_info_tabel(ax=self.ax4, summary=self.summarys[0][1], summary_base=summary_base, title='In Sample')
            
            # 样本外信息
            summary_base = self.summarys[1][2] if len(self.summarys) > 1 else None
            self.plot_info_tabel(ax=self.ax5, summary=self.summarys[0][2], summary_base=summary_base, title='Out Sample')
        
        if stat_result is not None:
            self.plot_stat(ax=self.ax6, stat_result=stat_result)
        
        # 返回样本外的summary
        return self.summarys[0][2]
        
    def show(self, title=None, suptitle=None):
        self.ax1.grid()
        self.ax1.legend(self.handles, self.labels)
        self.ax1.set_title(title)
        
        self.fig.suptitle(suptitle)
        
        self.fig.show()
        