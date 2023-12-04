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


# ============================== 回测绘图 ==============================

def plot(pnl, turnover, fee, position, ofs_time=None, base_dict=None, title='Backtest', btinfo=None):
    # 计算基本信息
    pnldf, summary = info(pnl, turnover, fee, position)
    if base_dict is not None:
        pnldf_base, summary_base = info(base_dict['pnl'], base_dict['turnover'] , base_dict['fee'], base_dict['position'])
    else:
        summary_base = None
    
    fig = plt.figure(figsize=(10, 8), layout='constrained')
    gs = fig.add_gridspec(4, 8)
    
    # 主图
    ax1 = fig.add_subplot(gs[:-1, :-1])
    if base_dict is not None:
        ax1_twin = ax1.twinx()
    handles, labels = [], []
    if ofs_time is None:
        line, = ax1.plot(pnldf['pnl_cum'].index, pnldf['pnl_cum'], c='b')
        handles.append(line); labels.append('pnl')
        if base_dict is not None:
            pnldf['diff'] = pnldf['pnl_cum'] - pnldf_base['pnl_cum']
            
            pdata = pnldf_base['pnl_cum']
            line, = ax1.plot(pdata.index, pdata, c='grey')
            handles.append(line); labels.append('base')
            
            pdata = pnldf['diff']
            line, = ax1_twin.plot(pdata.index, pdata, 'y--')
            handles.append(line); labels.append('diff')   
    else:
        line, = ax1.plot(pnldf['pnl_cum'].loc[:ofs_time].index, pnldf['pnl_cum'].loc[:ofs_time], c='steelblue')
        handles.append(line); labels.append('pnl')
        line, = ax1.plot(pnldf['pnl_cum'].loc[ofs_time:].index, pnldf['pnl_cum'].loc[ofs_time:], c='steelblue', ls='--', alpha=0.7)
        if base_dict is not None:
            pnldf['diff'] = pnldf['pnl_cum'] - pnldf_base['pnl_cum']
            
            line, = ax1.plot(pnldf_base['pnl_cum'].loc[:ofs_time].index, pnldf_base['pnl_cum'].loc[:ofs_time], 'lightgrey')
            handles.append(line); labels.append('base')
            line, = ax1.plot(pnldf_base['pnl_cum'].loc[ofs_time:].index, pnldf_base['pnl_cum'].loc[ofs_time:], 'lightgrey', alpha=0.7)
            
            line, = ax1_twin.plot(pnldf['diff'].loc[:ofs_time].index, pnldf['diff'].loc[:ofs_time], 'r--', lw=0.5, alpha=0.7)
            handles.append(line); labels.append('diff')
            line, = ax1_twin.plot(pnldf['diff'].loc[ofs_time:].index, pnldf['diff'].loc[ofs_time:], 'r--', lw=0.5, alpha=0.7)
    ax1.grid(True)
    ax1.set_xlabel('')
    ax1.legend(handles=handles, labels=labels)
    ax1.set_title(", ".join([f"{key}: {value}" for key, value in btinfo.items()]))
    
    # 回撤图
    ax2 = fig.add_subplot(gs[-1, :-1])
    ax2.bar(pnldf['drawdowm'].index, pnldf['drawdowm'], color='grey')
    ax2.xaxis.set_visible(False)
    
    # 信息表格
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
        return structed
    
    def plot_info_tabel(ax, summary, summary_base=None, title=None):
        text = np.expand_dims(restructure_summary(summary).to_numpy(), axis=0).T
        width = [1.2]; cc = [[('steelblue', 0.5)]]* len(summary)
        if summary_base is not None:    # 列出基准信息
            text = np.concatenate(
                [text, np.expand_dims(restructure_summary(summary_base).to_numpy(), axis=0).T], 
                axis=1)
            width = [1, 1]; cc = [[('steelblue', 0.5), 'lightgrey']] * len(summary)
            
        ax.table(cellText=text, cellColours=cc, rowLabels=summary.index, loc='best', colWidths=width)
        ax.axis('off')
        ax.set_title(title)
    
    # 全区间信息
    ax3 = fig.add_subplot(gs[0, -1])
    plot_info_tabel(ax3, summary, summary_base, title='All Time')
    
    if ofs_time is not None:
        # 样本内信息
        ax4 = fig.add_subplot(gs[1, -1])
        _, summary = info(pnl.loc[:ofs_time], turnover.loc[:ofs_time], fee.loc[:ofs_time], position.loc[:ofs_time])
        if base_dict is not None:
            _, summary_base = info(
                base_dict['pnl'].loc[:ofs_time], base_dict['turnover'].loc[:ofs_time], 
                base_dict['fee'].loc[:ofs_time], base_dict['position'].loc[:ofs_time])
        plot_info_tabel(ax4, summary, summary_base, title='In Sample')
        
        # 样本外信息
        ax5 = fig.add_subplot(gs[2, -1])
        _, summary = info(pnl.loc[ofs_time:], turnover.loc[ofs_time:], fee.loc[ofs_time:], position.loc[ofs_time:])
        if base_dict is not None:
            _, summary_base = info(
                base_dict['pnl'].loc[ofs_time:], base_dict['turnover'].loc[ofs_time:], 
                base_dict['fee'].loc[ofs_time:], base_dict['position'].loc[ofs_time:])
        plot_info_tabel(ax5, summary, summary_base, title='Out Sample')
    
    fig.suptitle(title)
    
    plt.plot()
    




