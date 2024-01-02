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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools

import multiprocessing as mp


class ParamSearcher(object):
    def __init__(self, func: callable=None, fpath=None, cols=['Sharpe', 'annual_ret']) -> None:
        
        self.func = func
        self.results = None
        self.fpath = fpath
        self.cols = cols
    
    def search_param(self, args=None, args_iter=None, args_name=None, method='iter', save_verbose=None, n_jobs=None, fpath=None):
        
        # 输入预处理
        fpath = self.fpath if fpath is None else fpath
        if args is not None:    # 生成待遍历参数列表
            for i in range(len(args)):
                if type(args[i]) == int or type(args[i]) == float:
                    args[i] = [args[i]]
            args_name = ["x"+str(i+1) for i in range(len(args))] if args_name is None else args_name
            multi_index = pd.MultiIndex.from_product(args, names=args_name)
            args_iter = np.array(list(itertools.product(*args)))
        else:   # 或直接传入
            assert args_iter is not None
            args_name = ["x"+str(i+1) for i in range(args_iter.index.nlevels)] if args_name is None else args_name
            multi_index = pd.MultiIndex.from_arrays(args_iter)
        
        # 回测
        if method == 'iter':    # 通过简单的循环实现
            results = []; indexs = []
            for i, arg in tqdm(enumerate(args_iter), total=len(args_iter)):
                results.append(self.func(*arg))
                indexs.append(arg)
                if save_verbose not in [0, None] and i % save_verbose == 0:
                    assert self.fpath is not None
                    temp = pd.concat(results, axis=1).T
                    temp = temp.set_index(pd.MultiIndex.from_tuples(indexs, names=args_name))
                    temp.to_feather(self.fpath)
            
        elif method == 'pool':  # 通过自动创建Pool，暂未调试
            n_jobs = os.cpu_count() - 1 if n_jobs is None else n_jobs
            pool = mp.Pool(n_jobs)
            results = list(tqdm(pool.imap(self.func, tuple(args_iter)), total=len(args_iter)))
            results = pd.DataFrame(
                results, index=multi_index, 
                columns=['SR', "MaxDD", "Calmar", 'Ret', 'Leg', 'TO', 'DD_st', 'DD_ed', 'DDD'])
            
        else:
            raise ValueError('method should in [\'iter\', \'pool\']')
        
        # 拼接全部结果并保存
        self.results = pd.concat(results, axis=1).T.set_index(multi_index)
        if self.fpath is not None:
            self.results.to_feather(self.fpath)
        
        return self.results
    
    def plot(self, cols=None, fpath=None, **kwargs):
        # 读取数据
        if fpath is not None:   # 优先根据输入读取文件
            if isinstance(fpath, str):
                results = pd.read_feather(self.fpath)
            elif isinstance(fpath, list):
                results = pd.concat([pd.read_feather(f) for f in fpath], axis=1)
                results = results.sort_index()
        elif self.results is not None:    # 其次读取类中的restult
            results = self.results
        else:   # 最后根据初始化时的输入读取文件
            assert self.fpath is not None
            results = pd.read_feather(self.fpath)
        
        cols = self.cols if cols is None else cols
        
        # 处理回测结果
        param_dict = {results.index.names[i]: set(results.index.get_level_values(i)) 
                      for i in range(results.index.nlevels)}
        results = results.droplevel(level=[i for i, v in enumerate(param_dict.values()) if len(v) == 1 ])
        
        # 若只有一个优化参数，直接画效果曲线图
        if results.index.nlevels == 1:
            self._plot_line(results, param_dict, cols, **kwargs)
        elif results.index.nlevels == 2:
            self._plot_heatmap(results, param_dict, cols, **kwargs)
        else:
            print(results[cols].astype(float).nlargest(10, columns=cols[0]))
            
    @staticmethod
    def _plot_line(results, param_dict, cols, **kwargs):
        
        idx_max = results[cols[0]].astype(float).nlargest().index
        fig, axs = plt.subplots(len(cols), 1, figsize=(8, len(cols) * 3), layout='tight', sharex='all')
        
        for col, ax in zip(cols, axs):
            ax.plot(results.index, results[col])
            for i in range(0, 1):
                ax.axvline(idx_max[i], c='r', ls='--', label=f"No.{i+1}: {idx_max[i]: .2f}, {results[col][idx_max[i]]}")
            ax.grid(); ax.legend(); 
            ax.set_ylabel(col)
            ax.set_xlabel(results.index.name)

        fig.suptitle(", ".join([f'{k}={list(v)[0]}' for k, v in param_dict.items() if len(v) == 1 ]))
        fig.show()
        
    @staticmethod
    def _plot_3D(results, param_dict, cols, **kwargs):
        args_name = results.index.names
        
        nb_rows = int(np.ceil(len(cols) / 2))
        fig, axs = plt.subplots(nb_rows, 2, figsize=(10, nb_rows * 5), subplot_kw={'projection': '3d'})
        axs = axs.flat
        for col, ax in zip(cols, axs):
            X1, X2 = np.meshgrid(results.index.levels[0], results.index.levels[1])
            Y = results.reset_index().pivot(columns=args_name[0], index=args_name[1], values=col)
            ax.plot_surface(X1, X2, Y, cmap='viridis')
            ax.set_xlabel(args_name[0])
            ax.set_ylabel(args_name[1])
            ax.set_title(col)
            
        fig.suptitle(", ".join([f'{k}={list(v)[0]}' for k, v in param_dict.items() if len(v) == 1 ]))
        plt.show()
        
        print(results[cols].astype(float).nlargest(10, columns=cols[0]))
    
    @staticmethod
    def _plot_heatmap(results, param_dict, cols, **kwargs):
        
        center = [None] * len(cols) if 'center' not in kwargs else kwargs['center']
        vmin = [None] * len(cols) if 'vmin' not in kwargs else kwargs['vmin']
        vmax = [None] * len(cols) if 'vmax' not in kwargs else kwargs['vmax']
        
        args_name = results.index.names
        
        nb_rows = int(np.ceil(len(cols) / 2))
        fig, axs = plt.subplots(nb_rows, 2, figsize=(16, nb_rows * 7))
        axs = axs.flat
        for i, (col, ax) in enumerate(zip(cols, axs)):
            data = results.pivot_table(values=col, index=args_name[0], columns=args_name[1]).astype(float)
            annot = True if len(data.index) * len(data.columns) <= 225 else False
            fmt = '.2f' if len(data.index) * len(data.columns) <= 144 else False
            sns.heatmap(data, ax=ax, cmap='coolwarm', annot=annot, fmt=fmt, 
                        center=center[i], vmin=vmin[i], vmax=vmax[i])
            ax.set_title(col)
            
        fig.suptitle(", ".join([f'{k}={list(v)[0]}' for k, v in param_dict.items() if len(v) == 1 ]))
        plt.show()
        
        print(results[cols].astype(float).nlargest(10, columns=cols[0]))
    
























