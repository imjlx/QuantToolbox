#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tool.py
@Time    :   2023/12/04 09:49:00
@Author  :   Haoran Jia
@Version :   1.0
@Contact :   21211140001@m.fudan.edu.cn
@License :   (C)Copyright 2023 Haoran Jia.
@Desc    :   None
'''

import os
import time
import datetime
import numpy as np
from numpy import ndarray
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.decomposition import PCA

from functools import partial
from multiprocessing import Pool
import multiprocessing as mp

import sys
sys.path.append('/home/jiahaoran/workspace')
import QuantToolbox as qb


def train_test_split_basic(data_path='data/data.feather', split_date='2021-06-30'):
    data = pd.read_feather(data_path)
    data.reset_index(inplace=True)
    
    data_train = data[data['date'] <= split_date]
    data_train.set_index(['date', 'code'], inplace=True)
    x_train = data_train.loc[:, 'x1':'x160']
    y_train = data_train['y6']
    
    data_test = data[data['date'] > split_date]
    data_test.set_index(['date', 'code'], inplace=True)
    x_test = data_test.loc[:, 'x1':'x160']
    y_test = data_test['y6']
    
    return (data_train, x_train, y_train), (data_test, x_test, y_test)


# ============================== 步进分析法 ==============================

class StepwiseSlection(object):
    """步进分析法：Wrapper Method 以线性回归作为Wrapper。
    作用：筛选出对回归变量y解释能力强的特征，去掉解释能力不显著的。（不能解决共线性问题）
    每轮
        正向：从0特征开始，每次计算引入新特征的p值，选择所有备选特征中p值最高的、且满足阈值条件的引入
        反向：新特征引入后检验旧特征是否变得不满足p值条件，若不满足就删除
    直到没有新的特征被引入
    """
    
    def __init__(self, x: pd.DataFrame, y: pd.Series) -> None:
        
        self.X = sm.add_constant(x)
        self.y = y
    
    @staticmethod
    def x_pvalue_(x_add, X, y):
        model = sm.OLS(y, pd.concat([X, x_add], axis=1))
        result = model.fit()
        pvalues: pd.Series = result.pvalues
        pvalues = pvalues.rename(index={x_add.name: 'x_add'})
        
        return pvalues
        
    def Execute(self, initial_list=[], threshold_in=0.05, threshold_out=0.1):
        current_list = ['const'] + initial_list
        
        next = True; i = 1
        while next:
            # 本轮待检验的列表
            test_list = list(set(self.X.columns) - set(current_list))
            
            x_list = []; pvalues_list = []
            pbar = tqdm(test_list)
            pbar.set_description(f"Round {i}, n_current={len(current_list)}")
            for col in pbar:
                x_add = self.X[col]
                x_list.append(col)
                pvalues_list.append(self.x_pvalue_(x_add, self.X[current_list], self.y))
            
            # 提取出新加入的x的p值
            p_list = [pvalues['x_add'] for pvalues in pvalues_list]
            idx = np.argmin(p_list)
            p = p_list[idx]
            x_add = x_list[idx]
            print(f"Best X: {x_add}, p_value: {p}")
            
            if p < threshold_in:
                # 最优的x满足准入条件，加入
                current_list.append(x_add)
                next = True
                # 删除变差到不可接受的x
                pvalues = pvalues_list[idx].drop('const')
                x_del = set(pvalues[pvalues > threshold_out].index)
                print(f"After add: worst pvalues = {pvalues.max()}, delete {x_del}")
                current_list = list(set(current_list) - x_del)
            else:
                next = False
            
            print()
            i += 1
    
    @staticmethod
    def calculate_new_features_pvalues(x_add: pd.DataFrame, X: pd.DataFrame, y: pd.Series, pvalues_dict, counts, i):
        """子进程中计算新加入一部分新特征后的pvalues

        Args:
            x_add (pd.DataFrame): 需要检验的新特征，一列一个
            X (pd.DataFrame): 原始的特征
            y (pd.Series): 预测变量
            pvalues_dict (_type_): 用于记录结果的mp.Manager.dict
            counts (_type_): 用于记录进程的mp.Array
            i (_type_): 用于记录进程的标记
        """
        for col in x_add.columns:
            x = x_add[col]
            model = sm.OLS(y, pd.concat([X, x], axis=1))
            result = model.fit()
            pvalues = result.pvalues
            pvalues = pvalues.rename(index={x.name: 'x_add'})
            pvalues_dict[col] = pvalues
            # 刷新进度条
            counts[i] += 1
    
    @staticmethod
    def progress_manager(progress, total):
        pbar = tqdm(total=total)
        last_values = [0] * len(progress)
        while True:
            current_values = list(progress)
            increments = [curr - last for curr, last in zip(current_values, last_values)]
            total_increment = sum(increments)
            if total_increment > 0:
                pbar.update(total_increment)
                last_values = current_values
            if sum(current_values) == total:
                break
            time.sleep(0.1)  # 更新间隔
        pbar.close()
    
    def CalculateNewFeaturesPvalues(self, current_list=[], n_jobs=10):
        """计算所有未引入特征引入后的pvalues

        Args:
            current_list (list, optional): 当前已引入特征. Defaults to [].
            n_jobs (int, optional): 并行进程数. Defaults to 10.

        Returns:
            _type_: _description_
        """
        test_list = list(set(self.X.columns) - set(current_list))
        nb_test = len(test_list)
        
        # 确定使用的特征
        X = self.X[current_list]
        # 根据进程数，确定带检验特征的分组
        nb_test_per_process = len(test_list) // n_jobs
        idx_sta = np.array(range(0, nb_test, nb_test_per_process))[:n_jobs] # 前nb_jobs个分组
        idx_end = idx_sta + nb_test_per_process
        idx_end[-1] = nb_test  # 最后一组扩展到结尾
       
        # manager = mp.Manager()
        with mp.Manager() as manager:
            pvalues_dict = manager.dict()
            counts = mp.Array('i', n_jobs)
            processes = []
            for i in range(n_jobs):
                x_cols = test_list[idx_sta[-1-i]:idx_end[-1-i]] # 倒着取x，保证最长的进程最先开始
                # 创建进程，完成一部分的计算，拟合pvalues结果保存在pvalues_dict，counts用于记录进程
                process = mp.Process(target=self.calculate_new_features_pvalues, args=(self.X[x_cols], X, self.y, pvalues_dict, counts, i))
                process.start()
                processes.append(process)
            
            # 新建子进程管理进度条
            pbar_manager = mp.Process(target=self.progress_manager, args=(counts, nb_test))
            pbar_manager.start()
            for process in processes:
                process.join()
            pbar_manager.join()
            
            keys = pvalues_dict.keys()
            vals = pvalues_dict.values()
            
            df = pd.DataFrame({key: val for key, val in zip(keys, vals)})
            
        # return {key: val for key, val in zip(keys, vals)}
        return df
    
    def RefreshFeatureList(self, df, current_list, threshold_in=0.05, threshold_out=0.1, n_refresh=1):
        x_selected_list = list(df.loc['x_add'].nsmallest(n_refresh).index)
        for x_selected in x_selected_list:
            if df.loc['x_add', x_selected] < threshold_in:
                # 引入新特征x
                print(f"Select feature: {x_selected}, \tp_value={df.loc['x_add', x_selected]}")
                current_list.append(x_selected)
                
                # 删除变差到不可接受的x
                x_delete_list = list(df[x_selected][df[x_selected] > threshold_out].index)
                for x_delete in x_delete_list:
                    if x_delete in current_list and x_delete != 'const':
                        print(f"Delete feature: {x_delete}, \tp_value={df[x_selected].max()}")
                        current_list.remove(x_delete)
                    
        return current_list
        
    def Execute_process(self, initial_list=[], n_jobs=10, threshold_in=0.05, threshold_out=0.1, refresh_speed=1, log=None):
        current_list = ['const'] + initial_list
        
        next = True; i = 1
        while next:
            old_list = current_list.copy()
            # 计算新增x的pvalues
            df = self.CalculateNewFeaturesPvalues(current_list, n_jobs)
            # 更新现有特征列表
            n_refresh = refresh_speed if refresh_speed > 1 else int(np.ceil((len(self.X.columns) - len(current_list)) * refresh_speed))
            print("refresh", n_refresh)
            current_list = self.RefreshFeatureList(df, current_list, threshold_in, threshold_out, n_refresh)
            
            info = f"{i}\t{current_list}\n"
            print(info)
            i += 1
            if log is not None:
                with open(log, 'a') as f:
                    f.write(info)
            if old_list == current_list:
                next = False

    # ===============================Old API==========================
    def one_loop(self, current_list=[], threshold_in=0.05, threshold_out=0.1, n_jobs=10):
        test_list = list(set(self.X.columns) - set(current_list))
        nb_test = len(test_list)
        
        # 确定使用的特征
        X = self.X[current_list]
        # 根据进程数，确定带检验特征的分组
        nb_test_per_process = len(test_list) // n_jobs
        idx_sta = np.array(range(0, nb_test, nb_test_per_process))[:n_jobs] # 前nb_jobs个分组
        idx_end = idx_sta + nb_test_per_process
        idx_end[-1] = nb_test  # 最后一组扩展到结尾
       
        with mp.Manager() as manager:
            pvalues_dict = manager.dict()
            counts = mp.Array('i', n_jobs)
            processes = []
            for i in range(n_jobs):
                x_cols = test_list[idx_sta[-1-i]:idx_end[-1-i]] # 倒着取x，保证最长的进程最先开始
                # 创建进程，完成一部分的计算，拟合pvalues结果保存在pvalues_dict，counts用于记录进程
                process = mp.Process(target=self.calculate_new_features_pvalues, args=(self.X[x_cols], X, self.y, pvalues_dict, counts, i))
                process.start()
                processes.append(process)
            
            # 进度条管理
            pbar_manager = mp.Process(target=self.progress_manager, args=(counts, nb_test))
            pbar_manager.start()
            for process in processes:
                process.join()
            pbar_manager.join()
            
            # pbar = tqdm(total=nb_test)
            # while True:
            #     total_progress = sum(counts)
            #     pbar.n = total_progress
            #     pbar.refresh()
            #     if total_progress >= counts:
            #         break
            #     time.sleep(0.1)
            # pbar.close()
            
            # for process in processes:
            #     process.join()
            
            # 分析pvalues_dict
            x_list = pvalues_dict.keys()
            pvalues_list = pvalues_dict.values()
            pvalue_list = [pvalues['x_add'] for pvalues in pvalues_list]
            
        idx = np.argmin(pvalue_list)
        x = x_list[idx]
        p = pvalue_list[idx]
        
        if p < threshold_in:
            print(f"Select feature: {x}, \tp_value={p}")
            current_list.append(x)
            # 删除变差到不可接受的x
            pvalues = pvalues_list[idx].drop('const')
            x_del = list(pvalues[pvalues > threshold_out].index)
            print(f"Delete feature: {x_del}, \tp_value={pvalues.max()}")
            for x in x_del:
                current_list.remove(x)
        
        return current_list

    def Execute_process_old(self, initial_list=[], threshold_in=0.05, threshold_out=0.1, n_jobs=10, log=None):
        current_list = ['const'] + initial_list
        
        next = True; i = 1
        while next:
            old_list = current_list.copy()
            current_list = self.one_loop(current_list, n_jobs=n_jobs, 
                                         threshold_in=threshold_in, threshold_out=threshold_out)
            info = f"{i}\t{current_list}\n"
            print(info)
            i += 1
            if log is not None:
                with open(log, 'a') as f:
                    f.write(info)
            if old_list == current_list:
                next = False
    

# ============================== 数据预处理类 ==============================

class ZeroFeatureEliminator(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None):
        # 判断是否有特征值恒等于0，提前删除
        self.zero_col = X.columns[(X == 0).all()]
        return self
    
    def transform(self, X, y=None):
        X = X.drop(columns=self.zero_col)
        return X
        

class ColinearityFeatureEliminator(BaseEstimator, TransformerMixin):
    """
    共线性特征消除器
    """
    def __init__(self, thresh=5) -> None:
        super().__init__()
        self.thresh = thresh
    
    @staticmethod
    def backwards_elimination(X: pd.DataFrame, thresh=5):
        """递归VIF特征消除，逐步删除VIF大于thresh的特征

        Args:
            X (pd.DataFrame): 一列一个特征
            thresh (int, optional): VIF阈值. Defaults to 5.
        """
        def find_worst_feature(X: pd.DataFrame, thresh):
            """找出特征矩阵中，VIF最大的特征

            Args:
                X (pd.DataFrame): 特征矩阵
                thresh (_type_): 阈值

            Returns:
                x: 待删除的特征名
            """
            corr = np.corrcoef(X.values, rowvar=False)   # 使用numpy的corrcoef计算，速度远高于pandas
            corr = pd.DataFrame(corr, index=X.columns, columns=X.columns)
            vif = pd.Series(np.linalg.inv(corr.to_numpy()).diagonal(), index=corr.columns, name='VIF')
            if vif.max() > thresh:
                x = vif.index[vif.argmax()]
                # print(f"Delete Feature: {xdel}, VIF={vif.max()}")
            else:
                x = None
            return x
        
        X_current = X
        next = True; i = 1
        while next:
            # 循环删除VIF超过阈值的x
            xdel = find_worst_feature(X_current, thresh)
            if xdel is not None:
                X_current = X_current.drop(columns=xdel)
                i += 1
            else:
                next = False
                
        return list(X_current.columns)
    
    def fit(self, X, y=None):
        self.col = self.backwards_elimination(X, thresh=self.thresh)
        return self
    
    def transform(self, X, y=None):
        return X[self.col]


class CombinedFeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, thresh_vif=5, thresh_cor=0.8, thresh_pca=0.95) -> None:
        super().__init__()
        self.thresh_vif = thresh_vif
        self.thresh_cor = thresh_cor
        self.thresh_pca = thresh_pca
    
    def fit(self, X, y=None):
        # 找出相关性高的一批特征
        cor, vif = qb.factor.corr_analyse(X, isPlotCorr=False, isPlotVIF=False)
        set_cor = set(cor[cor['corr'] > self.thresh_cor]['var1'].values
                       ) | set(cor[cor['corr'] > self.thresh_cor]['var2'].values)
        set_vif = set(vif[vif > self.thresh_vif].index)
        self.pca_list = list(set_cor | set_vif)
        self.pca = PCA(self.thresh_pca)
        self.pca.fit(X[self.pca_list])
        
        return self
    
    def transform(self, X, y=None):
        X_rest = X.drop(columns=self.pca_list).values
        X_new = self.pca.transform(X[self.pca_list])
        X_final = np.concatenate([X_rest, X_new], axis=1)
        # print(X_final.shape)
        return X_final


# ============================== 时序交叉验证 ==============================



def save_base():
    print('读取数据...', end='\t')
    data = pd.read_feather('data/data.feather')
    base = pd.read_feather('data/base.feather')

    pipeline = Pipeline([
        ('D0', qb.Tools.ZeroFeatureEliminator()),
        ('OLS', LinearRegression()),
    ])
    p = TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], pipeline=pipeline)
    p.train_test_split(m_train=24, m_test=1)
    p.Predict_()
    
    p.Backtest(ret=data['y1'], limit=data['ud_limit_h3'], pct_range=(0.95, 1), )
    
    base['base_24_1'] = p.y_pred
    base.to_feather('data/base.feather')
    print(base.columns)


if __name__ == "__main__":
    
    os.chdir('/home/jiahaoran/workspace')
    
    # print('读取数据...', end='\t'); start_time = time.time()
    # data = pd.read_feather('data/data.feather')
    # base = pd.read_feather('data/base.feather')
    # print(f'{time.time() - start_time: .2f} s')
    
    # pipeline = Pipeline([
    #     # ('D0', qb.Tools.ZeroFeatureEliminator()),
    #     # ('OLS', LinearRegression()),
    #     ('QuantReg', linear_model.QuantileRegressor(quantile=0.9, solver='highs', alpha=0))
    # ])
    # p = TimeSeriesCrossValidation(X=data.loc[:, 'x1':'x160'], y=data['y6'], pipeline=pipeline)
    # p.train_test_split(m_train=36, m_test=3)
    # p.Predict_()
    # print("画图...")
    # p.Backtest(ret=data['y1'], limit=data['ud_limit_h3'], base=base['base_36_3'], pct_range=(0.95, 1), 
    #            title='VIF backward feature elimination', 
    #            # title='thresh_vif=5, thresh_cor=0.7, thresh_pca=0.95'
    #            )
    # print(f'共{time.time() - start_time: .2f} s')
