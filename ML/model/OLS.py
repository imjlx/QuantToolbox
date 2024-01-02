
import os; os.chdir('/home/jiahaoran/workspace')
import sys; sys.path.append('/home/jiahaoran/workspace')
import time

import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin

from tqdm import tqdm, trange

import multiprocessing as mp
from multiprocessing import Pool

import QuantToolbox as qb


# ============================== 回归方法 ==============================
# 以下回归方法均继承sklearn基类BaseEstimator、RegressorMixin，接口的统一
# 实际的回归方法主要是调用statsmodels或者sklearn实现

class QuantileRegressor(BaseEstimator, RegressorMixin):
    """ 基于sm.QuantReg的分位数回归
    对不同分位数的拟合程度，表征了因子对不同收益区间的预测能力
    经测试，在大数据量时sm.QuantReg速度快于sklearn.linear_model.QuantileRegressor
    """
    def __init__(self, q) -> None:
        super().__init__()
        self.q = q
        
    def fit(self, X, y):
        self.model = sm.QuantReg(y, sm.add_constant(X))
        self.res = self.model.fit(q=self.q)
        
        self.params = self.res.params
        self.coef_ = self.res.params.iloc[1:]
        self.intercept_ = self.res.params.iloc[0]
        
        return self
    
    def predict(self, X):
        ypred = self.model.predict(self.res.params, sm.add_constant(X))
        
        return ypred

class QuantRegAnalyser(object):
    """
    计算单因子不同分位回归的系数
    """
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
    
    def quant_reg(self, q):
        m = QuantileRegressor(q)
        m.fit(self.X, self.y)
        res = m.res
        conf = res.conf_int()
        return q, res.params.iloc[0], res.params.iloc[1], conf.iloc[1, 0], conf.iloc[1, 1]
        
    def execute(self, iter_range):
        with Pool(processes=1) as pool:
            res = list(pool.map(self.quant_reg, iter_range))
        res = pd.DataFrame(res, index=iter_range, columns=['q', 'c', 'k', 'low', 'high'])
        return res


# 正则化加权回归
class WeightedRidge(BaseEstimator, RegressorMixin):
    """带L2正则化的加权回归
    通过对不同分位数区间、不同绝对值、不同时序的样本加权，改善模型预测效果
    这个类的目的是将权重（或计算权重的方法）在初始化模型时输入，无需在fit函数传入
    这样可以将model（或组成的pipeline）直接作为参数传入其他函数func，而不必在func中修改fit的实现。
    只带L2正则化是因为L1正则化在大数据量时大大减慢模型训练速度
    """
    def __init__(self, alpha=0, method: callable = None, args=(), sample_weight=None):
        """初始化函数
        Args:
            alpha (float, optional): 正则化系数，同linear_model.Ridge的alpha，取0时不加正则化
            method (callable, optional): 加权的实现方式. Defaults to None.
            args (tuple, optional): 传入method的参数. Defaults to ().
            sample_weight (optional): 指定的权重
        """
        self.alpha = alpha
        self.model = linear_model.Ridge(alpha=alpha)
        self.method = method
        self.args = args
        self.sample_weight = sample_weight

    def fit(self, X, y, sample_weight=None):
        
        # 设定权重
        if self.method is None: # 直接用给定的权重
            sample_weight = self.sample_weight
        # 根据method，计算权重
        elif self.method == "quant_reg": 
            sample_weight = WeightFunc.quant_reg(X, y, *self.args)
        elif self.method == "stair":
            sample_weight = WeightFunc.stair(X, y, *self.args)
        elif self.method == "section_stair":
            sample_weight = WeightFunc.section_stair(X, y, *self.args)
        elif self.method == "time_decay":
            sample_weight = WeightFunc.time_decay(X, y, *self.args)
        else:
            sample_weight = self.method(X, y, *self.args)
        
        # 训练模型
        self.model.fit(X, y, sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        return self.model.predict(X)

class WeightFunc:
    
    """预定义的加权函数，统一的接口为
    Args:
        X (_type_): 训练集特征X
        y (pd.Series): 训练集标签y
    Returns:
        np.ndarray: 权重
    """
    
    @staticmethod   # 类似Pipeline，将多个加权方式耦合，权重相乘
    def sequence(X, y, *args):
        weight = pd.Series(np.ones_like(y), index=y.index)
        for func, arg in args:
            weight *= func(X, y, *arg)    
        return weight.values
    
    @staticmethod   # 类比分位数回归的加权方式
    def quant_reg(X, y:pd.Series, *args):
        quantail = args[0]
        
        weight = y.rank(pct=True, ascending=True)
        weight[weight >= quantail] = quantail
        weight[weight < quantail] = 1-quantail
        
        return weight.values
    
    @staticmethod   # 对特定收益率分位区间（全部数据）的样本设置权重
    def stair(X, y:pd.Series, *args):
        
        rank = y.rank(pct=True, ascending=True)
        weight = pd.Series(np.ones_like(y), index=y.index)
        
        for low, high, w in args:
            weight[(rank >= low) & (rank < high)] = w
        
        return weight.values
    
    @staticmethod   # 对特定收益率分位区间（截面数据）的样本设置权重
    def section_stair(X, y:pd.Series, *args):
        
        rank = y.groupby('date').rank(pct=True, ascending=True)
        weight = pd.Series(np.ones_like(y), index=y.index)
        
        for low, high, w in args:
            weight[(rank >= low) & (rank < high)] = w
        
        return weight.values
    
    @staticmethod   # 时间指数衰变加权，越远的数据权重越小
    def time_decay(X, y:pd.Series, *args):
        
        lamb = args[0]
        k = args[1]
        
        weight = pd.Series(y.index.get_level_values(0).values).rank(ascending=False, pct=True).map(
            lambda x: np.e ** (-lamb * x ** k))
        
        return weight.values


