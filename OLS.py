
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


# 分位数回归
class QuantReg(BaseEstimator, RegressorMixin):
    """
    基于statsmodels实现，套sklearn壳的分位数回归
    大数据量时速度较快
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
        m = QuantReg(q)
        m.fit(self.X, self.y)
        res = m.res
        conf = res.conf_int()
        return q, res.params.iloc[0], res.params.iloc[1], conf.iloc[1, 0], conf.iloc[1, 1]
        
    def execute(self, iter_range):
        with Pool(processes=1) as pool:
            res = list(pool.map(self.quant_reg, iter_range))
        res = pd.DataFrame(res, index=iter_range, columns=['q', 'c', 'k', 'low', 'high'])
        return res


# 加权回归
class WLS(BaseEstimator, RegressorMixin):
    def __init__(self, method: tuple = None) -> None:
        self.method = method
        super().__init__()
        
    def fit(self, X, y, w=None):
        if w is None:
            if isinstance(self.method[0], str):
                if self.method[0] == 'quant_reg':
                    w = self.w_quant_reg(X, y, *self.method[1])
        
        self.model = sm.WLS(y, sm.add_constant(X), w)
        self.res = self.model.fit()
        
        self.params = self.res.params
        self.coef_ = self.res.params.iloc[1:]
        self.intercept_ = self.res.params.iloc[0]
        
        return self
    
    def predict(self, X):
        ypred = self.model.predict(self.res.params, sm.add_constant(X))
        
        return ypred
    
    @staticmethod   # 类比分位数回归的加权方式
    def w_quant_reg(X, y, *args):
        quantail = args[0]
        
        weight = y.rank(pct=True, ascending=True)
        weight[weight >= quantail] = quantail
        weight[weight < quantail] = 1-quantail
        
        return weight


# 正则化加权回归
class WeightedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, method: callable = None, args=()):
        self.alpha = alpha
        self.model = linear_model.Ridge(alpha=alpha)
        self.method = method
        self.args = args

    def fit(self, X, y, sample_weight=None):
        if self.method is not None:
            sample_weight = self.method(X, y, *self.args)
        elif sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        # 验证样本权重的长度
        if len(sample_weight) != X.shape[0]:
            raise ValueError("sample_weight must be of the same length as X")

        # 应用权重
        W_sqrt = np.sqrt(sample_weight)
        X_weighted = X * W_sqrt[:, np.newaxis]
        y_weighted = y * W_sqrt

        self.model.fit(X_weighted, y_weighted)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        # 可选：实现加权评分方法
        return super().score(X, y, sample_weight=sample_weight)
    
    @staticmethod
    def w_combine(X, y, *args):
        weight = pd.Series(np.ones_like(y), index=y.index)
        for func, arg in args:
            weight *= func(X, y, *arg)
        
        return weight.values
        
    
    @staticmethod   # 类比分位数回归的加权方式
    def w_quant_reg(X, y, *args):
        quantail = args[0]
        
        weight = y.rank(pct=True, ascending=True)
        weight[weight >= quantail] = quantail
        weight[weight < quantail] = 1-quantail
        
        return weight.values
    
    @staticmethod
    def w_staired(X, y:pd.Series, *args):
        
        rank = y.rank(pct=True, ascending=True)
        weight = pd.Series(np.zeros_like(y), index=y.index)
        
        for low, high, w in args:
            weight[(rank >= low) & (rank < high)] = w
        
        return weight.values
    
    @staticmethod
    def w_section_staired(X, y:pd.Series, *args):
        
        rank = y.groupby('date').rank(pct=True, ascending=True)
        weight = pd.Series(np.ones_like(y), index=y.index)
        
        for low, high, w in args:
            weight[(rank >= low) & (rank < high)] = w
        
        return weight.values
    
    @staticmethod
    def w_time_decay(X, y:pd.Series, *args):
        
        lamb = args[0]
        k = args[1]
        
        weight = pd.Series(y.index.get_level_values(0).values).rank(ascending=False, pct=True).map(
            lambda x: np.e ** (-lamb * x ** k))
        
        return weight.values
        


if __name__ == "__main__":
    (data_train, x_train, y_train), (data_test, x_test, y_test) = qb.Tools.train_test_split_basic()
    data = pd.read_feather('data/data.feather')
    # for i in trange(105, 160):
    #     col = 'x' + str(i+1)
    #     ana = QuantRegAnalyser(X=x_train[col], y=y_train)
    #     res = ana.execute(np.linspace(0.05, 0.95, 10))
    #     res.to_csv(f'data/QuantReg/{col}.csv', )
    
    def func(q, ypreds):
        model = qb.OLS.QuantReg(q=q)
        model.fit(x_train, y_train)
        # ypreds[q] = np.concatenate([model.predict(x_train), model.predict(x_test)])
        ypreds[q] = model.params
        
    with mp.Manager() as manager:
        ypreds = manager.dict()
        p1 = mp.Process(target=func, args=(0.1, ypreds)); p1.start(); print(p1.pid)
        p2 = mp.Process(target=func, args=(0.2, ypreds)); p2.start(); print(p2.pid)
        p3 = mp.Process(target=func, args=(0.3, ypreds)); p3.start(); print(p3.pid)
        p4 = mp.Process(target=func, args=(0.4, ypreds)); p4.start(); print(p4.pid)
        p5 = mp.Process(target=func, args=(0.5, ypreds)); p5.start(); print(p5.pid)
        
        p1.join(); p2.join(); p3.join(); p4.join(); p5.join()
        
        df = pd.DataFrame()
        df[0.1] = ypreds[0.1]
        df[0.2] = ypreds[0.2]
        df[0.3] = ypreds[0.3]
        df[0.4] = ypreds[0.4]
        df[0.5] = ypreds[0.5]
        print(df)
        
    
    df.to_feather('data/QuantReg/full0.feather')