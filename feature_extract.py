#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 19:51:33 2018

@author: lufei
"""
import pandas as pd
import gc, sys
sys.path.append('tools')
from tools import expend, digital_scaler


# =============================================================================
# # data2.0_expand.csv
# # 用gbdt筛选出有用的特征，用这些特征构造样本。
# feature = pd.read_csv('data/data_feature_importances.csv')
# feature = feature[feature.importances > 0].feature.tolist()
# feature.remove('TOOL_ID (#2)_C')
# feature.extend(['ID', 'Y', 'TOOL_ID (#2)'])
# data = pd.read_csv('data/data_fillna_median.csv', usecols=feature)
# string_feature = [c for c in data.columns if type(data[c][0]) == str]
# string_feature.append('Y')
# digital_feature = [c for c in data.columns if c not in string_feature]
# # TOOL_ID (#2)的值是否是C
# data['TOOL_ID (#2)'] = data['TOOL_ID (#2)'].apply(lambda x: 1 if x == 'C' else 0)
# train = data[data.Y.notnull()]
# test = data[data.Y.isnull()]
# digital = train[digital_feature]
# string = train[string_feature]
# del data, train
# gc.collect()
# digital = digital_scaler(digital)    
# data = pd.concat([string, digital], axis=1)
# del string, digital
# gc.collect()
# data = expend(data, string_feature, digital_feature, 9, 0.1)
# data = pd.concat([data, test], axis=0)
# data.to_csv('data/data2.0_expand.csv', index=False)
# =============================================================================
        
# =============================================================================
# # data1.4_expend.csv
# # 把小于9个值的列进行onehot，不管是否是数值。
# data = pd.read_csv('data/data_fillna_median.csv')
# train = data[data.Y.notnull()]
# test = data[data.Y.isnull()]
# n = pd.read_csv('data/data_nunique.csv', index_col=[0], header=None)
# # 选取9个值以下的特征
# n = n[n.iloc[:, 0] <= 9]
# string_feature = n.index.tolist()
# string_feature.extend(['Y', 'ID'])
# digital_feature = [c for c in train.columns if c not in string_feature]
# train = expend(train, string_feature, digital_feature, 9, 0.1)
# data = pd.concat([train, data[data.Y.isnull()]], axis=0)
# data.to_csv('data/data1.4_expend.csv', index=False)
# =============================================================================

# =============================================================================
# # data_nunique.csv
# # 对列的属性值进行计数。
# data = pd.read_csv('data/data_fillna_median.csv')
# n = data.nunique()
# n.to_csv('data/data_nunique.csv')
# =============================================================================

# =============================================================================
# # data_fillna_median.csv
# # 用列中位数填充缺失值，只有数值型特征有缺失值。
# data = pd.read_csv('data/data_del_single.csv')
# string_feature = [c for c in data.columns if type(data[c][0]) == str]
# digital_feature = [c for c in data.columns if c not in string_feature]
# digital_feature.remove('Y')
# digital = data[digital_feature]
# digital = digital.fillna(digital.median())
# save_data = pd.concat([data[string_feature], digital, data.Y], axis=1)
# save_data.to_csv('data/data_fillna_median.csv', index=False)
# =============================================================================

# =============================================================================
# # data1.3_expend.csv
# data = pd.read_csv('data/data1.3.csv')
# train = data[data.Y.notnull()]
# string_feature = [c for c in train.columns if type(train[c][0]) == str]
# string_feature.append('Y')
# digital_feature = [c for c in train.columns if c not in string_feature]
# train = expend(train, string_feature, digital_feature, 9)
# data = pd.concat([train, data[data.Y.notnull()]], axis=0)
# data.to_csv('data/data1.3_expend.csv', index=False)
# =============================================================================

# =============================================================================
# # data1.3.csv
# # 把样本数值缩放到0-1之间。
# data = pd.read_csv('data/data_fillna_mean.csv')
# string_feature = [c for c in data.columns if type(data[c][0]) == str]
# digital_feature = [c for c in data.columns if c not in string_feature]
# digital_feature.remove('Y')
# digital = data[digital_feature]
# digital_columns = list(digital.columns)
# scaler = MinMaxScaler()
# scaler.fit(digital)
# digital = scaler.transform(digital)
# digital = pd.DataFrame(digital)
# digital.columns = digital_columns
# data = pd.concat([data[string_feature], digital, data.Y], axis=1)
# data.to_csv('data/data1.3.csv', index=False)
# =============================================================================

# =============================================================================
# # data_fillna_mean.csv
# # 用列均值填充缺失值。只有数值型特征有缺失值，所以只用给数值型特征填充。
# data = pd.read_csv('data/data_del_single.csv')
# string_feature = [c for c in data.columns if type(data[c][0]) == str]
# digital_feature = [c for c in data.columns if c not in string_feature]
# digital_feature.remove('Y')
# digital = data[digital_feature]
# digital = digital.fillna(digital.mean())
# save_data = pd.concat([data[string_feature], digital, data.Y], axis=1)
# save_data.to_csv('data/data_fillna_mean.csv', index=False)
# =============================================================================

# =============================================================================
# # data.csv
# # 把input里的训练，测试A榜和B榜的样本都放到一起，用csv保存，读取xlsx太慢了。
# train = pd.read_excel('input/训练.xlsx')
# test_A = pd.read_excel('input/测试A.xlsx')
# test_B = pd.read_excel('input/测试B.xlsx')
# data = pd.concat([train, test_A, test_B], axis=0)
# data.to_csv('data/data.csv', index=False)
# =============================================================================

# =============================================================================
# # data_del_single.csv
# # 把只有一个或没有属性的列删除，留下删除后的样本。 
# data = pd.read_csv('data/data.csv')
# del_c = [c for c in data.columns if data[c].nunique() <= 1] # 需要删除的列
# save_feature = [c for c in data.columns if c not in del_c]
# data[save_feature].to_csv('data/data_del_single.csv', index=False)
# =============================================================================
