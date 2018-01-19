#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:04:03 2018

@author: lufei
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import gc, sys
sys.path.append('tools')
from tools import result_split

def xgboost(train, test, feature, label, save_file):
    x_train, x_test, y_train, y_test = train_test_split(train[feature], train.Y, 
                                                        test_size=0.2)
    watch_train = xgb.DMatrix(x_train, y_train)
    watch_test = xgb.DMatrix(x_test, y_test)
    watch_list = [(watch_train, 'train'),  (watch_test, 'test')]
    params = {
            'objective': 'reg:linear',
            #'gamma': 0.1,
            #'max_depth': 6,
            #'subsample': 0.5,
            #'eta': 0.2,
            #'lambda': 0,
            'seed': 0,
            #'min_child_weight': 1,
            'nthread': 2,
            'slient': 1,
            'eval_metric': 'rmse'
            }
    num_rounds = 50
    xgbtrain = xgb.DMatrix(train[feature], train.Y)
    xgbtest = xgb.DMatrix(test[feature])
    model = xgb.train(params, xgbtrain, num_rounds, watch_list)
    result = test[['ID']]
    result['predict'] = model.predict(xgbtest)
    result.to_csv(save_file, index=False)

def gbdt(train, feature, label):
    x_train, x_test, y_train, y_test = train_test_split(train[feature], 
                                                        train.Y, test_size=0.2)
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    start = 0
    for i in range(1, 1001):
        if i % 100 == 0:
            print('MSE: {0}'.format(mean_squared_error(y_test[start: i - 1], result[start: i - 1])))
            start = i
    return model

def rf(train, test, feature, label, save_file, cv=False):
    model = RandomForestRegressor()
    if cv:
        print('cross validation scroing...')
        score = cross_validate(model, train[feature],
            train[label], scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        print(score)
    model.fit(train[feature], train[label])
    result = test[['ID']]
    result['predict'] = model.predict(test[feature])
    result.to_csv(save_file, index=False)
    
    

data = pd.read_csv('data/data2.0_expand.csv')
train = data[data.Y.notnull()]
test = data[data.Y.isnull()]
feature = [c for c in train.columns if c not in ['ID', 'Y']]
label = 'Y'
# gbdt_model = gbdt(train, feature, label)
# xgboost(train, test, feature, label, 'result2.1_AB.csv', True)
rf(train, test, feature, label, 'result2.0_AB_rf.csv', True)


