# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler

def expend(data: 'pending dataframe', 
           string_feature: "do nothing for feature'list", 
           digital_feature: "need to process feature'list", 
           multiple: "need to expend the multiple", 
           noise: "noise's proportion" = 0.1) -> 'dataframe':
    '''
    给样本加噪声来构造更多的样本，只给数字特征加噪声，
    噪声比例可进行设置。对输入数据加噪声构成一个矩阵，
    可按需设置多个噪声矩阵，最后噪声矩阵会和原样本进行合并，然后返回。
    '''
    string = data[string_feature]
    digital = data[digital_feature]
    del data
    gc.collect()
    temp_digital = pd.DataFrame()
    temp_string = pd.DataFrame()
    for i in range(multiple):
        temp = digital.as_matrix() + (np.random.random(digital.shape) - 0.5)*2*noise
        temp = pd.DataFrame(temp)
        temp.columns = digital.columns
        temp_digital = pd.concat([temp_digital, temp], axis=0)
        temp_string = pd.concat([temp_string, string], axis=0)
    digital = pd.concat([digital, temp_digital], axis=0)
    string = pd.concat([string, temp_string], axis=0)
    del temp_string, temp_digital
    gc.collect()
    data = pd.concat([string, digital], axis=1)
    return data

def digital_scaler(digital: 'dataframe' ) -> 'dataframe':
    '''
    对数字型特征进行缩放，范围0-1之间。
    输入dataframe，返回缩放后的dataframe。
    '''
    columns = digital.columns.tolist()
    scaler = MinMaxScaler()
    scaler.fit(digital)
    digital = pd.DataFrame(scaler.transform(digital))
    digital.columns = columns
    return digital

def split_result(ab_file: 'result file path', 
                 res: "a or b" = 'a'):
    '''
    分割结果文件，之前的结构文件包含A，B榜，
    可把A榜或B榜数据分出来做一个文件进行提交。
    '''
    data = pd.read_csv(ab_file, index_col='ID')
    res_file = 'input/测试A.xlsx' if res == 'a' else 'input/测试B.xlsx'
    result = pd.read_excel(res_file, usecols=['ID'])
    result = data.loc[result.ID.tolist(), :]
    suffix = '_A' if res == 'a' else '_B'
    save_file = ab_file.split('_')[0]
    save_file = save_file + suffix + '.csv'
    result.to_csv(save_file, header=None)