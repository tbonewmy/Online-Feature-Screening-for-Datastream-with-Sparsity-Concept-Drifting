# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 22:24:30 2025

Wang, M. and Barbu, A., 2022. 
Online feature screening for data streams with concept drift. 
IEEE Transactions on Knowledge and Data Engineering, 35(11), 
pp.11693-11707.

@author: Mingyuan Wang
"""
import pandas as pd
import numpy as np
from pyscreeningfs import online_screening

data = pd.read_csv('./data/Day0_output.csv', header=None)
X = data.iloc[:,1:50].values
Y = data.iloc[:,0].values.reshape(-1,1)

if not X.flags['F_CONTIGUOUS']:
    X = np.asfortranarray(X, dtype=np.float64)
Y = Y.astype(np.float64)
if not Y.flags['F_CONTIGUOUS']:
    Y = np.asfortranarray(Y, dtype=np.float64)
       
# quantile summary
# 1.has bin, is sparse, is drifting
gotit = online_screening(is_sparse=True, base_engine='quantile', batch_alpha=0.9, 
                         data_file_folder_path='./data/url_svmlight',
                         main_data_file_name='Day', file_extension='svm', num_total_files=5)
# 2.has bin, is sparse, is not drifting
gotit = online_screening(is_sparse=True, base_engine='quantile', 
                         data_file_folder_path='./data/url_svmlight',
                         main_data_file_name='Day', file_extension='svm', num_total_files=5)
# 3.has bin, is not sparse, is drifting
gotit = online_screening(is_sparse=False, base_engine='quantile', 
                         batch_alpha=0.9, batch_penalty_freq=1, X=X, Y=Y, batch_size=1000)
# 4.has bin, is not sparse, is not drifting
gotit = online_screening(is_sparse=False, base_engine='quantile', 
                         X=X, Y=Y, batch_size=1000)

# moving average
# 1.no bin, is sparse, is drifting
gotit = online_screening(is_sparse=True, base_engine='moving_average', batch_alpha=0.9, 
                         data_file_folder_path='./data/url_svmlight',
                         main_data_file_name='Day', file_extension='svm', num_total_files=5)
# 2.no bin, is sparse, is not drifting 
gotit = online_screening(is_sparse=True, base_engine='moving_average', 
                         data_file_folder_path='./data/url_svmlight',
                         main_data_file_name='Day', file_extension='svm', num_total_files=5)
# 3.no bin, is not sparse, is drifting
gotit = online_screening(is_sparse=False, base_engine='moving_average', 
                         batch_alpha=0.9, batch_penalty_freq=1, X=X, Y=Y, batch_size=1000)
# 4.no bin, is not sparse, is not drifting
gotit = online_screening(is_sparse=False, base_engine='moving_average', 
                         X=X, Y=Y, batch_size=1000)
