#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Bo Dai

'''
Project:
    acc of SSVEP-BCI detectors

Versions:
    v1.0: 2021.10.21

'''

import numpy as np
from pandas.core.base import DataError
import scipy.io as sio
import itr
import matplotlib.pyplot as plt
from scipy.stats import sem

sub_num = 35
target_num = 40
len_gaze_s = 5
window_len = 1
# len_gaze_s = window_len
len_shift_s = 0.5
len_delay_s = 0.14
len_sel_s = len_gaze_s + len_shift_s
data_length_num = int(np.floor_divide(len_gaze_s, 0.25))
# 横坐标
X_t = np.zeros((data_length_num + 1))
for t_idx in range(0, 1 + int(np.floor_divide(len_gaze_s, 0.25))):
    t = t_idx * 0.25
    X_t[t_idx] = t

Data_path_acc = 'bin/results/without_training/acc/'
Data_path_itr = 'bin/results/without_training/itr/'

acc_cca_for_35sub = sio.loadmat(f'{Data_path_acc}acc_cca_benchmark{sub_num}subs.mat')
# acc_cca_for_35sub_arr = acc_cca_for_35sub['array']
acc_cca_for_35sub_arr = acc_cca_for_35sub['acc_cca_benchmark35subs']
acc_cca_for_35sub_arr = acc_cca_for_35sub_arr.T
acc_fbcca_for_35sub = sio.loadmat(f'{Data_path_acc}acc_fbcca_benchmark{sub_num}subs.mat')
acc_fbcca_for_35sub_arr = acc_fbcca_for_35sub['acc_fbcca_benchmark35subs']
acc_fbcca_for_35sub_arr = acc_fbcca_for_35sub_arr.T
acc_cnn_aggregate_for_35sub = sio.loadmat(f'{Data_path_acc}acc_cnn_aggregate_benchmark{sub_num}subs.mat')
acc_cnn_aggregate_for_35sub_arr = acc_cnn_aggregate_for_35sub['acc_cnn_aggregate_benchmark35subs']
acc_cnn_aggregate_for_35sub_arr = acc_cnn_aggregate_for_35sub_arr.T


itr_cca_for_35sub = sio.loadmat(f'{Data_path_itr}itr_cca_benchmark{sub_num}subs.mat')
itr_cca_for_35sub_arr = itr_cca_for_35sub['itr_cca_benchmark35subs']
itr_cca_for_35sub_arr = itr_cca_for_35sub_arr.T
itr_fbcca_for_35sub = sio.loadmat(f'{Data_path_itr}itr_fbcca_benchmark{sub_num}subs.mat')
itr_fbcca_for_35sub_arr = itr_fbcca_for_35sub['itr_fbcca_benchmark35subs']
itr_fbcca_for_35sub_arr = itr_fbcca_for_35sub_arr.T
itr_cnn_aggregate_for_35sub = sio.loadmat(f'{Data_path_itr}itr_cnn_aggregate_benchmark{sub_num}subs.mat')
itr_cnn_aggregate_for_35sub_arr = itr_cnn_aggregate_for_35sub['itr_cnn_aggregate_benchmark35subs']
itr_cnn_aggregate_for_35sub_arr = itr_cnn_aggregate_for_35sub_arr
'''
实验1：
    Title: 基于特定算法，不同用户, acc
    X: Data_length
    Y: Accuracy

'''

font1 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 16
}
font2 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 15
}

for subject in range(0, sub_num):
    plt.figure(1, figsize=(18,10))
    plt.plot(X_t, np.hstack((0.0, acc_cnn_aggregate_for_35sub_arr[subject, :])), '.-', label=f'sub_{subject+1}')
    plt.title(f'Accuracy of sub_{subject+1}')

    plt.figure(2, figsize=(18,10))
    plt.plot(X_t, np.hstack((0.0, itr_cnn_aggregate_for_35sub_arr[subject, :])), '.-', label=f'sub_{subject+1}')
    plt.title(f'ITR of sub_{subject+1}')
    
# acc
plt.figure(1)
plt.xticks(X_t)
y_start =0
y_end = 101
y_step = 10
y_ticks = np.array([i for i in range(y_start, y_end, y_step)])
plt.yticks(y_ticks)
# plt.title(f'Accuracy for all 35sub')
plt.legend(loc='upper left')
plt.savefig('bin/results/without_training/acc/acc_fbcca_for_benchmark_35subs.png')
plt.show()
# itr
plt.figure(2)
plt.xticks(X_t)
plt.legend(loc='lower right')
plt.savefig('bin/results/without_training/itr/itr_fbcca_for_benchmark_35subs.png')
plt.show()

# 用户间平均结果
# acc
plt.figure(3, figsize=(16, 8))
# print('np.size(acc_cnn_aggregate_for_35sub_arr)', np.size(acc_cnn_aggregate_for_35sub_arr, axis=0))
y_std_cnn = np.std(acc_cnn_aggregate_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(acc_cnn_aggregate_for_35sub_arr, axis=0))
y_std_cca = np.std(100.0 * acc_cca_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(100.0 * acc_cca_for_35sub_arr, axis=0))
y_std_fbcca = np.std(100.0 * acc_fbcca_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(100.0 * acc_fbcca_for_35sub_arr, axis=0))

# print('Mean acc for cca: ', 100.0 * np.mean(acc_cca_for_35sub_arr, axis=0))
# print('Mean acc for fbcca: ', 100.0 * np.mean(acc_fbcca_for_35sub_arr, axis=0))

plt.plot(X_t, np.hstack((0.0, np.mean(acc_cnn_aggregate_for_35sub_arr, axis=0))), '.-', label='acc_cnn_aggregate_mean_for_35subs', linewidth=2, color='r') 
plt.fill_between(X_t, np.hstack((0.0, np.mean(acc_cnn_aggregate_for_35sub_arr, axis=0) - y_std_cnn)), np.hstack((0.0, np.mean(acc_cnn_aggregate_for_35sub_arr, axis=0) + y_std_cnn)), alpha=.1, color='r')
plt.plot(X_t, np.hstack((0.0, 100.0 * np.mean(acc_fbcca_for_35sub_arr, axis=0))), '.-', label='acc_fbcca_mean_for_35subs', linewidth=2, color='b')
plt.fill_between(X_t, np.hstack((0.0, 100.0 * np.mean(acc_fbcca_for_35sub_arr, axis=0) - y_std_fbcca)), np.hstack((0.0, 100.0 * np.mean(acc_fbcca_for_35sub_arr, axis=0) + y_std_fbcca)), alpha=.1, color='b')
# plt.plot(X_t, np.hstack((0.0, 100.0 * np.mean(acc_cca_for_35sub_arr, axis=0))), '.-', label='acc_cca_mean_for_35subs', linewidth=2, color='g')  
# plt.fill_between(X_t, np.hstack((0.0, 100.0 * np.mean(acc_cca_for_35sub_arr, axis=0) - y_std_cca)), np.hstack((0.0, 100.0 * np.mean(acc_cca_for_35sub_arr, axis=0) + y_std_cca)), alpha=.1, color='g')

plt.xticks(X_t, fontsize=14)
y_start =0
y_end = 101
y_step = 10
y_ticks = np.array([i for i in range(y_start, y_end, y_step)])
plt.yticks(y_ticks, fontsize=14)

# plt.title('Mean accuracy for 35subs Benchmark', fontsize=20)
plt.legend(loc='lower right',prop=font1)
plt.savefig('bin/results/without_training/acc/acc_mean_for_benchmark_35subs.png')
plt.show()
# itr
plt.figure(4, figsize=(16, 8))
itr_std_cnn = np.std(itr_cnn_aggregate_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(itr_cnn_aggregate_for_35sub_arr, axis=0))
itr_std_cca = np.std(itr_cca_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(itr_cca_for_35sub_arr, axis=0))
itr_std_fbcca = np.std(itr_fbcca_for_35sub_arr, axis=0, ddof=1) / np.sqrt(np.size(itr_fbcca_for_35sub_arr, axis=0))

plt.plot(X_t, np.hstack((0.0, np.mean(itr_cnn_aggregate_for_35sub_arr, axis=0))), '.-', label='itr_cnn_aggregate_mean_for_35subs', linewidth=2, color='r') 
plt.fill_between(X_t, np.hstack((0.0, np.mean(itr_cnn_aggregate_for_35sub_arr, axis=0) - itr_std_cnn)), np.hstack((0.0, np.mean(itr_cnn_aggregate_for_35sub_arr, axis=0) + itr_std_cnn)), alpha=.1, color='r')
plt.plot(X_t, np.hstack((0.0, np.mean(itr_fbcca_for_35sub_arr, axis=0))), '.-', label='itr_fbcca_mean_for_35subs', linewidth=2, color='b')  
plt.fill_between(X_t, np.hstack((0.0, np.mean(itr_fbcca_for_35sub_arr, axis=0) - itr_std_fbcca)), np.hstack((0.0, np.mean(itr_fbcca_for_35sub_arr, axis=0) + itr_std_fbcca)), alpha=.1, color='b')
# plt.plot(X_t, np.hstack((0.0, np.mean(itr_cca_for_35sub_arr, axis=0))), '.-', label='itr_cca_mean_for_35subs', linewidth=2, color='g')  
# plt.fill_between(X_t, np.hstack((0.0, np.mean(itr_cca_for_35sub_arr, axis=0) - itr_std_cca)), np.hstack((0.0, np.mean(itr_cca_for_35sub_arr, axis=0) + itr_std_cca)), alpha=.1, color='g')

plt.xticks(X_t, fontsize=14)
plt.yticks(fontsize=14)
# plt.title('Mean itr for 35subs Benchmark', fontsize=16)

plt.legend(loc='upper right', prop=font1)
plt.savefig('bin/results/without_training/itr/itr_mean_for_benchmark_35subs.png')
plt.show()

'''
实验2：
    Title: 基于同一用户，不同算法
    X: 用户
    Y: acc、itr

'''  








