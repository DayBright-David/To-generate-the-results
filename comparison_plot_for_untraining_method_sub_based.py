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
实验2：
    Title: 基于同一用户，不同算法
    X: 用户
    Y: acc、itr

'''  
labels = ['S{}'.format(i) for i in range(1, sub_num+1)]
acc_cnn = acc_cnn_aggregate_for_35sub_arr[:, 4]
acc_fbcca = 100.0 * acc_fbcca_for_35sub_arr[:, 4]

x = np.arange(len(labels)) + 1.0    # the label location
width = 0.35    # the width of the bar

fig, ax = plt.subplots(1, figsize=(16, 8))
rect_fbcca = ax.bar(x - width/2, acc_fbcca, width, label='fbcca')
rect_cnn = ax.bar(x + width/2, acc_cnn, width, label='CNN')

# Add some text for labels, titles, and custom x-axis ticks labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('Subject number')
# ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# def autolabel(rects):
#     '''
#     Attach a text label above each bar in *rect, displaying its height.
#     '''
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     # textcoords='offset points',
#                     ha='center', va='bottom'

#         )

# autolabel(rect_cnn)
# autolabel(rect_fbcca)
fig.tight_layout()

plt.savefig('bin/results/without_training/acc/acc_for_subs_benchmark.png')
plt.show()


