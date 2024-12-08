#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:36:37 2024

@author: yanchuanmiao
"""


import csv
import tensorflow as tf
import numpy as np
from scipy.stats import norm
from functions import functions
from utils import leapfrog, to_pickle, from_pickle
from get_args import get_args

args = get_args()

# 定义动力学函数，使用 TensorFlow 的自动微分
def dynamics_fn(t, coords):
    coords = tf.Variable(coords, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(coords)  # 监视坐标以便计算梯度
        H = functions(coords)  # 计算哈密顿量
    dcoords = tape.gradient(H, coords)  # 自动求导得到坐标的变化率

    # 将 dcoords 拆分为多个变量并进行处理
    dic1 = tf.split(dcoords, args.input_dim)
    S = dic1[int(args.input_dim / 2)]
    
    # 拼接剩余的梯度
    for ii in range(int(args.input_dim / 2) + 1, args.input_dim):
        S = tf.concat([S, dic1[ii]], axis=0)
    for ii in range(0, int(args.input_dim / 2)):
        S = tf.concat([S, -dic1[ii]], axis=0)
        
    return S

# 生成轨迹函数
def get_trajectory(t_span=[0, args.len_sample], timescale=args.len_sample, y0=None, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    if y0 is None:
        y0 = np.zeros(args.input_dim)
        for ii in range(0, int(args.input_dim / 2)):
            y0[ii] = norm(loc=0, scale=1).rvs()  # 正态分布取样

    # 使用 leapfrog 积分器计算系统的演化
    lp_ivp = leapfrog(dynamics_fn, t_span, y0, int(timescale * (t_span[1] - t_span[0])), args.input_dim)

    # 分解结果
    dic1 = np.split(lp_ivp, args.input_dim)
    dydt = [dynamics_fn(None, lp_ivp[:, ii]) for ii in range(0, lp_ivp.shape[1])]
    dydt = np.stack(dydt).T
    ddic1 = np.split(dydt, args.input_dim)

    return dic1, ddic1, t_eval

# 创建数据集函数
def get_dataset(seed=0, samples=args.num_samples, test_split=(1.0 - args.test_fraction), **kwargs):

    if args.should_load:
        path = '{}/{}.pkl'.format(args.load_dir, args.load_file_name)
        data = from_pickle(path)
        print("Successfully loaded data")
    else:
        data = {'meta': locals()}
        np.random.seed(seed)  # 设置随机种子
        xs, dxs = [], []
        
        y_init = np.zeros(args.input_dim)
        for ii in range(0, int(args.input_dim / 2)):
            y_init[ii] = 0.0  # 初始化前半部分
        for ii in range(int(args.input_dim / 2), args.input_dim):
            y_init[ii] = norm(loc=0, scale=1).rvs()  # 初始化后半部分，取自正态分布

        print('Generating HMC samples for DNN training')

        for s in range(samples):
            print('Sample number ' + str(s + 1) + ' of ' + str(samples))
            dic1, ddic1, t = get_trajectory(y0=y_init, **kwargs)
            
            # 将生成的轨迹数据拼接
            xs.append(np.stack([dic1[ii].T.reshape(len(dic1[ii].T)) for ii in range(0, args.input_dim)]).T)
            dxs.append(np.stack([ddic1[ii].T.reshape(len(ddic1[ii].T)) for ii in range(0, args.input_dim)]).T)
            
            # 更新 y_init 为最后时刻的值
            y_init = np.zeros(args.input_dim)
            for ii in range(0, int(args.input_dim / 2)):
                y_init[ii] = dic1[ii].T[-1]
            for ii in range(int(args.input_dim / 2), args.input_dim):
                y_init[ii] = norm(loc=0, scale=1).rvs()

        # 将数据整理为合适的格式
        data['coords'] = np.concatenate(xs)
        data['dcoords'] = np.concatenate(dxs).squeeze()

        # 进行训练集和测试集的划分
        split_ix = int(len(data['coords']) * test_split)
        split_data = {}
        for k in ['coords', 'dcoords']:
            split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
        data = split_data

        # 保存数据
        path = '{}/{}.pkl'.format(args.save_dir, args.dist_name)
        to_pickle(data, path)

    return data