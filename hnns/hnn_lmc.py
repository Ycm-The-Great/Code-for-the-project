#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:27:40 2024

@author: yanchuanmiao
"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import pickle

# 导入自定义的模型和工具函数
from nn_models import MLP
from hnn import HNN
from get_args import get_args
from utils import leapfrog
from functions import functions
import argparse

# # 获取命令行参数
# args = get_args()

# ##### 用户定义的采样参数 #####
# chains = 1  # 马尔科夫链数量
# N = 10000  # 每条链的采样数量
# epsilon = 0.025  # 步长
# burn = 1000  # burn-in 样本数量

##### 采样代码 #####

# 定义加载模型的函数
def get_model(args, baseline):
    output_dim = args.input_dim
    # 使用 MLP 作为基础模型
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    # 包装成 Hamiltonian Neural Network (HNN)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=baseline)
    
    # 初始化模型变量
    dummy_input = tf.zeros((1, args.input_dim))    # 创建虚拟输入
    _ = model(dummy_input)  # 调用模型，初始化变量 
    # 加载模型权重，TensorFlow 使用 .h5 格式保存模型
    path = args.dist_name + ".h5"
    model.load_weights(path)  # TensorFlow 使用 load_weights() 加载权重
    return model

# 定义使用 leapfrog 方法进行积分的函数
def integrate_model(model, t_span, y0, n, args, **kwargs):
    # 定义动力学函数
    def fun(t, np_x):
        # 将 NumPy 数组转换为 TensorFlow 张量，并求解模型的时间导数
        x = tf.convert_to_tensor(np_x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            dx = model.time_derivative(x)
        return dx.numpy().reshape(-1)  # 将结果转换为 NumPy 数组
    # 使用 leapfrog 积分方法
    return leapfrog(fun, t_span, y0, n, args.input_dim)


def hnn_lmc_sampling(args, control_group):
    # 初始化时间跨度和步数
    chains = control_group.chains  # 马尔科夫链数量
    N = control_group.N  # 每条链的采样数
    L = control_group.L  # 每条链的哈密顿轨迹长度
    burn = control_group.burn  # 丢弃的burn-in样本数量
    epsilon = control_group.epsilon  # 时间积分步长
    t_span = [0, epsilon]
    steps = 2  # Langevin 动力学每次只进行一个小步
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
    
    # 加载 HNN 模型
    hnn_model = get_model(args, baseline=False)
    
    # 初始化状态向量 y0
    y0 = np.zeros(args.input_dim)
    
    # 初始化存储采样结果的数组
    hnn_fin = np.zeros((chains, N, int(args.input_dim / 2)))
    hnn_accept = np.zeros((chains, N))
    
    # 对每条链进行采样
    for ss in tqdm(np.arange(0, chains, 1)):
        x_req = np.zeros((N, int(args.input_dim / 2)))  # 存储接受的样本
        x_req[0, :] = y0[0:int(args.input_dim / 2)]
        accept = np.zeros(N)  # 存储每个样本的接受状态
    
        # 初始化 y0 的前半部分为 0，后半部分从正态分布中采样
        for ii in np.arange(0, int(args.input_dim / 2), 1):
            y0[ii] = 0.0
        for ii in np.arange(int(args.input_dim / 2), int(args.input_dim), 1):
            y0[ii] = norm(loc=0, scale=1).rvs()
    
        # 用于存储哈密顿轨迹的数组
        HNN_sto = np.zeros((args.input_dim, 1, N))
    
        # 进行 N 次采样
        for ii in tqdm(np.arange(0, N, 1)):
            # 使用 leapfrog 方法进行哈密顿动力学积分
            hnn_ivp = integrate_model(hnn_model, t_span, y0, steps - 1,args, **kwargs)
            
            # 存储积分的最后一步
            for sss in range(0, args.input_dim):
                HNN_sto[sss, :, ii] = hnn_ivp[sss, 1]
    
            # 计算新的哈密顿量
            yhamil = np.zeros(args.input_dim)
            for jj in np.arange(0, args.input_dim, 1):
                yhamil[jj] = hnn_ivp[jj, 1]
    
            # 计算新的哈密顿量和之前哈密顿量的差值
            H_star = functions(yhamil)  # 新的哈密顿量
            H_prev = functions(y0)  # 之前的哈密顿量
    
            # Metropolis-Hastings 接受率
            alpha = np.minimum(1, np.exp(H_prev - H_star))
    
            # 如果接受了新的样本，更新 y0
            if alpha > uniform().rvs():
                y0[0:int(args.input_dim / 2)] = hnn_ivp[0:int(args.input_dim / 2), 1]
                x_req[ii, :] = hnn_ivp[0:int(args.input_dim / 2), 1]
                accept[ii] = 1
            else:
                x_req[ii, :] = y0[0:int(args.input_dim / 2)]
    
            # 每次更新后，重新采样动量部分
            for jj in np.arange(int(args.input_dim / 2), args.input_dim, 1):
                y0[jj] = norm(loc=0, scale=1).rvs()
    
            #print(f"Sample: {ii} Chain: {ss}")
    
        # 存储当前链的接受率和采样结果
        hnn_accept[ss, :] = accept
        hnn_fin[ss, :, :] = x_req
    
    # 计算有效样本大小 (ESS)
    ess_hnn = np.zeros((chains, int(args.input_dim / 2)))
    for ss in np.arange(0, chains, 1):
        # 转换为 TensorFlow 张量
        hnn_tf = tf.convert_to_tensor(hnn_fin[ss, burn:N, :], dtype=tf.float32)
        # 使用 TensorFlow Probability 计算 ESS
        ess_hnn[ss, :] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    # 输出有效样本大小
    print("Effective Sample Size (ESS):", ess_hnn)
    total_gradient_evaluations = steps * N

    avg_ess = np.sum(ess_hnn)/total_gradient_evaluations
    print("Effective Sample Size (ESS):", ess_hnn)
    print("total_gradient_evaluations:", total_gradient_evaluations)
    print("Avg ESS per gradient", avg_ess)
    
    result = {
       "samples": hnn_fin,
       "effective_sample_sizes": ess_hnn,
       "total_gradient_evaluations": total_gradient_evaluations,
       "Avg ESS per gradient": avg_ess
   }
    print(result)
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，就创建它
    filename = os.path.join(save_dir, f"hnn_lmc_{args.dist_name}_{args.input_dim}.pkl")

    # 保存为 Pickle 文件
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    
    print(f"Results saved to {filename}")
    
    return result


def load_json_config(json_file):
    """
    从 JSON 文件加载参数并返回字典
    """
    with open(json_file, "r") as f:
        return json.load(f)

def get_control_group_lmc():
    parser = argparse.ArgumentParser(description="Control Group Parameters", add_help=False)

    # 定义参数
    parser.add_argument("--chains", type=int, default=1, help="Number of Markov chains")
    parser.add_argument("--epsilon", type=float, default=0.025, help="Time integration step size")
    parser.add_argument("--N", type=int, default=10000, help="Number of samples per chain")
    parser.add_argument("--L", type=int, default=10, help="Length of Hamiltonian trajectories")
    parser.add_argument("--burn", type=int, default=1000, help="Number of burn-in samples to discard")
    parser.add_argument("--config", type=str, help="Path to JSON config file", default=None)  # 可选的自定义配置文件路径

    # 解析命令行参数
    args = parser.parse_args()

    # 如果未指定 config 文件路径，默认使用 "config_lmc.json"
    config_path = args.config if args.config else "config_lmc.json"

    # 检查配置文件是否存在
    if os.path.exists(config_path):
        try:
            # 加载配置文件内容
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # 更新解析后的命令行参数
            for key, value in config.items():
                if hasattr(args, key):  # 仅覆盖已定义的参数
                    setattr(args, key, value)
        except Exception as e:
            print(f"Warning: Failed to load or parse {config_path}: {e}")
    
    return args
    
if __name__ == "__main__":
    args = get_args()
    control_group = get_control_group_lmc()
    hnn_lmc_sampling(args, control_group)