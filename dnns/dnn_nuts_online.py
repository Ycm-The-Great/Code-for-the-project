#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:29:55 2024

@author: yanchuanmiao
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, uniform
from functions import functions  # Assuming this module has user-defined functions
from utils import leapfrog  # Leapfrog integration logic
from get_args import get_args
from data import dynamics_fn  # Function defining dynamics for integration
from nn_models import MLP
from hnn import HNN
from tqdm import tqdm
import argparse
import json
import os
import pickle
# # Get arguments
# args = get_args()

# ##### User-defined sampling parameters #####
# N = 1000  # Number of samples
# burn = 100  # Number of burn-in samples
# epsilon = 0.025  # Step size
# N_lf = 20  # Number of cool-down samples when DNN integration errors are high (see https://arxiv.org/abs/2208.06120)
# hnn_threshold = 10.0  # DNN integration error threshold (see https://arxiv.org/abs/2208.06120)
# lf_threshold = 1000.0  # Numerical gradient integration error threshold

##### TensorFlow model loading and integration function #####

# Define the model creation function
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

# Integration function using leapfrog method
def integrate_model(model, t_span, y0, n,args, **kwargs):
    def fun(t, np_x):
        x = tf.Variable(np_x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            dx = model.time_derivative(x).numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

# hnn_model = get_model(args, baseline=True)

##### Hamiltonian Monte Carlo sampling functions #####

# Function to compute a slice from Hamiltonian value
def compute_slice(h_val):
    uni1 = uniform(loc=0, scale=np.exp(-h_val)).rvs()
    return np.log(uni1)

# Stopping criterion for the tree expansion in NUTS
def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)

# Recursive function to build the tree in the No-U-Turn Sampler
# def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf):
#     if j == 0:
#         # Base case: Integrate the system
#         t_span1 = [0, v * epsilon]
#         kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
#         y1 = np.concatenate((theta, r), axis=0)
#         hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, 1, **kwargs1)
#         thetaprime = hnn_ivp1[:int(args.input_dim / 2),1].reshape(-1)
#         rprime = hnn_ivp1[int(args.input_dim / 2):,1].reshape(-1)        
#         joint = functions(hnn_ivp1[:,1])
#         #call_lf = call_lf or int((np.log(logu) + joint) > 10.0)
#         call_lf = call_lf or int((np.log(logu) + joint.numpy()) > 10.0)
#         monitor = np.log(logu) + joint
#         sprime = int(monitor <= hnn_threshold)
        
#         if call_lf:
#             hnn_ivp1 = leapfrog(dynamics_fn, t_span1, y1, 1, args.input_dim)
#             thetaprime = hnn_ivp1[:int(args.input_dim / 2)].reshape(-1)
#             rprime = hnn_ivp1[int(args.input_dim / 2):].reshape(-1)
#             joint = functions(hnn_ivp1)
#             sprime = int(monitor <= lf_threshold)

#         nprime = int(logu <= np.exp(-joint))
#         thetaminus, thetaplus = thetaprime, thetaprime
#         rminus, rplus = rprime, rprime
#         alphaprime = min(1.0, np.exp(joint0 - joint))
#         nalphaprime = 1
#     else:
#         # Recursive case: build left and right subtrees
#         thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree(theta, r, logu, v, j - 1, epsilon, joint0, call_lf)
#         if sprime:
#             if v == -1:
#                 thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf)
#             else:
#                 _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf)

#             if np.random.uniform() < (float(nprime2) / max(float(nprime + nprime2), 1.0)):
#                 thetaprime, rprime = thetaprime2, rprime2

#             nprime += nprime2
#             sprime = sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus)
#             alphaprime += alphaprime2
#             nalphaprime += nalphaprime2
    
#     return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

def build_tree(theta, r, logu, v, j, epsilon, joint0, call_lf,args, control_group):
    """The main recursion."""
    hnn_model = get_model(args, baseline=True)
    hnn_threshold = control_group.hnn_threshold  # DNN integration error threshold (see https://arxiv.org/abs/2208.06120)
    lf_threshold = control_group.lf_threshold  # Numerical gradient integration error threshold
    if (j == 0):
        t_span1 = [0,v * epsilon]
        kwargs1 = {'t_eval': np.linspace(t_span1[0], t_span1[1], 1), 'rtol': 1e-10}
        y1 = np.concatenate((theta, r), axis=0)
        hnn_ivp1 = integrate_model(hnn_model, t_span1, y1, 1, args, **kwargs1)
        thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
        rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
        joint = functions(hnn_ivp1[:,1])
        call_lf = call_lf or int((np.log(logu) + joint) > 10.)
        monitor = np.log(logu) + joint
        sprime = int((np.log(logu) + joint) <= hnn_threshold)
        
        if call_lf:
            t_span1 = [0,v * epsilon]
            y1 = np.concatenate((theta, r), axis=0)
            hnn_ivp1 = leapfrog(dynamics_fn, t_span1, y1, 1, int(args.input_dim))
            thetaprime = hnn_ivp1[0:int(args.input_dim/2), 1].reshape(int(args.input_dim/2))
            rprime = hnn_ivp1[int(args.input_dim/2):int(args.input_dim), 1].reshape(int(args.input_dim/2))
            joint = functions(hnn_ivp1[:,1])
            sprime = int((np.log(logu) + joint) <= lf_threshold)
        
        nprime = int(logu <= np.exp(-joint))
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        alphaprime = min(1., np.exp(joint0 - joint))
        nalphaprime = 1
    else:
        thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf = build_tree(theta, r, logu, v, j - 1, epsilon, joint0, call_lf,args, control_group)
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, _, _, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j - 1, epsilon, joint0, call_lf,args, control_group)
            else:
                _, _, thetaplus, rplus, thetaprime2, rprime2, nprime2, sprime2, alphaprime2, nalphaprime2, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j - 1, epsilon, joint0, call_lf,args, control_group)
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                rprime = rprime2[:]
            nprime = int(nprime) + int(nprime2)
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alphaprime, nalphaprime, monitor, call_lf

##### Main Hamiltonian sampling loop #####
def dnn_nuts_sampling(args, control_group):
    N = control_group.N  # Number of samples
    burn = control_group.burn  # Number of burn-in samples
    epsilon = control_group.epsilon  # Step size
    N_lf = control_group.N_lf  # Number of cool-down samples when DNN integration errors are high (see https://arxiv.org/abs/2208.06120)
    hnn_threshold = control_group.hnn_threshold  # DNN integration error threshold (see https://arxiv.org/abs/2208.06120)
    lf_threshold = control_group.lf_threshold  # Numerical gradient integration error threshold
    D = int(args.input_dim / 2)
    M = N
    Madapt = 0
    theta0 = np.ones(D)
    samples = np.empty((M + Madapt, D))
    samples[0, :] = theta0
    y0 = np.zeros(args.input_dim)
    
    # Initializing random samples for y0
    for ii in range(int(args.input_dim / 2)):
        y0[ii] = norm(loc=0, scale=1).rvs()
    for ii in range(int(args.input_dim / 2), args.input_dim):
        y0[ii] = norm(loc=0, scale=1).rvs()
    
    HNN_accept = np.ones(M)
    traj_len = np.zeros(M)
    alpha_req = np.zeros(M)
    H_store = np.zeros(M)
    monitor_err = np.zeros(M)
    call_lf = 0
    counter_lf = 0
    is_lf = np.zeros(M)
    total_gradient_evaluations = 0
    # Sampling loop
    for m in tqdm(range(1, M + Madapt)):
        #print(m)
        for ii in range(int(args.input_dim / 2), args.input_dim):
            y0[ii] = norm(loc=0, scale=1).rvs()
        
        joint = functions(y0)
        logu = np.random.uniform(0, np.exp(-joint))
        samples[m, :] = samples[m - 1, :]
    
        # Initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = y0[int(args.input_dim / 2):]
        rplus = y0[int(args.input_dim / 2):]
        
        j = 0
        n = 1
        s = 1
    
        if call_lf:
            counter_lf += 1
        if counter_lf == N_lf:
            call_lf = 0
            counter_lf = 0
    
        while s == 1:
            v = int(2 * (np.random.uniform() < 0.5) - 1)
    
            if v == -1:
                # print("v == 1 \n")
                thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint, call_lf, args, control_group)
            else:
                # print("else \n")
                _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint, call_lf, args, control_group)
            
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                # print(f"sprime : {sprime}")
                samples[m, :] = thetaprime
                r_sto = rprime
    
            n += nprime
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            j += 1
            monitor_err[m] = monitor
        total_gradient_evaluations += (2 ** j) - 1
  
        is_lf[m] = call_lf
        traj_len[m] = j
        alpha_req[m] = alpha
        y0[:int(args.input_dim / 2)] = samples[m, :]
        H_store[m] = functions(np.concatenate((samples[m, :], r_sto)))
    
    ##### Post-processing with TensorFlow Probability #####
    
    hnn_tf = tf.convert_to_tensor(samples[burn:M, :], dtype=tf.float32)
    ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    avg_grad = np.sum(ess_hnn)/total_gradient_evaluations
    # Plot results (optional)
    plt.plot(samples[:, 0], samples[:, 1], 'r+')
    plt.show()
    
    avg_grad = np.sum(ess_hnn)/total_gradient_evaluations
    
    result = {
        "samples": hnn_tf,
        "effective_sample_sizes": ess_hnn,
        "total_gradient_evaluations": total_gradient_evaluations,
        "Avg ESS per gradient": avg_grad
    }
    print("Effective Sample Size (ESS):", ess_hnn)
    print("total_gradient_evaluations:", total_gradient_evaluations)
    print("Avg ESS per gradient", avg_grad)
    print(result)
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，就创建它
    filename = os.path.join(save_dir, f"dnn_nuts_{args.dist_name}_{args.input_dim}.pkl")
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


def get_args_NUTs():
    """
    解析采样参数，并支持从 JSON 文件加载和命令行参数覆盖
    """
    parser = argparse.ArgumentParser(description="User-defined Sampling Parameters")

    # 采样参数
    parser.add_argument("--N", type=int, default=1000, help="Number of samples")
    parser.add_argument("--burn", type=int, default=100, help="Number of burn-in samples")
    parser.add_argument("--epsilon", type=float, default=0.025, help="Step size")
    parser.add_argument("--N_lf", type=int, default=20, help="Number of cool-down samples when DNN integration errors are high")
    parser.add_argument("--hnn_threshold", type=float, default=10.0, help="DNN integration error threshold")
    parser.add_argument("--lf_threshold", type=float, default=1000.0, help="Numerical gradient integration error threshold")
    parser.add_argument("--config", type=str, help="Path to JSON config file", default="config_nuts.json")

    # 解析命令行参数
    args = parser.parse_args()

    # 检查配置文件是否存在（默认是 config_nuts.json）
    config_path = args.config if args.config else "config_nuts.json"
    if os.path.exists(config_path):
        # 加载 JSON 配置文件
        json_config = load_json_config(config_path)
        # 用配置文件中的值覆盖命令行参数
        for key in ["N", "burn", "epsilon", "N_lf", "hnn_threshold", "lf_threshold"]:
            if key in json_config:
                setattr(args, key, json_config[key])

    # 返回解析后的参数
    return args
if __name__ == "__main__":
    args = get_args()
    hnn_model = get_model(args, baseline=True)
    control_group = get_args_NUTs()
    dnn_nuts_sampling(args, control_group)




# ##### Main Hamiltonian sampling loop #####

# D = int(args.input_dim / 2)
# M = N
# Madapt = 0
# theta0 = np.ones(D)
# samples = np.empty((M + Madapt, D))
# samples[0, :] = theta0
# y0 = np.zeros(args.input_dim)

# # Initializing random samples for y0
# for ii in range(int(args.input_dim / 2)):
#     y0[ii] = norm(loc=0, scale=1).rvs()
# for ii in range(int(args.input_dim / 2), args.input_dim):
#     y0[ii] = norm(loc=0, scale=1).rvs()

# HNN_accept = np.ones(M)
# traj_len = np.zeros(M)
# alpha_req = np.zeros(M)
# H_store = np.zeros(M)
# monitor_err = np.zeros(M)
# call_lf = 0
# counter_lf = 0
# is_lf = np.zeros(M)

# # Sampling loop
# for m in tqdm(range(1, M + Madapt)):
#     #print(m)
#     for ii in range(int(args.input_dim / 2), args.input_dim):
#         y0[ii] = norm(loc=0, scale=1).rvs()
    
#     joint = functions(y0)
#     logu = np.random.uniform(0, np.exp(-joint))
#     samples[m, :] = samples[m - 1, :]

#     # Initialize the tree
#     thetaminus = samples[m - 1, :]
#     thetaplus = samples[m - 1, :]
#     rminus = y0[int(args.input_dim / 2):]
#     rplus = y0[int(args.input_dim / 2):]
    
#     j = 0
#     n = 1
#     s = 1

#     if call_lf:
#         counter_lf += 1
#     if counter_lf == N_lf:
#         call_lf = 0
#         counter_lf = 0

#     while s == 1:
#         v = int(2 * (np.random.uniform() < 0.5) - 1)

#         if v == -1:
#             thetaminus, rminus, _, _, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaminus, rminus, logu, v, j, epsilon, joint, call_lf)
#         else:
#             _, _, thetaplus, rplus, thetaprime, rprime, nprime, sprime, alpha, nalpha, monitor, call_lf = build_tree(thetaplus, rplus, logu, v, j, epsilon, joint, call_lf)

#         if sprime == 1 and np.random.uniform() < (float(nprime) / float(n)):
#             samples[m, :] = thetaprime
#             r_sto = rprime

#         n += nprime
#         s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
#         j += 1
#         monitor_err[m] = monitor
        
#     is_lf[m] = call_lf
#     traj_len[m] = j
#     alpha_req[m] = alpha
#     y0[:int(args.input_dim / 2)] = samples[m, :]
#     H_store[m] = functions(np.concatenate((samples[m, :], r_sto)))

# ##### Post-processing with TensorFlow Probability #####

# hnn_tf = tf.convert_to_tensor(samples[burn:M, :], dtype=tf.float32)
# ess_hnn = np.array(tfp.mcmc.effective_sample_size(hnn_tf))

# # Plot results (optional)
# plt.plot(samples[:, 0], samples[:, 1], 'r+')
# plt.show()