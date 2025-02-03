#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:04:13 2025

@author: yanchuanmiao
"""
import numpy as np
from scipy.stats import norm, uniform
import tensorflow as tf
from utils import to_pickle, from_pickle, L2_loss
import tensorflow_probability as tfp

from tqdm import tqdm
from get_data import * 
from Args import Args
from PHNN import PHNN
import os

# 定义模型加载函数
def get_model(args):
    
    # 使用多层感知机 MLP 作为基础模型
    dim_theta = args.input_theta_dim
    dim_u = args.aux_dim
    input_dim = 2 * dim_theta + 2 * dim_u
    # 包装成 Hamiltonian Neural Network (HNN)
    model = PHNN(dim_theta=dim_theta, dim_u=dim_u, hidden_dim=args.hidden_dim)
    # 初始化模型变量
    dummy_input = tf.zeros((2, input_dim))    # 创建虚拟输入
    theta = dummy_input[:, :dim_theta]  # 第一部分: theta
    rho = dummy_input[:, dim_theta:2 * dim_theta] # 第二部分: phi
    u = dummy_input[:, 2 * dim_theta:2 * dim_theta + dim_u] # 第三部分: u
    p = dummy_input[:, -dim_u:]  # 第四部分: p
    _ = model(theta, rho,u,p )  # 调用模型，初始化变量
    
    # 加载预训练模型的权重
    path = f"{args.model}.h5"
    model.load_weights(path)  # TensorFlow 使用 load_weights() 加载权重
    return model

def leapfrog(dynamics_fn, t_span, y0, n_steps, input_dim, h, model, aux_dim):
    """
    传统 Leapfrog 积分器（完整更新 theta, rho, u, p）
    """
    # 解包初始状态
    theta = tf.convert_to_tensor(y0[:input_dim], dtype=tf.float32)
    rho = tf.convert_to_tensor(y0[input_dim:2*input_dim], dtype=tf.float32)
    u = tf.convert_to_tensor(y0[2*input_dim:2*input_dim+aux_dim], dtype=tf.float32)
    p = tf.convert_to_tensor(y0[2*input_dim+aux_dim:], dtype=tf.float32)
    
    trajectory = [tf.concat([theta, rho, u, p], axis=0).numpy()]

    for _ in range(n_steps):
        # 1. 动量半步更新：rho 和 p
        dtheta_dt, drho_dt, du_dt, dp_dt = dynamics_fn(theta, rho, u, p)
        rho_half = rho + 0.5 * h * drho_dt
        p_half = p + 0.5 * h * dp_dt
        
        # 2. 位置全步更新：theta 和 u
        theta = theta + h * dtheta_dt
        u = u + h * du_dt
        
        # 3. 动量再半步更新：rho 和 p（基于新位置）
        dtheta_dt_new, drho_dt_new, du_dt_new, dp_dt_new = dynamics_fn(theta, rho_half, u, p_half)
        rho = rho_half + 0.5 * h * drho_dt_new
        p = p_half + 0.5 * h * dp_dt_new
        
        # 记录轨迹
        trajectory.append(tf.concat([theta, rho, u, p], axis=0).numpy())
    return np.array(trajectory).T  # 形状 [input_dim, n_steps+1]

def integrate_model(model1, t_span, y0, n,input_theta_dim, h, model, aux_dim ):
    # 定义动力学函数
    def fun(theta, rho, u, p):
        # if len(x.shape) == 1:  # Input is 1D, e.g., (input_dim,)
        #     x = tf.expand_dims(x, axis=0)
        # 将 NumPy 数组转换为 TensorFlow 张量，并求解模型的时间导数
        tf_theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        tf_rho = tf.convert_to_tensor(rho, dtype=tf.float32)
        tf_u = tf.convert_to_tensor(u, dtype=tf.float32)
        tf_p = tf.convert_to_tensor(p, dtype=tf.float32)
        if len(tf_theta.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            tf_theta = tf.expand_dims(tf_theta, axis=0)
        if len(tf_rho.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            tf_rho = tf.expand_dims(tf_rho, axis=0)
        if len(tf_u.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            tf_u = tf.expand_dims(tf_u, axis=0)
        if len(tf_p.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            tf_p = tf.expand_dims(tf_p, axis=0)
        dtheta_dt, drho_dt, du_dt, dp_dt = model1.hamiltonian_dynamics(tf_theta, tf_rho, tf_u, tf_p)
        return dtheta_dt.numpy().reshape(-1), drho_dt.numpy().reshape(-1), du_dt.numpy().reshape(-1), dp_dt.numpy().reshape(-1)  # 将结果转换为 NumPy 数组
    # 使用 leapfrog 积分方法
    return leapfrog(fun, t_span, y0, n, input_theta_dim, h, model, aux_dim)



def phnn_sampling(args):
    ##### 采样代码 #####
    # args = Args()
    phnn_model = get_model(args)
    
    
    ##### 用户定义的采样参数 #####
    chains = args.chains  # 马尔科夫链数量
    N = args.N  # 每条链的采样数
    L = args.L  # 每条链的哈密顿轨迹长度
    burn = args.burn  # 丢弃的burn-in样本数量
    epsilon = args.epsilon  # 时间积分步长
    
    
    y = from_pickle(args.y_path)
    input_dim = 2 * args.input_theta_dim + 2 * args.aux_dim  
    t_eval = np.linspace(args.t_span[0], args.t_span[1], int(args.timescale * (args.t_span[1] - args.t_span[0])))
    h = (args.t_span[1] - args.t_span[0]) / len(t_eval)  # 计算步长
    y0 = np.zeros(input_dim,dtype=np.float32)
    steps = len(t_eval)
    # 初始化存储采样结果的数组
    hnn_fin = np.zeros((chains, N, args.input_theta_dim))
    hnn_accept = np.zeros((chains, N))
    
    # 对每条链进行采样
    for ss in np.arange(0, chains, 1):
        x_req = np.zeros((N, args.input_theta_dim))  # 存储接受的样本
        x_req[0, :] = y0[0: args.input_theta_dim]
        accept = np.zeros(N)  # 存储每个样本的接受状态
        dim_theta = args.input_theta_dim
        dim_u = args.aux_dim
        # # 初始化 y0 的前半部分为 0，后半部分从正态分布中采样
        # for ii in np.arange(0, int(args.input_dim / 2), 1):
        #     y0[ii] = 0.0
        # for ii in np.arange(int(args.input_dim / 2), int(args.input_dim), 1):
        #     y0[ii] = norm(loc=0, scale=1).rvs()
        y0[args.input_theta_dim:2*args.input_theta_dim] = np.random.randn(args.input_theta_dim)  # rho
        y0[2 * args.input_theta_dim : ] = np.random.randn(2*args.aux_dim)
        # 用于存储哈密顿轨迹的数组
        HNN_sto = np.zeros((input_dim, len(t_eval), N))
        
        # 进行 N 次采样
        for ii in tqdm(np.arange(0, N, 1)):
            # 使用 leapfrog 方法进行哈密顿动力学积分
            hnn_ivp = integrate_model(phnn_model, t_span = args.t_span, y0 = y0, n = len(t_eval) - 1, input_theta_dim = args.input_theta_dim, h = h, model = args.model, aux_dim = args.aux_dim)
            # for sss in range(0, input_dim):
            #     HNN_sto[sss, :, ii] = hnn_ivp[sss, :]
            HNN_sto[:, :, ii] = hnn_ivp[:, :]
    
            # 计算新的哈密顿量
            yhamil = np.zeros(input_dim, dtype=np.float32)
    
            
            # for jj in np.arange(0, input_dim, 1):
            #     yhamil[jj] = hnn_ivp[jj, len(t_eval) - 1]
            yhamil[:] = hnn_ivp[:, len(t_eval) - 1]
            theta_star = yhamil[ :dim_theta]  # 第一部分: theta
            rho_star = yhamil[ dim_theta:2 * dim_theta] # 第二部分: phi
            u_star = yhamil[ 2 * dim_theta:2 * dim_theta + dim_u] # 第三部分: u
            p_star = yhamil[ -dim_u:]  # 第四部分: p
            H_star = compute_hamiltonian(theta_star, rho_star, u_star, p_star, y, args.model) # 新的哈密顿量
            
            theta_0 = y0[ :dim_theta]  # 第一部分: theta
            rho_0 = y0[ dim_theta:2 * dim_theta] # 第二部分: phi
            u_0 = y0[ 2 * dim_theta:2 * dim_theta + dim_u] # 第三部分: u
            p_0 = y0[ -dim_u:]  # 第四部分: p
            H_prev = compute_hamiltonian(theta_0, rho_0, u_0, p_0, y, args.model)  # 之前的哈密顿量
            
            # Metropolis-Hastings 接受率
            alpha = np.minimum(1, np.exp(H_prev - H_star))
            
            # 如果接受了新的样本，更新 y0
            if alpha > uniform().rvs():
                y0[0:args.input_theta_dim] = hnn_ivp[0:args.input_theta_dim, steps - 1]
                x_req[ii, :] = hnn_ivp[0:args.input_theta_dim, steps - 1]
                accept[ii] = 1
            else:
                x_req[ii, :] = y0[0:args.input_theta_dim]
            
            # 每次更新后，重新采样动量部分
            # for jj in np.arange(args.input_theta_dim, input_dim, 1):
            #     y0[jj] = norm(loc=0, scale=1).rvs()
            y0[args.input_theta_dim:input_dim] = norm(loc=0, scale=1).rvs(size=(input_dim - args.input_theta_dim))
           # print(f"Sample: {ii} Chain: {ss}")
        
        # 存储当前链的接受率和采样结果
        hnn_accept[ss, :] = accept
        hnn_fin[ss, :, :] = x_req
        
    
if __name__ == "__main__":
    args = Args().parse_args()
    
    phnn_sampling(args)
    
    
# # 初始化状态
# y0 = np.zeros(input_dim)
# # theta 初始化为零
# y0[args.input_theta_dim:2*args.input_theta_dim] = np.random.randn(args.input_theta_dim)  # rho
# y0[2 * args.input_theta_dim : ] = np.random.randn(2*args.aux_dim)  # u 和 p

# leapfrog_traditional(fun, t_span = args.t_span, y0 = y0, n_steps = len(t_eval) - 1, input_dim = args.input_theta_dim, h = h, model = args.model , y_data =None, aux_dim = args.aux_dim)

# # 计算有效样本大小 (ESS)
# ess_hnn = np.zeros((chains, int(args.input_dim / 2)))
# for ss in np.arange(0, chains, 1):
#     # 转换为 TensorFlow 张量
#     hnn_tf = tf.convert_to_tensor(hnn_fin[ss, burn:N, :], dtype=tf.float32)
#     # 使用 TensorFlow Probability 计算 ESS
#     ess_hnn[ss, :] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))

# print("Effective Sample Size (ESS):", ess_hnn)

