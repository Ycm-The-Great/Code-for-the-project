#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:40:26 2024

@author: yanchuanmiao
"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import os
# 导入自定义的模型和工具函数
from nn_models import MLP
from hnn import HNN
from get_args import get_args
from utils import leapfrog,leapfrog_tf
from functions import functions
import pickle 

# args = get_args()

##### 用户定义的采样参数 #####
# chains = 1  # 马尔科夫链数量
# N = 1000  # 每条链的采样数
# L = 10  # 每条链的哈密顿轨迹长度
# burn = 100  # 丢弃的burn-in样本数量
# epsilon = 0.025  # 时间积分步长

# ##### 采样代码 #####

# # 初始状态向量
# y0 = np.zeros(args.input_dim)

# 定义模型加载函数
def get_model(args, baseline):
    output_dim = args.input_dim
    # 使用多层感知机 MLP 作为基础模型
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    # 包装成 Hamiltonian Neural Network (HNN)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type, baseline=baseline)
      
    # 初始化模型变量
    dummy_input = tf.zeros((2, args.input_dim))    # 创建虚拟输入
    _ = model(dummy_input)  # 调用模型，初始化变量
    
    # 加载预训练模型的权重
    path = args.dist_name + ".h5"
    model.load_weights(path)  # TensorFlow 使用 load_weights() 加载权重
    return model

# 定义使用 leapfrog 方法对模型进行积分的函数
def integrate_model(model, t_span, y0, n, args, **kwargs):
    # 定义动力学函数
    def fun(t, np_x):
        # 将 NumPy 数组转换为 TensorFlow 张量，并求解模型的时间导数
        x = tf.convert_to_tensor(np_x, dtype=tf.float32)
        if len(x.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            x = tf.expand_dims(x, axis=0)
        with tf.GradientTape() as tape:
            tape.watch(x)
            dx = model.time_derivative(x)
        return dx.numpy().reshape(-1)  # 将结果转换为 NumPy 数组
    # 使用 leapfrog 积分方法
    return leapfrog(fun, t_span, y0, n, args.input_dim)

def integrate_model_tf(model, t_span, y0, n, args, **kwargs):
    # 定义动力学函数
    def fun(t, np_x):
        # 将 NumPy 数组转换为 TensorFlow 张量，并求解模型的时间导数
        x = tf.convert_to_tensor(np_x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            dx = model.time_derivative(x)
        dx = tf.reshape(dx, [-1])
        # print(tf.shape(dx))
        return dx

    # 使用 leapfrog 积分方法
    return leapfrog_tf(fun, t_span, y0, n, args.input_dim)

tfd = tfp.distributions

class HNN_HMC_Kernel(tfp.mcmc.TransitionKernel):
    def __init__(self, hnn_model, integrate_model, functions, step_size, args, **kwargs):
        """
        自定义 HMC Kernel，使用 HNN 进行动力学积分。
        :param hnn_model: 训练好的 HNN 模型
        :param integrate_model: HNN 用于计算动力学轨迹的函数
        :param functions: 计算哈密顿量的函数
        :param step_size: HMC 采样步长
        :param args: 额外参数
        :param kwargs: 额外关键字参数
        """
        
        self.hnn_model = hnn_model
        self.integrate_model = integrate_model
        self.functions = functions
        self.step_size = step_size
        self.args = args
        self.kwargs = kwargs

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results):
        """
        执行 HMC 采样的一步
        :param current_state: 当前状态 (q, p)
        :param previous_kernel_results: 上一次采样的结果
        :return: (新状态, 是否接受)
        """
        q, p = current_state  # 位置 q, 动量 p
        input_dim = self.args.input_dim
        # 初始化新状态
        y0 = tf.concat([q, p], axis=-1)
        y0 = tf.reshape(y0, [-1])
        # print(y0)
        L = self.kwargs.get("L", 10)  # 设置默认值为 1，防止 L 为空
       
        # 使用 HNN 进行动力学积分
        steps = int(1 / self.step_size) * L
        t_span = [0, L]
        kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}

        hnn_ivp = self.integrate_model(self.hnn_model, t_span, y0, steps - 1, self.args, **self.kwargs)
       
        # 取最终的状态
        yhamil = hnn_ivp[-1, :]
        # print(hnn_ivp)
        # 计算新的哈密顿量
        H_star = self.functions(yhamil)
        H_prev = self.functions(y0)

        # 计算 Metropolis-Hastings 接受率
        alpha = np.minimum(1, np.exp(H_prev - H_star)+ 0.001)
        a = uniform().rvs()
        # 采样是否接受
        accept = alpha > a
        accept = bool(accept)  
        # print(a, alpha,accept)
        # accept = tf.constant(accept, dtype=tf.bool)  # 转为标量布尔tensor

        # print(f"accept.shape: {accept.shape}")  
        # print(f"yhamil.shape: {yhamil.shape}")  
        # print(f"q.shape: {q.shape}") 
        # print(f"p.shape: {p.shape}") 
        # print(tf.shape(hnn_ivp))
        # accept = tf.reshape(accept, [1, 1])  # ✅ 变成 (1, 1)
        # yhamil_selected = tf.reshape(yhamil[:input_dim // 2], (1, 2))  # ✅ 变成 (1, 2)
        # next_q = tf.where(accept, yhamil_selected, q)  # ✅ 确保形状一致

        # 更新 q
        next_q = tf.where(accept, tf.convert_to_tensor(yhamil[:input_dim // 2], dtype=tf.float32), q)

        # 重新采样动量部分 p
        next_p = tf.random.normal(shape=[input_dim // 2], mean=0.0, stddev=1.0, dtype=tf.float32)
        next_p = tf.reshape(next_p, [1,-1])
        # print(f"next_q.shape: {next_q.shape}") 
        # print(f"next_p.shape: {next_p.shape}") 
        # accept = tf.reshape(accept, [1,-1])
        # print(f"is_accepted_tensor.shape: {tf.shape(accept)}")
        # print(f"is_accepted_tensor: {accept}")
        return [next_q, next_p], [accept]

    def bootstrap_results(self, init_state):
        """
        初始化 kernel 结果
        :param init_state: 初始状态
        :return: 是否接受 (布尔值)
        """
        return [tf.constant(False, dtype=tf.bool)]
        # return tf.zeros([], dtype=tf.bool)
   

def hnn_sampling(args, control_group):
    """
    使用 HNN 采样的函数，接收两个参数组。
    参数:
        args_group: 包含模型和输入维度的参数。
        control_group: 包含采样控制参数的字典。
    返回:
        hnn_fin: 采样结果数组 (chains, N, input_dim / 2)。

        ess_hnn: 每条链的有效样本大小 (chains, input_dim / 2)。
    """
    chains = control_group.chains  # 马尔科夫链数量
    N = control_group.N  # 每条链的采样数
    L = control_group.L  # 每条链的哈密顿轨迹长度
    burn = control_group.burn  # 丢弃的burn-in样本数量
    epsilon = control_group.epsilon  # 时间积分步长
    input_dim = args.input_dim  # 变量维度

    # 加载 HNN 模型
    hnn_model = get_model(args, baseline=False)
    y0 = np.zeros(args.input_dim)
    
    # 采样步骤数
    steps = L * int(1 / epsilon)
    t_span = [0, L]  # 时间跨度
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
    
    # 初始化 (q, p)
    initial_q = tf.zeros([chains, input_dim // 2], dtype=tf.float32)  # 位置 q
    initial_p = tf.random.normal([chains, input_dim // 2], dtype=tf.float32)  # 动量 p
    initial_state = [initial_q, initial_p]
    # 定义自定义 HMC Kernel
    hmc_kernel = HNN_HMC_Kernel(hnn_model, integrate_model_tf, functions, epsilon,  args, L = L)
    
    
    
   

    # 运行 MCMC 采样
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=N,
        num_burnin_steps=burn,
        current_state=initial_state,
        kernel=hmc_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results
    )
   
    # 解析采样结果
    # q_samples, p_samples = samples.numpy()
    q_samples = samples[0].numpy()  # 第一个元素是 q 样本
    p_samples = samples[1].numpy()  # 第二个元素是 p 样本
    # 计算有效样本大小 (ESS)
    ess_hnn = np.array([
        tfp.mcmc.effective_sample_size(tf.convert_to_tensor(q_samples[ss, burn:N, :], dtype=tf.float32)).numpy()
        for ss in range(chains)
    ])
    
    total_gradient_evaluations = N * int(1 / epsilon) * L
    avg_ess = np.sum(ess_hnn) / total_gradient_evaluations


    # # 初始化存储采样结果的数组
    # hnn_fin = np.zeros((chains, N, int(args.input_dim / 2)))
    # hnn_accept = np.zeros((chains, N))
    
    # # 对每条链进行采样
    # for ss in np.arange(0, chains, 1):
    #     x_req = np.zeros((N, int(args.input_dim / 2)))  # 存储接受的样本
    #     x_req[0, :] = y0[0:int(args.input_dim / 2)]
    #     accept = np.zeros(N)  # 存储每个样本的接受状态
    
    #     # 初始化 y0 的前半部分为 0，后半部分从正态分布中采样
    #     for ii in np.arange(0, int(args.input_dim / 2), 1):
    #         y0[ii] = 0.0
    #     for ii in np.arange(int(args.input_dim / 2), int(args.input_dim), 1):
    #         y0[ii] = norm(loc=0, scale=1).rvs()
        
    #     # 用于存储哈密顿轨迹的数组
    #     HNN_sto = np.zeros((args.input_dim, steps, N))
        
    #     # 进行 N 次采样
    #     for ii in tqdm(np.arange(0, N, 1)):
    #         # 使用 leapfrog 方法进行哈密顿动力学积分
    #         hnn_ivp = integrate_model(hnn_model, t_span, y0, steps - 1,args, **kwargs)
    #         for sss in range(0, args.input_dim):
    #             HNN_sto[sss, :, ii] = hnn_ivp[sss, :]
            
    #         # 计算新的哈密顿量
    #         yhamil = np.zeros(args.input_dim)
    #         for jj in np.arange(0, args.input_dim, 1):
    #             yhamil[jj] = hnn_ivp[jj, steps - 1]
            
    #         H_star = functions(yhamil)  # 新的哈密顿量
    #         H_prev = functions(y0)  # 之前的哈密顿量
            
    #         # Metropolis-Hastings 接受率
    #         alpha = np.minimum(1, np.exp(H_prev - H_star))
            
    #         # 如果接受了新的样本，更新 y0
    #         if alpha > uniform().rvs():
    #             y0[0:int(args.input_dim / 2)] = hnn_ivp[0:int(args.input_dim / 2), steps - 1]
    #             x_req[ii, :] = hnn_ivp[0:int(args.input_dim / 2), steps - 1]
    #             accept[ii] = 1
    #         else:
    #             x_req[ii, :] = y0[0:int(args.input_dim / 2)]
            
    #         # 每次更新后，重新采样动量部分
    #         for jj in np.arange(int(args.input_dim / 2), args.input_dim, 1):
    #             y0[jj] = norm(loc=0, scale=1).rvs()
            
    #        # print(f"Sample: {ii} Chain: {ss}")
        
    #     # 存储当前链的接受率和采样结果
    #     hnn_accept[ss, :] = accept
    #     hnn_fin[ss, :, :] = x_req
    
    # 计算有效样本大小 (ESS)
    # ess_hnn = np.zeros((chains, int(args.input_dim / 2)))
    # for ss in np.arange(0, chains, 1):
    #     # 转换为 TensorFlow 张量
    #     hnn_tf = tf.convert_to_tensor(hnn_fin[ss, burn:N, :], dtype=tf.float32)
    #     # 使用 TensorFlow Probability 计算 ESS
    #     ess_hnn[ss, :] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    
    total_gradient_evaluations = steps * N

    avg_ess = np.sum(ess_hnn)/total_gradient_evaluations
    print("Effective Sample Size (ESS):", ess_hnn)
    print("total_gradient_evaluations:", total_gradient_evaluations)
    print("Avg ESS per gradient", avg_ess)
    
    result = {
        "samples": q_samples,
        "effective_sample_sizes": ess_hnn,
        "total_gradient_evaluations": total_gradient_evaluations,
        "Avg ESS per gradient": avg_ess
    }
   #  result = {
   #     "samples": hnn_fin,
   #     "effective_sample_sizes": ess_hnn,
   #     "total_gradient_evaluations": total_gradient_evaluations,
   #     "Avg ESS per gradient": avg_ess
   # }
    print(result)
    # 创建保存路径
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，就创建它
    filename = os.path.join(save_dir, f"hnn_hmc_{args.dist_name}_{args.input_dim}.pkl")
    # 保存为 Pickle 文件
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    
    print(f"Results saved to {filename}")
    # for ss in np.arange(0, chains, 1):
    #     # 转换为 TensorFlow 张量
    #     hnn_tf = tf.convert_to_tensor(hnn_fin[ss, burn:N, :], dtype=tf.float32)
    #     # 使用 TensorFlow Probability 计算 ESS
    #     ess_hnn[ss, :] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
    # result = {
    #     "samples": hnn_fin,
    #     "effective_sample_sizes": ess_hnn,
    # }
    # print("Effective Sample Size (ESS):", ess_hnn)
    return result

def load_json_config(json_file):
    """
    从 JSON 文件加载参数并返回字典
    """
    with open(json_file, "r") as f:
        return json.load(f)

def get_control_group_hmc():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Control Group Parameters", add_help=False)
    parser.add_argument("--chains", type=int, default=1, help="Number of Markov chains")
    parser.add_argument("--epsilon", type=float, default=0.025, help="Time integration step size")
    parser.add_argument("--N", type=int, default=1000, help="Number of samples per chain")
    parser.add_argument("--L", type=int, default=20, help="Length of Hamiltonian trajectories")
    parser.add_argument("--burn", type=int, default=100, help="Number of burn-in samples to discard")
    parser.add_argument("--config", type=str, help="Path to JSON config file", default="config_hmc.json")

    # 解析命令行参数
    args = parser.parse_args()

    # 检查配置文件是否存在（默认值是 config_hmc.json）
    config_path = args.config if args.config else "config_hmc.json"
    if os.path.exists(config_path):
        # 加载 JSON 配置文件
        json_config = load_json_config(config_path)
        # 用配置文件中的值覆盖命令行参数
        for key in ["chains", "epsilon", "N", "L", "burn"]:
            if key in json_config:
                setattr(args, key, json_config[key])

    return args

if __name__ == "__main__":
    args = get_args()
    control_group = get_control_group_hmc()
    hnn_sampling(args, control_group)
    



