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
import argparse
import pickle
from collections import namedtuple


# 导入自定义的模型和工具函数
from nn_models import MLP
from hnn import HNN
from get_args import get_args
from utils import leapfrog,leapfrog_tf
from functions import functions

# # 获取命令行参数
# args = get_args()

# ##### 用户定义的采样参数 #####
# chains = 1  # 马尔科夫链数量
# N = 10000  # 每条链的采样数量
# epsilon = 0.025  # 步长
# burn = 1000  # burn-in 样本数量

# ##### 采样代码 #####

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
            dy = dx.numpy().reshape(-1)
            # print(dy.shape)
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

HNNKernelResults = namedtuple("HNNKernelResults", [
    "accept",          # [chains] 是否接受
    "log_accept_ratio",# [chains] log(α)
    "grad_evals"       # 标量：累积梯度调用次数
])


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
        return True

    def one_step(self, current_state, previous_kernel_results, seed = None):
        """
        执行 HMC 采样的一步
        :param current_state: 当前状态 (q, p)
        :param previous_kernel_results: 上一次采样的结果
        :return: (新状态, 是否接受)
        """
        q, p = current_state  # 位置 q, 动量 p
        input_dim = self.args.input_dim
        # print(current_state)
        # 初始化新状态
        y0 = tf.concat([q, p], axis=-1)
        # y0 = tf.reshape(y0, [-1])
        
        # print(y0)
        L = self.kwargs.get("L", 10)  # 设置默认值为 1，防止 L 为空
       
        # 使用 HNN 进行动力学积分
        steps = 2
        t_span = [0, L]
        kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}

        hnn_ivp = self.integrate_model(self.hnn_model, t_span, y0, steps - 1, self.args, **self.kwargs)
        chains = q.shape[0]
        # print(chains)
        # 取最终的状态
        yhamil = hnn_ivp[-1, :]
        # print(hnn_ivp)
        yhamil1 = tf.reshape(yhamil, [args.input_dim, -1])
        y0 = tf.reshape(y0, [args.input_dim, -1])
        # 计算新的哈密顿量
        H_star = self.functions(yhamil1)
        H_prev = self.functions(y0)
        
        log_accept_ratio = tf.clip_by_value(H_prev - H_star, -50.0, 50.0)
        alpha = tf.math.exp(log_accept_ratio)
       
        
        # 计算 Metropolis-Hastings 接受率
        # alpha = tf.minimum(1, tf.exp(H_prev - H_star)+0.01)
        alpha = tf.minimum(1, alpha+0.01)
        # print(alpha)
        tf_seed = tfp.random.sanitize_seed(seed)
        rand_val = tf.random.stateless_uniform(shape=[chains], seed=tf_seed, dtype=tf.float32)
        accept = alpha > rand_val
        grad_evals_this_step = steps 
        total_grad_evals = previous_kernel_results.grad_evals + grad_evals_this_step
        # 更新 q
        accept_transposed = tf.transpose(accept)
        next_q = tf.where(accept_transposed, tf.convert_to_tensor(yhamil[:,:input_dim // 2], dtype=tf.float32), q)
        # 重新采样动量部分 p
        next_p = tf.random.normal(shape=[chains,input_dim // 2], mean=0.0, stddev=1.0, dtype=tf.float32)
        
        log_accept_ratio = tf.squeeze(log_accept_ratio, axis=0)  # 明确去掉第 0 维
        accept = tf.squeeze(accept, axis=0)  # 明确去掉第 0 维
        # next_p = tf.reshape(next_p, [1,-1])
        # print(f"next_q.shape: {next_q.shape}") 
        # print(f"next_p.shape: {next_p.shape}") 
        return [next_q, next_p], HNNKernelResults(
            accept=accept,
            log_accept_ratio=log_accept_ratio,
            grad_evals=total_grad_evals
        )

    def bootstrap_results(self, init_state):
        """
        初始化 kernel 结果
        :param init_state: 初始状态
        :return: 是否接受 (布尔值)
        """
        chains = init_state[0].shape[0]
        return HNNKernelResults(
            accept=tf.zeros([chains], dtype=tf.bool),
            log_accept_ratio=tf.zeros([chains], dtype=tf.float32),
            grad_evals=tf.zeros([], dtype=tf.int32)
        )
        # return [tf.zeros([1,chains], dtype=tf.bool)]
        # return [tf.constant(False, dtype=tf.bool)]
        # return tf.zeros([], dtype=tf.bool)
        

def dnn_lmc_sampling(args, control_group):
    # 初始化时间跨度和步数
    chains = control_group.chains  # 马尔科夫链数量
    N = control_group.N  # 每条链的采样数
    L = control_group.L  # 每条链的哈密顿轨迹长度
    burn = control_group.burn  # 丢弃的burn-in样本数量
    epsilon = control_group.epsilon  # 时间积分步长
    input_dim = args.input_dim  # 变量维度

    # 初始化时间跨度和步数
    t_span = [0, epsilon]
    steps = 2  # Langevin 动力学每次只进行一个小步
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
    
    # 加载 HNN 模型
    hnn_model = get_model(args, baseline=True)

    # 初始化 (q, p)
    initial_q = tf.zeros([chains, input_dim // 2], dtype=tf.float32)  # 位置 q
    initial_p = tf.random.normal([chains, input_dim // 2], dtype=tf.float32)  # 动量 p
    initial_state = [initial_q, initial_p]
    # 定义自定义 HMC Kernel
    hmc_kernel = HNN_HMC_Kernel(hnn_model, integrate_model_tf, functions, epsilon,  args, L = L)
    
    
    seed_tf = tf.constant(args.seed_tf, dtype=tf.int32)
    seed = tfp.random.sanitize_seed(seed_tf)
    

    # 运行 MCMC 采样
    with tf.device('/cpu:0'):
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=N,
            num_burnin_steps=burn,
            current_state=initial_state,
            kernel=hmc_kernel,
            trace_fn=lambda current_state, kernel_results: kernel_results,
            seed = seed
        )
  
    
    # 解析采样结果
    # q_samples, p_samples = samples.numpy()
    q_samples = samples[0].numpy()  # 第一个元素是 q 样本
    p_samples = samples[1].numpy()  # 第二个元素是 p 样本
   
    ess_hnn_tf = tfp.mcmc.effective_sample_size(q_samples)
    accept_ratio_per_chain = tf.reduce_mean(tf.cast(kernel_results.accept, tf.float32), axis=0)

    # 如需要 NumPy 格式
    ess_hnn = ess_hnn_tf.numpy()
    
    total_gradient_evaluations = N * int(1 / epsilon) * L
    total_gradient_evaluations = kernel_results.grad_evals[-1] * N/(N+burn)
    avg_ess = np.nanmean(ess_hnn) / total_gradient_evaluations

    result = {
        "samples": q_samples,
        "effective_sample_sizes": ess_hnn,
        "total_gradient_evaluations": total_gradient_evaluations,
        "Avg ESS per gradient": avg_ess
    }

    # 创建保存路径
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"dnn_lmc_{args.dist_name}_{args.input_dim}.pkl")
    print(result)
    # 保存为 Pickle 文件
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    
    print(f"Results saved to {filename}")
    return result
    
 
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
    dnn_lmc_sampling(args, control_group)

