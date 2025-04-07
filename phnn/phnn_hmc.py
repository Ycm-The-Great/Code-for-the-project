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
from collections import namedtuple
from tqdm import tqdm
from get_data import * 
from Args import Args
from PHNN import PHNN
import os
import pickle

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

def integrate_model_tf(model1, t_span, y0, n, input_theta_dim, h, model, aux_dim):
    tf.debugging.assert_rank(y0, 1, message="y0 must be a 1D tensor (single chain)")

    # 定义动力学函数
    def fun(theta, rho, u, p):
        tf_theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        tf_rho = tf.convert_to_tensor(rho, dtype=tf.float32)
        tf_u = tf.convert_to_tensor(u, dtype=tf.float32)
        tf_p = tf.convert_to_tensor(p, dtype=tf.float32)
        if len(tf_theta.shape) == 1:
            tf_theta = tf.expand_dims(tf_theta, axis=0)
        if len(tf_rho.shape) == 1:
            tf_rho = tf.expand_dims(tf_rho, axis=0)
        if len(tf_u.shape) == 1:
            tf_u = tf.expand_dims(tf_u, axis=0)
        if len(tf_p.shape) == 1:
            tf_p = tf.expand_dims(tf_p, axis=0)
        dtheta_dt, drho_dt, du_dt, dp_dt = model1.hamiltonian_dynamics(tf_theta, tf_rho, tf_u, tf_p)
        return (
            dtheta_dt.numpy().reshape(-1),
            drho_dt.numpy().reshape(-1),
            du_dt.numpy().reshape(-1),
            dp_dt.numpy().reshape(-1)
        )
    # 使用 leapfrog 积分方法，并返回 TensorFlow 数据类型
    return leapfrog_tf(fun, t_span, y0, n, input_theta_dim, h, model, aux_dim)

def leapfrog_tf(dynamics_fn, t_span, y0, n_steps, input_dim, h, model, aux_dim):
    """
    传统 Leapfrog 积分器（完整更新 theta, rho, u, p）
    """
    theta = tf.convert_to_tensor(y0[:input_dim], dtype=tf.float32)
    rho = tf.convert_to_tensor(y0[input_dim:2*input_dim], dtype=tf.float32)
    u = tf.convert_to_tensor(y0[2*input_dim:2*input_dim+aux_dim], dtype=tf.float32)
    p = tf.convert_to_tensor(y0[2*input_dim+aux_dim:], dtype=tf.float32)

    trajectory = [tf.concat([theta, rho, u, p], axis=0).numpy()]
    for _ in range(n_steps):
        dtheta_dt, drho_dt, du_dt, dp_dt = dynamics_fn(theta, rho, u, p)
        rho_half = rho + 0.5 * h * drho_dt
        p_half = p + 0.5 * h * dp_dt

        theta = theta + h * dtheta_dt
        u = u + h * du_dt

        dtheta_dt_new, drho_dt_new, du_dt_new, dp_dt_new = dynamics_fn(theta, rho_half, u, p_half)
        rho = rho_half + 0.5 * h * drho_dt_new
        p = p_half + 0.5 * h * dp_dt_new

        trajectory.append(tf.concat([theta, rho, u, p], axis=0).numpy())

    # 返回 TensorFlow 张量，shape 为 [input_dim, n_steps+1]
    return tf.convert_to_tensor(np.array(trajectory).T, dtype=tf.float32)


class PHNN_HMC_Kernel(tfp.mcmc.TransitionKernel):
    def __init__(self, phnn_model, integrate_model, functions, y_data, args, **kwargs):
        """
        自定义 HMC Kernel，使用 HNN 进行动力学积分。
        :param hnn_model: 训练好的 HNN 模型
        :param integrate_model: HNN 用于计算动力学轨迹的函数
        :param functions: 计算哈密顿量的函数
        :param step_size: HMC 采样步长
        :param args: 额外参数
        :param kwargs: 额外关键字参数
        """
        
        self.phnn_model = phnn_model
        self.integrate_model = integrate_model
        self.functions = functions
        self.y_data = y_data
        self.args = args
        self.kwargs = kwargs
        

    @property
    def is_calibrated(self):
        return True
    def one_step(self, current_state, previous_kernel_results, seed = None):
       """Perform one iteration of PHNN-based HMC using Leapfrog integration."""
       # current_state: [batch_size, theta_dim]
       args = self.args
       input_theta_dim = args.input_theta_dim
       aux_dim = args.aux_dim
       input_dim = 2*input_theta_dim + 2*aux_dim
      
       t_span = args.t_span
       model = args.model
       y_data = self.y_data# 观测数据
       theta_0 = current_state
       num_chains = tf.shape(current_state)[0]
       t_eval = np.linspace(args.t_span[0], args.t_span[1], int(args.timescale * (args.t_span[1] - args.t_span[0])))
       h = (args.t_span[1] - args.t_span[0]) / len(t_eval)  # 计算步长
       steps = len(t_eval)
       phnn_model = self.phnn_model

       # === 初始化 rho, u, p ===
       rho_0 = tf.random.normal(shape=(num_chains, input_theta_dim))
       u_0 = tf.random.normal(shape=(num_chains, aux_dim))
       p_0 = tf.random.normal(shape=(num_chains, aux_dim))
       # 拼接为完整 y0 向量: [theta, rho, u, p]
       y0 = tf.concat([theta_0, rho_0, u_0, p_0], axis=1)
       def integrate_multiple_chains(phnn_model, t_span, y0_all, n, input_theta_dim, h, model_name, aux_dim):
            def single_chain_fn(y0_single):
                return integrate_model_tf(
                        phnn_model,
                        t_span=t_span,
                        y0=y0_single,  # [dim]
                        n=n,
                        input_theta_dim=input_theta_dim,
                        h=h,
                        model=model_name,
                        aux_dim=aux_dim
                    ) # 得到 [dim, steps+1]
        
            results = tf.map_fn(
                fn=single_chain_fn,
                elems=y0_all,
                fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32)
            )  # [num_chains, dim, steps+1]
        
            return tf.transpose(results, perm=[1, 2, 0])  # [dim,steps+1,num_chains]
       # 积分仿真（调用你的 integrate_model 函数）
       # y_new = integrate_model(phnn_model, t_span = args.t_span, y0 = y0, n = len(t_eval) - 1, input_theta_dim = args.input_theta_dim, h = h, model = args.model, aux_dim = args.aux_dim) # 输出形状: [dim, steps+1]
       y_new = integrate_multiple_chains(
            phnn_model,
            t_span=args.t_span,
            y0_all=y0,
            n=len(t_eval) - 1,
            input_theta_dim=args.input_theta_dim,
            h=h,
            model_name=args.model,
            aux_dim=args.aux_dim
        )
       # 取最后一步的状态
       y_last = y_new[:, -1,:]
       y_last = tf.reshape(y_last, [num_chains, -1])
       theta_star = y_last[ :,:input_theta_dim]
       rho_star = y_last[ :,input_theta_dim:2*input_theta_dim]
       u_star = y_last[: ,2*input_theta_dim : 2*input_theta_dim + aux_dim]
       p_star = y_last[:, -aux_dim:]
       # 计算哈密顿量
       # H_prev = self.functions(theta_0, rho_0, u_0, p_0, y_data, model)
       # H_star = self.functions(theta_star, rho_star, u_star, p_star, y_data, model)
       H_prev = tf.vectorized_map(
            lambda x: self.functions(x[0], x[1], x[2], x[3], y_data, model),
            elems=(theta_0, rho_0, u_0, p_0)
        )
        
       H_star = tf.vectorized_map(
            lambda x: self.functions(x[0], x[1], x[2], x[3], y_data, model),
            elems=(theta_star, rho_star, u_star, p_star)
        )
        
       # MH 接受率
       # === MH 接受率（数值稳定）===
       log_accept_ratio = tf.clip_by_value(H_prev - H_star, -50.0, 50.0)  # 限制范围
       alpha = tf.exp(log_accept_ratio)
       alpha = tf.minimum(1.0, alpha + 0.01)
       tf_seed = tfp.random.sanitize_seed(seed)
       rand_val = tf.random.stateless_uniform(shape=[num_chains], seed=tf_seed, dtype=tf.float32)
       
       accept = rand_val < alpha
        
       # new_state = tf.cond(accept, lambda: theta_star, lambda: theta_0)
       # MH 接受率
       accept_mask = tf.cast(tf.expand_dims(accept, axis=1), tf.float32)
       new_state = accept_mask * theta_star + (1 - accept_mask) * current_state

       # === 返回 kernel 结果 ===
       kernel_results = {
            "accept_prob": alpha,
            "accepted": accept,
            "theta_star": theta_star,
            "rho_star": rho_star,
            "u_star": u_star,
            "p_star": p_star,
            "H_star": H_star,
            "H_prev": H_prev
        }

       # print("step finishes")
       return new_state, kernel_results
    def bootstrap_results(self, init_state):
        """
        初始化 kernel_results，用于第一次调用 one_step。
        此版本将 rho_0, u_0, p_0 和 H_prev 全部初始化为 0（用于调试或 warm start）。
        """
        args = self.args
        input_theta_dim = args.input_theta_dim
        aux_dim = args.aux_dim
    
        num_chains = tf.shape(init_state)[0]
    
        # === 全零初始化 ===
        rho_0 = tf.zeros(shape=(num_chains, input_theta_dim), dtype=tf.float32)
        u_0   = tf.zeros(shape=(num_chains, aux_dim), dtype=tf.float32)
        p_0   = tf.zeros(shape=(num_chains, aux_dim), dtype=tf.float32)
        H_prev = tf.zeros(shape=(num_chains,), dtype=tf.float32)
    
        # === 初始化结果 ===
        kernel_results = {
            "accept_prob": tf.zeros_like(H_prev),
            "accepted": tf.zeros_like(H_prev, dtype=tf.bool),
            "theta_star": init_state,
            "rho_star": rho_0,
            "u_star": u_0,
            "p_star":p_0,
            "H_star": H_prev,
            "H_prev": H_prev
        }
    
        return kernel_results
PHNNKernelResults = namedtuple(
    "PHNNKernelResults",
    [
        "accept_prob",  # 接受概率 alpha
        "accepted",     # 是否接受（bool）
        "theta_star",   # 候选参数 theta*
        "rho_star",     # 候选动量 rho*
        "u_star",       # 候选辅助变量 u*
        "p_star",       # 候选 p（辅助动量变量）
        "H_star",       # 候选哈密顿量
        "H_prev"        # 当前哈密顿量
    ]
)
    # def one_step(self, current_state, previous_kernel_results, seed = None):
    #     """
    #     执行 HMC 采样的一步
    #     :param current_state: 当前状态 (q, p)
    #     :param previous_kernel_results: 上一次采样的结果
    #     :return: (新状态, 是否接受)
    #     """
    #     q, p = current_state  # 位置 q, 动量 p
    #     input_dim = self.args.input_dim
    #     # print(current_state)
    #     # 初始化新状态
    #     y0 = tf.concat([q, p], axis=-1)
    #     # y0 = tf.reshape(y0, [-1])
        
    #     # print(y0)
    #     L = self.kwargs.get("L", 10)  # 设置默认值为 1，防止 L 为空
       
    #     # 使用 HNN 进行动力学积分
    #     steps = 2
    #     t_span = [0, L]
    #     kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}

    #     hnn_ivp = self.integrate_model(self.hnn_model, t_span, y0, steps - 1, self.args, **self.kwargs)
    #     chains = q.shape[0]
    #     # print(chains)
    #     # 取最终的状态
    #     yhamil = hnn_ivp[-1, :]
    #     # print(hnn_ivp)
    #     yhamil1 = tf.reshape(yhamil, [args.input_dim, -1])
    #     y0 = tf.reshape(y0, [args.input_dim, -1])
    #     # 计算新的哈密顿量
    #     H_star = self.functions(yhamil1)
    #     H_prev = self.functions(y0)
        
    #     log_accept_ratio = tf.clip_by_value(H_prev - H_star, -50.0, 50.0)
    #     alpha = tf.math.exp(log_accept_ratio)
       
        
    #     # 计算 Metropolis-Hastings 接受率
    #     # alpha = tf.minimum(1, tf.exp(H_prev - H_star)+0.01)
    #     alpha = tf.minimum(1, alpha+0.01)
    #     # print(alpha)
    #     tf_seed = tfp.random.sanitize_seed(seed)
    #     rand_val = tf.random.stateless_uniform(shape=[chains], seed=tf_seed, dtype=tf.float32)
    #     accept = alpha > rand_val
    #     grad_evals_this_step = steps 
    #     total_grad_evals = previous_kernel_results.grad_evals + grad_evals_this_step
    #     # 更新 q
    #     accept_transposed = tf.transpose(accept)
    #     next_q = tf.where(accept_transposed, tf.convert_to_tensor(yhamil[:,:input_dim // 2], dtype=tf.float32), q)
    #     # 重新采样动量部分 p
    #     next_p = tf.random.normal(shape=[chains,input_dim // 2], mean=0.0, stddev=1.0, dtype=tf.float32)
        
    #     log_accept_ratio = tf.squeeze(log_accept_ratio, axis=0)  # 明确去掉第 0 维
    #     accept = tf.squeeze(accept, axis=0)  # 明确去掉第 0 维
    #     # next_p = tf.reshape(next_p, [1,-1])
    #     # print(f"next_q.shape: {next_q.shape}") 
    #     # print(f"next_p.shape: {next_p.shape}") 
    #     return [next_q, next_p], HNNKernelResults(
    #         accept=accept,
    #         log_accept_ratio=log_accept_ratio,
    #         grad_evals=total_grad_evals
    #     )


       
        


def phnn_sampling(args):
    ##### 采样代码 #####
    # args = Args()
    phnn_model = get_model(args)
    
    
    ##### 用户定义的采样参数 #####
    chains = args.chains  # 马尔科夫链数量
    N = args.N  # 每条链的采样数
    num_samples = args.num_samples
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
    
    hnn_fin = tf.zeros((chains, args.input_theta_dim), dtype=tf.float32)
    
    
    # 实例化自定义 Kernel
    hnn_kernel = PHNN_HMC_Kernel(
        phnn_model=phnn_model,
        integrate_model=integrate_model,
        functions= compute_hamiltonian,
        y_data = y,
        args=args
    )
    
    # 运行采样器
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_samples,            # 抽样步数
        current_state=hnn_fin,           # 初始状态
        kernel=hnn_kernel,                  # 自定义 HNN-HMC 核
        num_burnin_steps=burn,        # 热身步数
        trace_fn=lambda _, pkr: pkr         # 追踪 kernel_results（比如接受率）
    )
    
    # hnn_fin = np.zeros((chains, N, args.input_theta_dim))
    # hnn_accept = np.zeros((chains, N))
    # # 对每条链进行采样
    # for ss in np.arange(0, chains, 1):
    #     x_req = np.zeros((N, args.input_theta_dim))  # 存储接受的样本
    #     x_req[0, :] = y0[0: args.input_theta_dim]
    #     accept = np.zeros(N)  # 存储每个样本的接受状态
    #     dim_theta = args.input_theta_dim
    #     dim_u = args.aux_dim
    #     # # 初始化 y0 的前半部分为 0，后半部分从正态分布中采样
    #     # for ii in np.arange(0, int(args.input_dim / 2), 1):
    #     #     y0[ii] = 0.0
    #     # for ii in np.arange(int(args.input_dim / 2), int(args.input_dim), 1):
    #     #     y0[ii] = norm(loc=0, scale=1).rvs()
    #     y0[args.input_theta_dim:2*args.input_theta_dim] = np.random.randn(args.input_theta_dim)  # rho
    #     y0[2 * args.input_theta_dim : ] = np.random.randn(2*args.aux_dim)
    #     # 用于存储哈密顿轨迹的数组
    #     HNN_sto = np.zeros((input_dim, len(t_eval), N))
        
    #     # 进行 N 次采样
    #     for ii in tqdm(np.arange(0, N, 1)):
    #         # 使用 leapfrog 方法进行哈密顿动力学积分
    #         hnn_ivp = integrate_model(phnn_model, t_span = args.t_span, y0 = y0, n = len(t_eval) - 1, input_theta_dim = args.input_theta_dim, h = h, model = args.model, aux_dim = args.aux_dim)
    #         # for sss in range(0, input_dim):
    #         #     HNN_sto[sss, :, ii] = hnn_ivp[sss, :]
    #         HNN_sto[:, :, ii] = hnn_ivp[:, :]
    
    #         # 计算新的哈密顿量
    #         yhamil = np.zeros(input_dim, dtype=np.float32)
    
            
    #         # for jj in np.arange(0, input_dim, 1):
    #         #     yhamil[jj] = hnn_ivp[jj, len(t_eval) - 1]
    #         yhamil[:] = hnn_ivp[:, len(t_eval) - 1]
    #         theta_star = yhamil[ :dim_theta]  # 第一部分: theta
    #         rho_star = yhamil[ dim_theta:2 * dim_theta] # 第二部分: phi
    #         u_star = yhamil[ 2 * dim_theta:2 * dim_theta + dim_u] # 第三部分: u
    #         p_star = yhamil[ -dim_u:]  # 第四部分: p
    #         H_star = compute_hamiltonian(theta_star, rho_star, u_star, p_star, y, args.model) # 新的哈密顿量
            
    #         theta_0 = y0[ :dim_theta]  # 第一部分: theta
    #         rho_0 = y0[ dim_theta:2 * dim_theta] # 第二部分: phi
    #         u_0 = y0[ 2 * dim_theta:2 * dim_theta + dim_u] # 第三部分: u
    #         p_0 = y0[ -dim_u:]  # 第四部分: p
    #         H_prev = compute_hamiltonian(theta_0, rho_0, u_0, p_0, y, args.model)  # 之前的哈密顿量
            
    #         # Metropolis-Hastings 接受率
    #         alpha = np.minimum(1, np.exp(H_prev - H_star))
            
    #         # 如果接受了新的样本，更新 y0
    #         if alpha > uniform().rvs():
    #             y0[0:args.input_theta_dim] = hnn_ivp[0:args.input_theta_dim, steps - 1]
    #             x_req[ii, :] = hnn_ivp[0:args.input_theta_dim, steps - 1]
    #             accept[ii] = 1
    #         else:
    #             x_req[ii, :] = y0[0:args.input_theta_dim]
            
    #         # 每次更新后，重新采样动量部分
    #         # for jj in np.arange(args.input_theta_dim, input_dim, 1):
    #         #     y0[jj] = norm(loc=0, scale=1).rvs()
    #         y0[args.input_theta_dim:input_dim] = norm(loc=0, scale=1).rvs(size=(input_dim - args.input_theta_dim))
    #        # print(f"Sample: {ii} Chain: {ss}")
        
    #     # 存储当前链的接受率和采样结果
    #     hnn_accept[ss, :] = accept
    #     hnn_fin[ss, :, :] = x_req
    # 保存 hnn_accept 和 hnn_fin 为 pkl 文件
    with open("hnn_kernel.pkl", "wb") as f:
        pickle.dump(kernel_results, f)
    
    with open("hnn_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
    print("files save")
    
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

