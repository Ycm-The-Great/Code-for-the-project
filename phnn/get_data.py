#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 00:13:26 2025

@author: yanchuanmiao
"""

import numpy as np
import tensorflow as tf
from scipy.stats import norm
from utils import to_pickle, from_pickle
from tqdm import tqdm

def initialize_data(model, T, N, **kwargs):
    """
    根据指定的模型类型初始化 y, theta, rho, u, p。
    
    参数：
    - model: 模型类型，支持 "gaussian", "diffraction", "glmm"。
    - T: 时间序列的长度。
    - N: 每个时间点的粒子数量。
    - **kwargs: 根据模型不同传入的其他参数。
    
    返回值：
    - y: 模拟生成的观测数据。
    - theta: 模型参数。
    - rho: 与 theta 同维的标准正态分布。
    - u: 隐变量，形状为 [T * N * dim(theta)] 的标准正态分布。
    - p: 隐变量动量，与 u 形状相同的标准正态分布。
    """
    if model == "gaussian":
        # 参数
        mu_theta = kwargs.get("mu_theta", 0.0)
        sigma_theta = kwargs.get("sigma_theta", 10.0)
        sigma_x = kwargs.get("sigma_x", 0.1)
        sigma_y = kwargs.get("sigma_y", 1.0)

        # 初始化 theta
        theta = np.random.normal(loc=0.0, scale=sigma_theta, size=2)  # 2 维初始化
        theta = tf.constant(theta, dtype=tf.float32)  # 转为 Tensor

        
        

        # 生成 X_k ~ N(mu_theta, sigma_x^2)
        x = np.random.normal(loc=mu_theta, scale=sigma_x, size=T)

        # 生成 Y_k | X_k ~ N(X_k, sigma_y^2)
        y = np.random.normal(loc=x, scale=sigma_y, size=T)

    elif model == "diffraction":
        # 参数
        mu = kwargs.get("mu", 1.0)
        sigma = kwargs.get("sigma", 1.0)
        lambda_ = kwargs.get("lambda_", 0.1)

        # 初始化 theta
        theta = np.random.normal(loc=0.0, scale=100.0, size=3)  # 3 维初始化
        theta = tf.constant(theta, dtype=tf.float32)  # 转为 Tensor

        

        # 生成 X_k ~ N(mu, sigma^2)
        x = np.random.normal(loc=mu, scale=sigma, size=T)

        # 生成 Y_k | X_k ~ g(y | x, lambda)
        y = x + np.random.uniform(-1 / lambda_, 1 / lambda_, size=T)

    elif model == "glmm":
        # 参数
        p = kwargs.get("p", 8)  # beta 的维度
        mu1 = kwargs.get("mu1", 0.0)
        mu2 = kwargs.get("mu2", 3.0)
        lambda1 = kwargs.get("lambda1", 10.0)
        lambda2 = kwargs.get("lambda2", 3.0)
        w1 = kwargs.get("w1", 0.5)
        n_i = kwargs.get("n_i", 6)  # 每个主体的样本数量

        # 初始化 beta 和混合分布参数
        beta = np.random.normal(loc=0.0, scale=1.0, size=p)
        theta = np.random.normal(loc=0.0, scale=100.0, size=p + 5)  # p + 5 维初始化
        theta = tf.constant(theta, dtype=tf.float32)  # 转为 Tensor

        
        # 从 theta 中提取参数
        #beta = theta[:-5].numpy()
        # mu1, mu2 = theta[-5].numpy(), theta[-4].numpy()
        # lambda1, lambda2 = np.exp(theta[-3].numpy()), np.exp(theta[-2].numpy())
        # w1 = tf.sigmoid(theta[-1]).numpy()

       # 模拟生成 X_i ~ 混合分布
        indicator = np.random.choice([0, 1], size=T, p=[w1, 1 - w1])
        X_i = np.where(
            indicator == 0,
            np.random.normal(loc=mu1, scale=1 / np.sqrt(lambda1), size=T),
            np.random.normal(loc=mu2, scale=1 / np.sqrt(lambda2), size=T)
        )

        # 生成 Z_ij 和 Y_ij
        Z_ij = np.random.normal(loc=0.0, scale=1.0, size=(T, n_i, p))
        linear_predictor = np.einsum("ijp,p->ij", Z_ij, beta) + X_i[:, None]
        p_ij = 1 / (1 + np.exp(-linear_predictor))  # logistic link function
        y = np.random.binomial(n=1, p=p_ij)
        y = tf.reshape(y, [-1])
        y = tf.cast(y, dtype=tf.float32)

    else:
        raise ValueError("Invalid model. Choose from 'gaussian', 'diffraction', or 'glmm'.")
    if model == "glmm":
        # 初始化 rho, u, p
        rho = np.random.normal(loc=0.0, scale=1.0, size=theta.shape)
        u = np.random.normal(loc=0.0, scale=1.0, size=(T * N * n_i * theta.shape[0]))
        p = np.random.normal(loc=0.0, scale=1.0, size=(T * N * n_i * theta.shape[0]))
    else:
        rho = np.random.normal(loc=0.0, scale=1.0, size=theta.shape)
        u = np.random.normal(loc=0.0, scale=1.0, size=(T * N * theta.shape[0]))
        p = np.random.normal(loc=0.0, scale=1.0, size=(T * N * theta.shape[0]))
    return tf.constant(y, dtype=tf.float32), theta, tf.constant(rho, dtype=tf.float32), tf.constant(u, dtype=tf.float32), tf.constant(p, dtype=tf.float32)


def compute_weights_and_x_gaussian(y, theta, u, sigma_x, sigma_y):
    """
    Gaussian 模型的权重计算。
    """
    T = tf.shape(y)[0]
    N = tf.shape(u)[0] // (T * tf.shape(theta)[0])  # 考虑 theta 的维度

    # 生成 X_k,i = theta + sigma_x * U_k,i
    theta_exp = tf.expand_dims(theta, axis=0)  # (1, theta_dim)
    u_reshaped = tf.reshape(u, (T, N))  # (T, N)
    x = theta_exp + sigma_x * u_reshaped  # 广播计算 X_k,i

    # 计算权重
    sigma_y2 = sigma_y ** 2
    weights = tf.exp(-0.5 * ((y[:, None] - x) ** 2) / sigma_y2) / tf.sqrt(2 * np.pi * sigma_y2)
    return weights, x


def compute_likelihood_gradients_gaussian(y, theta, u, sigma_x, sigma_y):
    """
    Gaussian 模型的梯度计算。
    """
    T = tf.shape(y)[0]
    N = tf.shape(u)[0] // (T * tf.shape(theta)[0])
    sigma_y2 = sigma_y ** 2

    # 计算权重和 X_k,i
    weights, x = compute_weights_and_x_gaussian(y, theta, u, sigma_x, sigma_y)

    # 归一化权重
    weights_sum = tf.reduce_sum(weights, axis=1, keepdims=True)  # 形状 [T, 1]
    normalized_weights = weights / weights_sum  # 形状 [T, N]

    # 对 theta 的梯度
    grad_theta = tf.reduce_sum(normalized_weights * (y[:, None] - x) / sigma_y2) / sigma_x

    # 对 u 的梯度
    grad_u = tf.TensorArray(tf.float32, size=T * N)
    for k in range(T):
        for i in range(N):
            grad_log_weight = (y[k] - x[k, i]) / sigma_y2 * sigma_x
            grad_u = grad_u.write(k * N + i, normalized_weights[k, i] * grad_log_weight)
    grad_u = grad_u.stack()

    return grad_theta, grad_u

def compute_weights_and_x_diffraction(y, theta, u, sigma_x, lambda_):
    """
    Diffraction 模型的权重计算。
    """
    T = tf.shape(y)[0]
    N = tf.shape(u)[0] // (T * tf.shape(theta)[0])  # 考虑 theta 的维度

    # 生成 X_k,i = theta + sigma_x * U_k,i
    theta_exp = tf.expand_dims(theta, axis=0)  # (1, theta_dim)
    u_reshaped = tf.reshape(u, (T, N))  # (T, N)
    x = theta_exp + sigma_x * u_reshaped  # 广播计算 X_k,i

    # 计算权重 (sinc^2)
    diff = (y[:, None] - x) / lambda_
    weights = tf.sinc(diff) ** 2
    return weights, x


def compute_likelihood_gradients_diffraction(y, theta, u, sigma_x, lambda_):
    """
    Diffraction 模型的梯度计算。
    """
    T = tf.shape(y)[0]
    N = tf.shape(u)[0] // (T * tf.shape(theta)[0])

    # 计算权重和 X_k,i
    weights, x = compute_weights_and_x_diffraction(y, theta, u, sigma_x, lambda_)

    # 归一化权重
    weights_sum = tf.reduce_sum(weights, axis=1, keepdims=True)  # 形状 [T, 1]
    normalized_weights = weights / weights_sum  # 形状 [T, N]

    # 对 theta 的梯度
    grad_theta = tf.reduce_sum(normalized_weights * (-2 * (y[:, None] - x) * tf.sinc((y[:, None] - x) / lambda_))) / (
        lambda_ * sigma_x
    )

    # 对 u 的梯度
    grad_u = tf.TensorArray(tf.float32, size=T * N)
    for k in range(T):
        for i in range(N):
            grad_log_weight = -2 * (y[k] - x[k, i]) * tf.sinc((y[k] - x[k, i]) / lambda_) / lambda_
            grad_u = grad_u.write(k * N + i, normalized_weights[k, i] * grad_log_weight)
    grad_u = grad_u.stack()

    return grad_theta, grad_u

def compute_weights_and_x_glmm(y, theta, u, sigma_x):
    """
    GLMM 模型的权重计算。
    
    参数：
    - y: 观测数据，形状 [T]。
    - theta: 模型参数向量，包含 beta 和混合分布的参数。
      - 前 d-5 维是 beta。
      - 后 5 维是 [mu1, mu2, log_lambda1, log_lambda2, logit_w1]。
    - u: 隐变量向量，形状 [T * N * dim(theta)]。
    - sigma_x: 高斯噪声的标准差。
    
    返回值：
    - weights: 权重矩阵，形状 [T, N]。
    - x: 生成的 X_k,i 矩阵，形状 [T, N]。
    """
    T = tf.shape(y)[0]
    d = tf.shape(theta)[0]
    N = tf.shape(u)[0] // (T * d)  # 隐变量的数量除以 T 和 theta 的维度
    # 显式连接 theta 和 u 到计算图
    theta = tf.convert_to_tensor(theta)
    u = tf.convert_to_tensor(u)
    # 提取 theta 的各部分
    beta = theta[:-5]  # 前 d-5 维是 beta
    mu1, mu2 = theta[-5], theta[-4]  # 混合分布的均值
    log_lambda1, log_lambda2 = theta[-3], theta[-2]  # 混合分布的对数精度
    logit_w1 = theta[-1]  # 混合分布中第一个分量的权重

    # 计算混合分布的参数
    lambda1, lambda2 = tf.exp(log_lambda1), tf.exp(log_lambda2)  # 将 log_lambda 转化为精度
    # print(theta)
    w1 = tf.sigmoid(logit_w1)+0.001  # 将 logit_w1 转化为概率
    w2 = 1 - w1  # 第二个分量的权重

    # 生成 X_k,i = theta + sigma_x * U_k,i
    u_reshaped = tf.reshape(u, (T, N, d))  # 隐变量 reshape 为 [T, N, d]
    u_reshaped = tf.debugging.check_numerics(u_reshaped, "u_reshaped 包含非法值")
    
    x = theta + sigma_x * u_reshaped  # 广播计算 X_k,i
    # print(x)
    # print(lambda1,lambda2)
    # 计算混合分布的概率密度
    p1 = w1 * tf.exp(-0.5 * lambda1 * (x[:, :, -5] - mu1) ** 2) * tf.sqrt(lambda1 / (2 * np.pi))
    p1 = tf.debugging.check_numerics(p1, "p1 包含非法值")
    p2 = w2 * tf.exp(-0.5 * lambda2 * (x[:, :, -4] - mu2) ** 2) * tf.sqrt(lambda2/ (2 * np.pi))
    p2 = tf.debugging.check_numerics(p2, "p2 包含非法值")

    
    # 计算 Bernoulli 概率 p_ij
    linear_predictor = tf.tensordot(x[:, :, :-5], beta, axes=1)  # (T, N)
    p_ij = tf.sigmoid(linear_predictor)
    
    # 计算权重
    weights = tf.expand_dims(y, axis=1) * p_ij  + (1 - tf.expand_dims(y, axis=1)) * (1 - p_ij )
    weights = (weights * (p1 + p2)) + 0.001 # 结合混合分布的概率密度
    
    weights = tf.debugging.check_numerics(weights, "weights 包含非法值")
    return weights, x


def compute_likelihood_gradients_glmm(y, theta, u, sigma_x):
    """
    GLMM 模型的梯度计算。
    
    参数：
    - y: 观测数据，形状 [T]。
    - theta: 模型参数向量，包含 beta 和混合分布的参数。
    - u: 隐变量向量，形状 [T * N * dim(theta)]。
    - sigma_x: 高斯噪声的标准差。
    
    返回值：
    - grad_theta: 对 theta 的似然梯度，形状与 theta 相同。
    - grad_u: 对 u 的似然梯度，形状与 u 相同。
    """
    T = tf.shape(y)[0]
    d = tf.shape(theta)[0]
    N = tf.shape(u)[0] // (T * d)  # 隐变量的数量除以 T 和 theta 的维度

    # 计算权重和 X_k,i
    weights, x = compute_weights_and_x_glmm(y, theta, u, sigma_x)
    weights = tf.debugging.check_numerics(weights, "weights 包含非法值")
    
    # 归一化权重
    weights_sum = tf.reduce_sum(weights, axis=1, keepdims=True)  # 形状 [T, 1]
    normalized_weights = weights / (weights_sum + 0.0001) # 形状 [T, N]
    normalized_weights = tf.debugging.check_numerics(normalized_weights, "normalized_weights 包含非法值")

    # 对 theta 的梯度
    # grad_theta = tf.zeros_like(theta)
    # with tf.GradientTape() as tape:
    #     tape.watch(theta)  # 确保追踪 theta
    #     for k in range(T):
    #         for i in range(N):
    #             log_omega = weights[k, i]  # 提取预计算的 weights
    #             grad_log_omega = tape.gradient(log_omega, theta)  # 对 theta 求梯度
    
    # # for k in range(T):
    # #     for i in range(N):
    # #         with tf.GradientTape() as tape:
    # #             tape.watch(theta)  # 记录 theta 变量
    # #             weights, x = compute_weights_and_x_glmm(y, theta, u, sigma_x)
    # #             log_omega = weights[k, i]
    # #         grad_log_omega = tape.gradient(log_omega, theta)
    #     grad_theta += normalized_weights[k, i] * grad_log_omega
    #         # grad_log_omega = tf.gradients(weights[k, i], theta)[0]  # 对 theta 求梯度
    #         # grad_theta += normalized_weights[k, i] * grad_log_omega
   
    # # 对 u 的梯度
    # grad_u = tf.TensorArray(tf.float32, size=T * N)
    # for k in range(T):
    #     for i in range(N):
    #         with tf.GradientTape() as tape:
    #             tape.watch(u)  # 记录 u 变量
    #             log_weight = weights[k, i]
    
    #         # 对 log_weight 求 u 的梯度
    #         grad_log_weight = tape.gradient(log_weight, u)
    #         grad_u = grad_u.write(k * N + i, normalized_weights[k, i] * grad_log_weight)

    #         # grad_log_weight = tf.gradients(weights[k, i], u)[0]  # 对 u 求梯度
    #         # grad_u = grad_u.write(k * N + i, normalized_weights[k, i] * grad_log_weight)
    # grad_u = grad_u.stack()
    # 计算 theta 和 u 的梯度
    grad_theta = tf.zeros_like(theta)
    grad_u = tf.zeros_like(u)
    with tf.GradientTape() as tape:
        tape.watch([theta, u])
        # 前向计算权重
        weights, x = compute_weights_and_x_glmm(y, theta, u, sigma_x)
        # 限制 weights 的范围（防止梯度消失/爆炸）
        # weights = tf.clip_by_value(weights, -10.0, 10.0)
        log_weights = tf.math.log(tf.maximum(weights,0.00001))
        # 构造标量损失函数：加权和（等价于逐元素梯度累加）
        loss = tf.reduce_sum(log_weights * tf.stop_gradient(weights/normalized_weights))  # 假设 normalized_weights 是预计算的常数

    # 一次性计算梯度
    grad_theta, grad_u = tape.gradient(loss, [theta, u])
    # print(grad_theta, grad_u,weights/normalized_weights)
    # grad_theta, grad_u = tf.reduce_sum(grad_theta*normalized_weights),tf.reduce_sum(grad_u*normalized_weights)
    # # 使用持久性梯度带一次性计算所有权重的梯度
    # with tf.GradientTape(persistent=True) as tape:
    #     tape.watch([theta, u])  # 同时追踪 theta 和 u
    #     # 在 Tape 上下文中计算权重（确保梯度可追踪）
    #     weights, x = compute_weights_and_x_glmm(y, theta, u, sigma_x)  # 确保此函数在 Tape 内执行
    
    #     # 批量提取梯度（避免逐样本循环）
    #     for k in tf.range(T):
    #         for i in tf.range(N):
    #             # 提取预计算的权重（已在 Tape 上下文中追踪）
    #             log_omega = weights[k, i]
                
    #             # 对 theta 的梯度
    #             grad_log_omega_theta = tape.gradient(log_omega, theta)
    #             grad_theta += normalized_weights[k, i] * grad_log_omega_theta
                
    #             # 对 u 的梯度（直接操作张量，避免 TensorArray）
    #             grad_log_omega_u = tape.gradient(log_omega, u)
    #             grad_u += normalized_weights[k, i] * grad_log_omega_u
    
    # # 释放持久性 Tape 资源
    # del tape
    # 限制梯度绝对值不超过 1.0
    clip_value = 1.0
    grad_theta = tf.clip_by_value(grad_theta, -clip_value, clip_value)
    grad_u = tf.clip_by_value(grad_u, -clip_value, clip_value)
   
    grad_theta = tf.debugging.check_numerics(grad_theta, "grad_theta 包含非法值")
    grad_u = tf.debugging.check_numerics(grad_u, "grad_u 包含非法值")
    return grad_theta, grad_u

def compute_hamiltonian(theta, rho, u, p, y, model, **kwargs):
    """
    计算扩展 Hamiltonian H(θ, ρ, u, p)。
    
    参数：
    - theta: 模型的参数向量（包含先验和数据生成过程的所有参数）。
      - 对于 Gaussian 模型，theta 是 2 维，分别表示 [mu_theta, sigma_x, sigma_y]。
      - 对于 Diffraction 模型，theta 是 3 维，分别表示 [mu, log_sigma, log_lambda]。
      - 对于 GLMM 模型，theta 是 5 维以上，前几维是 beta，后 5 维分别是 [mu1, mu2, log_lambda1, log_lambda2, logit_w1]。
    - rho: 动量变量，对应 theta。
    - u: 隐变量的动量。
    - p: 其他动量变量。
    - y: 观测数据。
    - model: 指定模型类型，支持 'gaussian', 'diffraction', 或 'glmm'。
    - **kwargs: 其他可选参数。

    返回值：
    - Hamiltonian 值（标量）。
    """
    if model == 'gaussian':
        # 从 theta 中提取参数
        mu_theta = kwargs.get("mu_theta", 0.0)
        sigma_x = tf.exp(theta[0])  # 确保 sigma_x > 0
        sigma_y = tf.exp(theta[1])  # 确保 sigma_y > 0

        # 先验 log p(θ)
        sigma_theta = kwargs.get('sigma_theta', 10.0)
        log_prior = -0.5 * ((mu_theta**2) / sigma_theta**2 + tf.math.log(2 * np.pi * sigma_theta**2))

        # 似然 log \hat{p}(y | θ, u)
        weights, _ = compute_weights_and_x_gaussian(y, mu_theta, u, sigma_x, sigma_y)
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1)))

    elif model == 'diffraction':
        # 从 theta 中提取参数
        mu = theta[0]
        log_sigma = theta[1]
        log_lambda = theta[2]
        sigma = tf.exp(log_sigma)  # 确保 sigma > 0
        lambda_ = tf.exp(log_lambda)  # 确保 lambda > 0

        # 先验 log p(θ)
        log_prior = -0.5 * (
            mu**2 / 100.0 + log_sigma**2 / 100.0 + log_lambda**2 / 100.0
        )

        # 似然 log \hat{p}(y | θ, u)
        weights, _ = compute_weights_and_x_diffraction(y, mu, u, sigma, lambda_)
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1)))

    elif model == 'glmm':
        # 从 theta 中提取参数
        d = tf.shape(theta)[0]  # theta 的维度
        beta = theta[:-5]  # 前 d-5 维是 beta
        mu1, mu2 = theta[-5], theta[-4]
        log_lambda1, log_lambda2 = theta[-3], theta[-2]
        logit_w1 = theta[-1]
        lambda1, lambda2 = tf.exp(log_lambda1), tf.exp(log_lambda2)  # 确保 lambda > 0
        w1 = tf.sigmoid(logit_w1)  # 确保 w1 在 (0, 1)

        # 先验 log p(θ)
        log_prior = -0.5 * (
            tf.reduce_sum(beta**2) / 100.0 +
            mu1**2 / 100.0 +
            mu2**2 / 100.0 +
            log_lambda1**2 / 100.0 +
            log_lambda2**2 / 100.0 +
            logit_w1**2 / 100.0
        )

       
        # 似然 log \hat{p}(y | θ, u)
        weights, _ = compute_weights_and_x_glmm(y, theta, u, sigma_x=kwargs.get('sigma_x', 0.1))
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1)+0.001))

    else:
        raise ValueError("Invalid model. Choose from 'gaussian', 'diffraction', or 'glmm'.")

    # 动量项
    kinetic_energy = 0.5 * (tf.reduce_sum(rho**2) + tf.reduce_sum(u**2) + tf.reduce_sum(p**2))
    H = -log_prior - log_likelihood + kinetic_energy
    # Hamiltonian
    return H


def compute_time_derivative(theta, rho, u, p, y, model, **kwargs):
    """
    计算时间导数 d/dt [θ, ρ, u, p]。
    
    参数：
    - theta: 模型的参数向量（结构同上）。
    - rho: 动量变量，对应 theta。
    - u: 隐变量的动量。
    - p: 其他动量变量。
    - y: 观测数据。
    - model: 指定模型类型，支持 'gaussian', 'diffraction', 或 'glmm'。
    - **kwargs: 其他可选参数。

    返回值：
    - dtheta_dt: theta 的时间导数。
    - drho_dt: rho 的时间导数。
    - du_dt: u 的时间导数。
    - dp_dt: p 的时间导数。
    """
    if model == 'gaussian':
        # 从 theta 中提取参数
        mu_theta = theta[0]
        sigma_x = tf.exp(theta[1])
        sigma_y = tf.exp(theta[2])

        # 梯度
        sigma_theta = kwargs.get('sigma_theta', 10.0)
        grad_theta_prior = tf.constant([-mu_theta / sigma_theta**2, 0.0, 0.0])
        grad_theta_likelihood, grad_u_likelihood = compute_likelihood_gradients_gaussian(y, mu_theta, u, sigma_x, sigma_y)

    elif model == 'diffraction':
        # 从 theta 中提取参数
        mu = theta[0]
        log_sigma = theta[1]
        log_lambda = theta[2]
        lambda_ = tf.exp(log_lambda)
        sigma = tf.exp(log_sigma)

        # 先验梯度
        grad_theta_prior = tf.constant([-mu / 100.0, -log_sigma / 100.0, -log_lambda / 100.0])

        # 似然梯度
        grad_mu_likelihood, grad_u_likelihood = compute_likelihood_gradients_diffraction(y, mu, u, lambda_)
        grad_theta_likelihood = tf.concat(
            [grad_mu_likelihood, tf.zeros_like(log_sigma), tf.zeros_like(log_lambda)], axis=0
        )

    elif model == 'glmm':
        # 从 theta 中提取参数
        d = tf.shape(theta)[0]
        beta = theta[:-5]
        mu1, mu2 = theta[-5], theta[-4]
        log_lambda1, log_lambda2 = theta[-3], theta[-2]
        logit_w1 = theta[-1]

        # 先验梯度
        grad_theta_prior = tf.concat([
            -beta / 100.0,
            tf.expand_dims(-mu1 / 100.0, axis=0),
            tf.expand_dims(-mu2 / 100.0, axis=0),
            tf.expand_dims(-log_lambda1 / 100.0, axis=0),
            tf.expand_dims(-log_lambda2 / 100.0, axis=0),
            tf.expand_dims(-logit_w1 / 100.0, axis=0)
        ], axis=0)

        # 似然梯度
        grad_theta_likelihood, grad_u_likelihood = compute_likelihood_gradients_glmm(y, theta, u, kwargs.get('sigma_x', 0.1))
        
        # grad_theta_likelihood = tf.concat([grad_beta_likelihood, tf.zeros(5)], axis=0)

    else:
        raise ValueError("Invalid model. Choose from 'gaussian', 'diffraction', or 'glmm'.")

    # 时间导数
    dtheta_dt = rho
    drho_dt = grad_theta_prior + grad_theta_likelihood
    du_dt = -u + grad_u_likelihood
    dp_dt = p

    return dtheta_dt, drho_dt, du_dt, dp_dt


def leapfrog_special(dynamics_fn, t_span, y0, n_steps, input_dim, h, model, y_data, **kwargs):
    """
    自定义 Leapfrog 积分器，严格按要求的更新步骤实现
    """
    # 解包初始状态 [theta, rho, u, p]
    theta = y0[:input_dim] 
    rho = y0[input_dim:2*input_dim]
    u = y0[2*input_dim:2*input_dim+kwargs['aux_dim']]
    p = y0[2*input_dim+kwargs['aux_dim']:]
    
    # 转换为 TensorFlow 张量
    theta = tf.convert_to_tensor(theta, dtype=tf.float32)
    rho = tf.convert_to_tensor(rho, dtype=tf.float32)
    u = tf.convert_to_tensor(u, dtype=tf.float32)
    p = tf.convert_to_tensor(p, dtype=tf.float32)
    
    # 存储轨迹的列表
    trajectory = [tf.concat([theta, rho, u, p], axis=0).numpy()]
    derivatives = []  # 新增：存储导数

    for _ in range(n_steps):
        dtheta_dt, drho_dt, du_dt, dp_dt = dynamics_fn(theta, rho, u, p, y_data, model, **kwargs)
        derivatives.append(tf.concat([dtheta_dt, drho_dt, du_dt, dp_dt], axis=0).numpy())
        # A 步骤 (动量更新前半)
        u_half = u * tf.cos(h/2) + p * tf.sin(h/2)
        # p_half = -u * tf.sin(h/2) + p * tf.cos(h/2)
        theta_half = theta + (h/2) * rho
        
        # 通过动力学函数计算梯度项
        if model == 'gaussian':
            # 从 theta 中提取参数
            mu_theta = theta[0]
            sigma_x = tf.exp(theta[1])
            sigma_y = tf.exp(theta[2])

           
            grad_theta, grad_u  = compute_likelihood_gradients_gaussian(y_data, mu_theta, u, sigma_x, sigma_y)

        elif model == 'diffraction':
            # 从 theta 中提取参数
            mu = theta[0]
            log_sigma = theta[1]
            log_lambda = theta[2]
            lambda_ = tf.exp(log_lambda)
            sigma = tf.exp(log_sigma)
      

            # 似然梯度
            grad_theta, grad_u  = compute_likelihood_gradients_diffraction(y_data, mu, u, lambda_)
           

        elif model == 'glmm':
            # 从 theta 中提取参数
           
            # 似然梯度
            grad_theta, grad_u = compute_likelihood_gradients_glmm(y_data, theta_half, u_half, kwargs.get('sigma_x', 0.1))
        # _, grad_rho, grad_u, grad_p = dynamics_fn(theta_half, rho, u_half, p_half, y_data, model, **kwargs)
        
        # 应用梯度更新
        u_half = u_half + tf.sin(h/2) * h * grad_u
       
        
        # B 步骤 (位置更新)
        # dtheta_dt, drho_dt, _, _ = dynamics_fn(theta_half, rho, u_half, p_half, y_data, model, **kwargs)
        theta = theta + h * rho + (h**2 / 2) * grad_theta
        rho = rho + h * grad_theta
        
        # A 步骤 (动量更新后半)
        u = u * tf.cos(h) + p * tf.sin(h)
        p = -u * tf.sin(h) + p * tf.cos(h)+ tf.cos(h/2) * h * grad_u
        
        dtheta_dt, drho_dt, du_dt, dp_dt = dynamics_fn(theta, rho, u, p, y_data, model, **kwargs)

        # 记录轨迹
        trajectory.append(tf.concat([theta, rho, u, p], axis=0).numpy())
    derivatives.append(derivatives[-1])
    
    return np.array(trajectory).T, np.array(derivatives).T  # 形状 [dim, n_steps+1]

def leapfrog_traditional(dynamics_fn, t_span, y0, n_steps, input_dim, h, model, y_data, **kwargs):
    """
    传统 Leapfrog 积分器（完整更新 theta, rho, u, p）
    """
    # 解包初始状态
    theta = tf.convert_to_tensor(y0[:input_dim], dtype=tf.float32)
    rho = tf.convert_to_tensor(y0[input_dim:2*input_dim], dtype=tf.float32)
    u = tf.convert_to_tensor(y0[2*input_dim:2*input_dim+kwargs['aux_dim']], dtype=tf.float32)
    p = tf.convert_to_tensor(y0[2*input_dim+kwargs['aux_dim']:], dtype=tf.float32)
    
    trajectory = [tf.concat([theta, rho, u, p], axis=0).numpy()]
    derivatives = []  # 新增：存储导数

    for _ in range(n_steps):
        # 1. 动量半步更新：rho 和 p
        dtheta_dt, drho_dt, du_dt, dp_dt = dynamics_fn(theta, rho, u, p, y_data, model, **kwargs)
        derivatives.append(tf.concat([dtheta_dt, drho_dt, du_dt, dp_dt], axis=0).numpy())
        rho_half = rho + 0.5 * h * drho_dt
        p_half = p + 0.5 * h * dp_dt
        
        # 2. 位置全步更新：theta 和 u
        theta = theta + h * dtheta_dt
        u = u + h * du_dt
        
        # 3. 动量再半步更新：rho 和 p（基于新位置）
        dtheta_dt_new, drho_dt_new, du_dt_new, dp_dt_new = dynamics_fn(theta, rho_half, u, p_half, y_data, model, **kwargs)
        rho = rho_half + 0.5 * h * drho_dt_new
        p = p_half + 0.5 * h * dp_dt_new
        
        # 记录轨迹
        trajectory.append(tf.concat([theta, rho, u, p], axis=0).numpy())
    derivatives.append(derivatives[-1])
    return np.array(trajectory).T, np.array(derivatives).T  # 形状 [input_dim, n_steps+1]

def get_trajectory(
    t_span=[0, 10],  
    timescale=10,  
    y0=None,  
    dynamics_fn=None,  
    input_theta_dim=4,  
    aux_dim=2,  
    model='glmm',
    y_data=None,
    **kwargs
):
    """
    修改后的轨迹生成函数，适配 HNN 训练数据格式
    """
    input_dim = 2 * input_theta_dim + 2 * aux_dim  
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
    h = (t_span[1] - t_span[0]) / len(t_eval)  # 计算步长
    # 初始化状态
    if y0 is None:
        y0 = np.zeros(input_dim)
        # theta 初始化为零
        y0[input_theta_dim:2*input_theta_dim] = np.random.randn(input_theta_dim)  # rho
        y0[2*input_theta_dim:] = np.random.randn(2*aux_dim)  # u 和 p

    # 使用自定义 Leapfrog 积分
    trajectory, derivatives = leapfrog_special(
        dynamics_fn=dynamics_fn,
        t_span=t_span,
        y0=y0,
        n_steps=len(t_eval)-1,  
        input_dim=input_theta_dim,
        h=h,
        model=model,
        y_data=y_data,
        aux_dim=aux_dim,
        **kwargs
    )
    
    # 分解轨迹数据
    dic1 = np.split(trajectory, trajectory.shape[0])
    ddic1 = np.split(derivatives, trajectory.shape[0])
    
    return dic1, ddic1, t_eval

def get_dataset(
    num_samples: int = 1000,
    test_split: float = 0.2,
    input_theta_dim: int = 4,
    aux_dim: int = 2,
    model: str = 'glmm',
    y_data: np.ndarray = None,
    save_path: str = None,
    should_load: bool = False,  # 新增：是否加载数据
    load_path: str = None,      # 新增：数据加载路径
    **kwargs
):
    """
    生成/加载轨迹数据集 (带加载控制参数)
    """
    input_dim = 2 * input_theta_dim + 2 * aux_dim

    # 若需加载且路径有效，则尝试加载
    if should_load and load_path:
        try:
            data = from_pickle(load_path)
            print(f"Data loaded from {load_path}")
            return data
        except FileNotFoundError:
            print(f"No data found at {load_path}, generating new data")
            should_load = False  # 加载失败后强制生成数据

    # 若不需要加载或加载失败，则生成新数据
    if not should_load:
        data = {}
        np.random.seed(0)
        xs, dxs = [], []
        y_init = np.zeros(input_dim)
        # theta 初始化为零
        y_init[input_theta_dim:2*input_theta_dim] = np.random.randn(input_theta_dim)  # rho
        y_init[2*input_theta_dim:] = np.random.randn(2*aux_dim)  # u 和 p  # 初始化后半部分
        
        for s in tqdm(range(num_samples)):
            print(f"Generating sample {s+1}/{num_samples}")
            dic1, ddic1, t = get_trajectory(
                y0=y_init,
                input_theta_dim=input_theta_dim,
                aux_dim=aux_dim,
                model=model,
                y_data=y_data,
                **kwargs
            )
            # 轨迹和导数重组
            x = np.concatenate([var.T.flatten() for var in dic1])
            dx = np.concatenate([var.T.flatten() for var in ddic1])
            xs.append(x)
            dxs.append(dx)
            
            # 更新初始条件
            y_init[input_theta_dim:2*input_theta_dim] = np.random.randn(input_theta_dim)  # rho
            y_init[2*input_theta_dim:] = np.random.randn(2*aux_dim)  # u 和 p  # 初始化后半部分
        
        # 合并数据集
        data['coords'] = np.stack(xs)
        data['dcoords'] = np.stack(dxs)
        
        # 分割训练/测试集
        split_ix = int(num_samples * (1 - test_split))
        split_iy = num_samples - split_ix
        data_train = {
            'coords': data['coords'][:split_ix].reshape(split_ix*len(t),-1),
            'dcoords': data['dcoords'][:split_ix].reshape(split_ix*len(t),-1)
        }
        if test_split != 0:            
            data_test = {
                'coords': data['coords'][split_ix:].reshape(split_iy*len(t),-1),
                'dcoords': data['dcoords'][split_ix:].reshape(split_iy*len(t),-1)
            }
        else:
            data_test = {
                'coords': data['coords'][split_ix:],
                'dcoords': data['dcoords'][split_ix:]
            }
        data = {'train': data_train, 'test': data_test}

        # 保存数据
        if save_path:
            to_pickle(data, save_path)
            print(f"Data saved to {save_path}")

        return data

    return {}

# # 示例
# if __name__ == "__main__":
#     T = 30
#     N = 10
#     y, theta, rho, u, p = initialize_data(
#         model="gaussian", T=T, N=N, mu_theta=0.0, sigma_theta=10.0, sigma_x=0.1, sigma_y=1.0
#     )
#     T = 100
#     N = 16
#     y, theta, rho, u, p = initialize_data(
#         model="diffraction", T=T, N=N, mu=1.0, sigma=1.0, lambda_=0.1
#     )
#     T = 500
#     N = 10
#     y, theta, rho, u, p = initialize_data(
#         model="glmm", T=T, N=N, p=8, mu1=0.0, mu2=3.0, lambda1=10.0, lambda2=3.0, w1=0.5, n_i=6
#     )
    
#     to_pickle(y,'y_glmm.pkl')
    
#     # # 生成轨迹,步长请小于0.01
#     # dic1, ddic1, t = get_trajectory(
#     #     t_span=[0, 0.1],
#     #     timescale=100,
#     #     dynamics_fn=compute_time_derivative,
#     #     model="glmm",
#     #     y_data=y,
#     #     input_theta_dim=theta.shape[0],
#     #     aux_dim=u.shape[0],
#     #     sigma_x=0.1
#     # )
    
   
    
#     # 加载数据
#     data = get_dataset(
#     num_samples=200,
#     test_split=0,
#     input_theta_dim=theta.shape[0],
#     aux_dim=u.shape[0],
#     model='glmm',
#     y_data=y,
#     save_path='glmm.pkl',
#     should_load=False,
#     load_path='glmm.pkl',
#     t_span=[0, 0.1],
#     timescale=50,
#     dynamics_fn=compute_time_derivative,
#     sigma_x = 0.1
# )
    
