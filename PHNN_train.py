#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 05:25:30 2025

@author: yanchuanmiao
"""
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from utils import to_pickle, from_pickle, L2_loss
from tqdm import tqdm
from get_data import * 
from Args import Args
from PHNN import PHNN
import os
args = Args()

def train(args):
    # 设置随机种子
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    dim_theta = args.input_theta_dim
    dim_u = args.aux_dim
    # 初始化 PHNN 模型
    model = PHNN(dim_theta=dim_theta, dim_u=dim_u, hidden_dim=args.hidden_dim)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learn_rate)
    input_dim = 2 * dim_theta + 2 * dim_u
    if args.should_load:
        data = get_dataset(
        should_load=True,
        load_path= args.load_path,
        )
        y = from_pickle(args.y_path)
    else:
        # 确保所有参数都从 args 获取
        T = args.T
        N = args.N
    
        if args.model == "gaussian":
            y, theta, rho, u, p = initialize_data(
                model="gaussian",
                T=T,
                N=N,
                mu_theta=args.mu_theta,
                sigma_theta=args.sigma_theta,
                sigma_x=args.sigma_x,
                sigma_y=args.sigma_y
            )
        elif args.model == "diffraction":
            y, theta, rho, u, p = initialize_data(
                model="diffraction",
                T=T,
                N=N,
                mu=args.mu,
                sigma=args.sigma,
                lambda_=args.lambda_
            )
        elif args.model == "glmm":
            y, theta, rho, u, p = initialize_data(
                model="glmm",
                T=T,
                N=N,
                p=args.p,
                mu1=args.mu1,
                mu2=args.mu2,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                w1=args.w1,
                n_i=args.n_i
            )
            data = get_dataset(
                        num_samples=args.num_samples,           # 从 args 获取样本数量
                        test_split=args.test_split,             # 从 args 获取测试集比例
                        input_theta_dim=args.dim_theta,         # 从 args 获取 theta 的维度
                        aux_dim=args.aux_dim,                   # 从 args 获取辅助变量的维度
                        model=args.model,                       # 从 args 获取模型名称
                        y_data=args.y_data,                     # 从 args 获取初始数据 y
                        save_path=args.save_path,               # 从 args 获取保存路径
                        should_load=args.should_load,           # 从 args 获取是否加载已有数据
                        load_path=args.load_path,               # 从 args 获取加载路径
                        t_span=args.t_span,                     # 从 args 获取时间范围
                        timescale=args.timescale,               # 从 args 获取时间刻度
                        dynamics_fn=compute_time_derivative,    # 动力学函数保持不变
                        lambda1=args.lambda1,
                        lambda2=args.lambda2,
                        w1=args.w1,
                        n_i=args.n_i,
                        mu=args.mu,
                        sigma=args.sigma,
                        lambda_=args.lambda_ ,
                        mu_theta=args.mu_theta,
                        sigma_theta=args.sigma_theta,
                        sigma_x=args.sigma_x,
                        sigma_y=args.sigma_y# 从 args 获取 sigma_x
                            )
        
     
    # 加载数据集
   
    # x = tf.convert_to_tensor(data['train']['coords'], dtype=tf.float32)  # 训练输入数据
    test_x = tf.convert_to_tensor(data['test']['coords'], dtype=tf.float32)  # 测试输入数据
    # dxdt = tf.convert_to_tensor(data['train']['dcoords'], dtype=tf.float32)  # 训练数据的时间导数
    test_dxdt = tf.convert_to_tensor(data['test']['dcoords'], dtype=tf.float32)  # 测试数据的时间导数
    theta_dim = args.input_theta_dim
    aux_dim = args.aux_dim
    coords = data['train']['coords']
    dcoords = data['train']['dcoords']
    
    theta = tf.convert_to_tensor(coords[:, :theta_dim], dtype=tf.float32)  # 第一部分: theta
    phi = tf.convert_to_tensor(coords[:, theta_dim:2 * theta_dim], dtype=tf.float32)  # 第二部分: phi
    u = tf.convert_to_tensor(coords[:, 2 * theta_dim:2 * theta_dim + aux_dim], dtype=tf.float32)  # 第三部分: u
    p = tf.convert_to_tensor(coords[:, -aux_dim:], dtype=tf.float32)  # 第四部分: p
    dtheta_dt = tf.convert_to_tensor(dcoords[:, :theta_dim], dtype=tf.float32)  # 第一部分: theta
    dphi_dt = tf.convert_to_tensor(dcoords[:, theta_dim:2 * theta_dim], dtype=tf.float32)  # 第二部分: phi
    du_dt = tf.convert_to_tensor(dcoords[:, 2 * theta_dim:2 * theta_dim + aux_dim], dtype=tf.float32)  # 第三部分: u
    dp_dt = tf.convert_to_tensor(dcoords[:, -aux_dim:], dtype=tf.float32)  # 第四部分: p
    

    # 训练循环
    stats = {'train_loss': [], 'test_loss': []}
    for step in tqdm(range(args.num_steps + 1)):
        # 随机采样批次
        idxs = np.random.choice(theta.shape[0], args.batch_size, replace=False)
        theta_batch = tf.gather(theta, idxs)
        phi_batch = tf.gather(phi, idxs)
        u_batch = tf.gather(u, idxs)
        p_batch = tf.gather(p, idxs)
        dtheta_batch = tf.gather(dtheta_dt, idxs)
        dphi_batch = tf.gather(dphi_dt, idxs)
        du_batch = tf.gather(du_dt, idxs)
        dp_batch = tf.gather(dp_dt, idxs)
        
        theta_batch = tf.Variable(theta_batch)
        phi_batch = tf.Variable(phi_batch)
        u_batch = tf.Variable(u_batch)
        p_batch = tf.Variable(p_batch)
        # 计算梯度与反向传播
        with tf.GradientTape() as tape:
            # 前向传递计算H
            # 预测导数
            dphi_pred, dtheta_pred, dp_pred, du_pred = model.hamiltonian_dynamics(
               theta_batch, phi_batch, u_batch, p_batch)
            # # 计算导数
            # with tf.GradientTape(persistent=True, watch_accessed_variables=True) as h_tape:
            #     h_tape.watch([theta_batch, phi_batch, u_batch, p_batch])
            #     H_internal = model(theta_batch, phi_batch, u_batch, p_batch)
            # dH_dtheta = h_tape.gradient(H_internal, theta_batch)
            # dH_dphi = h_tape.gradient(H_internal, phi_batch)
            # dH_du = h_tape.gradient(H_internal, u_batch)
            # dH_dp = h_tape.gradient(H_internal, p_batch)
            # del h_tape
        
            # # 计算预测的导数和损失
            # dphi_pred = dH_dphi
            # dtheta_pred = -dH_dtheta
            # dp_pred = dH_dp
            # du_pred = -dH_du
        
            loss_theta = L2_loss(dtheta_batch, dtheta_pred)
            loss_phi = L2_loss(dphi_batch, dphi_pred)
            loss_u = L2_loss(du_batch, du_pred)
            loss_p = L2_loss(dp_batch, dp_pred)
            total_loss = loss_theta + loss_phi + loss_u + loss_p

        # 更新参数
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 这里因为target梯度本身，所以偏置项没有梯度很正常
        # 记录与打印日志（类似原有逻辑）
        stats['train_loss'].append(total_loss.numpy())
        if step % args.print_every == 0:
            print(f"step {step}, loss {total_loss.numpy():.4e}")

    return model, stats

if __name__ == "__main__":
    args = Args().parse_args()  # 获取命令行参数
    model, stats = train(args)  # 调用训练函数
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    # 保存模型
    path = os.path.join(THIS_DIR, f"{args.model}.h5")  # 模型保存路径
    model.save_weights(path)  # 保存模型的权重
