#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:53:07 2024

@author: yanchuanmiao
"""

from numbers import Real
import tensorflow as tf
import numpy as np
import os
from nn_models import MLP  # 假设 MLP 是一个自定义的 TensorFlow 模型
from hnn import HNN       # 假设 HNN 是一个自定义的 TensorFlow 类
from data import get_dataset
from utils import L2_loss, to_pickle
from get_args import get_args


def train(args):
    """
    训练函数：实现基于 TensorFlow 的训练逻辑，用于训练 HNN 模型或基线神经网络。
    args: 命令行参数，包含训练配置和超参数。
    """
    # 设置随机种子以确保结果的可重复性
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化模型和优化器
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)  # 定义神经网络
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=False)  # 初始化 HNN 模型
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learn_rate)

    # 准备数据集
    data = get_dataset(seed=args.seed)
    x = tf.convert_to_tensor(data['coords'], dtype=tf.float32)  # 训练输入数据
    test_x = tf.convert_to_tensor(data['test_coords'], dtype=tf.float32)  # 测试输入数据
    dxdt = tf.convert_to_tensor(data['dcoords'], dtype=tf.float32)  # 训练数据的时间导数
    test_dxdt = tf.convert_to_tensor(data['test_dcoords'], dtype=tf.float32)  # 测试数据的时间导数

    # 初始化字典来存储训练和测试损失
    stats = {'train_loss': [], 'test_loss': []}

    # 训练循环
    for step in range(args.total_steps + 1):
        # 随机采样一个批次数据
        idxs = np.random.choice(x.shape[0], size=args.batch_size, replace=False)
        x_batch = tf.gather(x, idxs)
        dxdt_batch = tf.gather(dxdt, idxs)

        # 计算损失并进行反向传播
        with tf.GradientTape() as tape:
            dxdt_hat = model.time_derivative(x_batch)  # 模型预测的时间导数
            loss = L2_loss(dxdt_batch, dxdt_hat)  # 计算训练损失（均方误差）

        grads = tape.gradient(loss, model.trainable_variables)  # 计算梯度
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 应用梯度更新模型参数

        # 测试数据上的损失
        test_dxdt_hat = model.time_derivative(test_x)  # 测试数据的预测时间导数
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)  # 计算测试损失

        # 记录损失
        stats['train_loss'].append(loss.numpy())
        stats['test_loss'].append(test_loss.numpy())

        # 打印训练进度
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.numpy(), test_loss.numpy()))

    # 计算最终的损失分布
    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    if args.test_fraction == 0:
        test_dxdt_hat = 0
        test_dist = 1
        print("No test set")   
        print('Final train loss {:.4e} +/- {:.4e}'.format(tf.reduce_mean(train_dist).numpy(),
                tf.math.reduce_std(train_dist).numpy() / np.sqrt(train_dist.shape[0]) ))
    else: 
        print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(tf.reduce_mean(train_dist).numpy(),
                  tf.math.reduce_std(train_dist).numpy() / np.sqrt(train_dist.shape[0]),
                  tf.reduce_mean(test_dist).numpy(),
                  tf.math.reduce_std(test_dist).numpy() / np.sqrt(test_dist.shape[0])))
    return model, stats


if __name__ == "__main__":
    args = get_args()  # 获取命令行参数
    model, stats = train(args)  # 调用训练函数

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)  # 如果保存目录不存在，则创建
    path = os.path.join(args.save_dir, f"{args.dist_name}.h5")  # 模型保存路径
    model.save_weights(path)  # 保存模型的权重