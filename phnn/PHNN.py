#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:21:43 2025

@author: yanchuanmiao
"""
import tensorflow as tf
import numpy as np
from utils import lfrog
from Args import Args


# 定义Hamiltonian神经网络
class PHNN(tf.keras.Model):
    def __init__(self, dim_theta, dim_u,hidden_dim = 64):
        super(PHNN, self).__init__()
        self.dim_theta = dim_theta
        self.dim_u = dim_u
        self.hidden_dim = hidden_dim
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1,use_bias= False)  # 输出一个标量Hamiltonian值
        ])
        # 这里因为target梯度本身，所以偏置项没有梯度很正常，所以就不用梯度就好了

    def call(self, theta, phi, u, p):
        # 将所有变量拼接输入网络
        inputs = tf.concat([theta, phi, u, p], axis=-1)
        return self.net(inputs)

    # 定义动力系统的微分方程
    def hamiltonian_dynamics(self,theta, phi, u, p):
        """
        根据Equation 12定义动力系统
        """
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
           tape.watch([theta, phi, u, p])
           H = self.call(theta, phi, u, p)
        dH_dtheta = tape.gradient(H, theta)
        dH_dphi = tape.gradient(H, phi)
        dH_du = tape.gradient(H, u)
        dH_dp = tape.gradient(H, p)
        del tape
        return [dH_dphi, -dH_dtheta, dH_dp, -dH_du]

