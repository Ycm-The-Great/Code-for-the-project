#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 01:43:42 2025

@author: yanchuanmiao
"""

import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# 假设你的 kernel 类已经定义好
from phnn_hmc import PHNN_HMC_Kernel  # 替换为你的实际文件名

tfd = tfp.distributions
class MockPHNNModel:
    def hamiltonian_dynamics(self, theta, rho, u, p):
        return theta, rho, u, p

# mock 函数：模拟哈密顿量函数
def mock_hamiltonian(theta, rho, u, p, y_data, model):
    return tf.reduce_sum(theta**2 + rho**2 + u**2 + p**2)

# mock 积分器：返回 y0 + 一点扰动
def mock_integrator(phnn_model, t_span, y0, n, input_theta_dim, h, model, aux_dim):
    steps = n + 1
    y0 = tf.expand_dims(y0, axis=1)  # shape [dim, 1]
    noise = tf.random.normal(shape=(tf.shape(y0)[0], steps), stddev=0.01)
    return tf.concat([y0, y0 + noise[:, 1:]], axis=1)

# 模拟 args
class Args:
    def __init__(self):
        self.input_theta_dim = 2
        self.aux_dim = 1
        self.t_span = [0.0, 1.0]
        self.timescale = 10
        self.model = "mock_model"

class TestPHNNKernel(unittest.TestCase):

    def setUp(self):
        self.args = Args()
        self.y_data = tf.constant([[1.0, 2.0]], dtype=tf.float32)  # 模拟观测数据
        self.kernel = PHNN_HMC_Kernel(
            phnn_model=MockPHNNModel(),  
            integrate_model=mock_integrator,
            functions=mock_hamiltonian,
            y_data=self.y_data,
            args=self.args
        )

    def test_one_step(self):
        # 输入状态：num_chains = 4，theta_dim = 2
        num_chains = 4
        theta_dim = self.args.input_theta_dim
        initial_state = tf.random.normal(shape=(num_chains, theta_dim))

        new_state, kernel_results = self.kernel.one_step(initial_state, previous_kernel_results=None, seed=(42, 1))

        # === 断言部分 ===
        self.assertEqual(new_state.shape, initial_state.shape)
        self.assertIn("accept_prob", kernel_results)
        self.assertIn("accepted", kernel_results)
        self.assertIn("theta_star", kernel_results)
        self.assertEqual(kernel_results["theta_star"].shape, initial_state.shape)
        self.assertEqual(kernel_results["H_prev"].shape, (num_chains,))
        self.assertEqual(kernel_results["H_star"].shape, (num_chains,))
        self.assertTrue(tf.reduce_all(kernel_results["accept_prob"] <= 1.01))
        self.assertTrue(tf.reduce_all(kernel_results["accept_prob"] >= 0.0))

        print("one_step test passed。")
        print("Acception Raitio:", kernel_results["accept_prob"].numpy())

if __name__ == '__main__':
    unittest.main()