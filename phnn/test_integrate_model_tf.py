#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 01:51:17 2025

@author: yanchuanmiao
"""

import unittest
import numpy as np
import tensorflow as tf
from phnn_hmc import integrate_model_tf
# 模拟一个假的 dynamical model
class MockPHNNModel:
    def hamiltonian_dynamics(self, theta, rho, u, p):
        # 返回导数，维度与输入一致
        return theta, rho, u, p

# === mock 模型和 leapfrog ===
class MockPHNNModel:
    def hamiltonian_dynamics(self, theta, rho, u, p):
        # 返回导数，形状与输入相同
        return theta, rho, u, p

def leapfrog_tf(fun, t_span, y0, n, input_theta_dim, h, model, aux_dim):
    # 这里直接模拟一个 [dim, n+1] 的假输出
    dim = y0.shape[0]
    steps = n + 1
    y_series = tf.stack([y0 + tf.cast(i, tf.float32) * 0.01 for i in range(steps)], axis=1)
    return y_series  # shape: [dim, steps]

# === 单元测试 ===
class TestIntegrateModelTF(unittest.TestCase):
    def test_integrate_model_tf(self):
        input_theta_dim = 2
        aux_dim = 1
        total_dim = 2 * input_theta_dim + 2 * aux_dim  # = 6
        y0 = tf.constant(np.random.randn(total_dim), dtype=tf.float32)  # shape: [6]
        t_span = [0.0, 1.0]
        h = 0.1
        n = 10  # steps = 11
        model = "mock"
        assert len(y0.shape) == 1, "Expected y0 to be 1D tensor (single chain), but got shape {}".format(y0.shape)
        result = integrate_model_tf(
            model1=MockPHNNModel(),
            t_span=t_span,
            y0=y0,
            n=n,
            input_theta_dim=input_theta_dim,
            h=h,
            model=model,
            aux_dim=aux_dim
        )

        # === 检查输出形状 ===
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], total_dim)
        self.assertEqual(result.shape[1], n + 1)
        print("✅ integrate_model_tf test passed")
if __name__ == "__main__":
    unittest.main()