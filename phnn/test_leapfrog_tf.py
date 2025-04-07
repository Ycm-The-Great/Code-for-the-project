#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 02:01:19 2025

@author: yanchuanmiao
"""

import unittest
import tensorflow as tf
import numpy as np

from phnn_hmc import leapfrog_tf

# === 模拟动力学函数 ===
def mock_dynamics_fn(theta, rho, u, p):
    # 简单返回常数导数
    return (
        tf.ones_like(theta),
        tf.ones_like(rho),
        tf.ones_like(u),
        tf.ones_like(p)
    )

# === 单元测试 ===
class TestLeapfrogTF(unittest.TestCase):
    def test_leapfrog_two_steps_with_expected_values(self):
        input_theta_dim = 2
        aux_dim = 1
        total_dim = 2 * input_theta_dim + 2 * aux_dim  # 6

        # 初始值全为 0
        y0 = tf.constant(np.zeros(total_dim), dtype=tf.float32)
        n_steps = 2
        h = 0.1
        t_span = [0.0, 0.2]

        result = leapfrog_tf(
            mock_dynamics_fn,
            t_span,
            y0,
            n_steps,
            input_theta_dim,
            h,
            model="mock",
            aux_dim=aux_dim
        )

        # 构造期望值 [6, 3]
        expected = tf.constant([
            [0.0, 0.1, 0.2],  # theta1
            [0.0, 0.1, 0.2],  # theta2
            [0.0, 0.1, 0.2],  # rho1
            [0.0, 0.1, 0.2],  # rho2
            [0.0, 0.1, 0.2],  # u
            [0.0, 0.1, 0.2],  # p
        ], dtype=tf.float32)

        # === 验证结果 ===
        tf.debugging.assert_near(result, expected, atol=1e-6)

        # === 验证类型和形状 ===
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape[0], total_dim)
        self.assertEqual(result.shape[1], n_steps + 1)

        print("Leapfrog test passed")


if __name__ == '__main__':
    unittest.main()