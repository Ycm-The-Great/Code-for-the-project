#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 05:43:17 2025

@author: yanchuanmiao
"""

import unittest
import tensorflow as tf
import numpy as np

# 假参数类
class Args:
    input_dim = 4

args = Args()

# 假模型：dx/dt = x
class DummyModel:
    def time_derivative(self, x):
        return x  # 简单返回自身

# 假 leapfrog 方法（模拟）
def leapfrog_tf(fun, t_span, y0, n, dim):
    """
    模拟 Leapfrog：每一步加 dt * fun(current)
    """
    dt = (t_span[1] - t_span[0]) / n
    y_current = y0
    ys = [y0]
    for _ in range(n):
        dy = fun(0, y_current)
        y_current = y_current + dt * dy
        ys.append(y_current)
    return tf.stack(ys, axis=0)  # shape: [n+1, input_dim]

# 被测试函数
def integrate_model_tf(model, t_span, y0, n, args, **kwargs):
    def fun(t, np_x):
        x = tf.convert_to_tensor(np_x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            dx = model.time_derivative(x)
        dx = tf.reshape(dx, [-1])
        return dx

    return leapfrog_tf(fun, t_span, y0, n, args.input_dim)

# 单元测试类
class TestIntegrateModelTF(unittest.TestCase):
    def test_integrate_model_tf(self):
        model = DummyModel()
        t_span = [0.0, 1.0]
        y0 = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        n = 5  # 步数

        result = integrate_model_tf(model, t_span, y0, n, args)

        # ✅ 检查输出 shape
        self.assertEqual(result.shape, (n + 1, args.input_dim))

        # ✅ 检查类型
        self.assertIsInstance(result, tf.Tensor)

        # ✅ 检查是否递增（因为 dx/dt = x）
        # 所以每一维都应该是递增的（指数增长趋势）
        diffs = result[1:, 0] - result[:-1, 0]
        self.assertTrue(tf.reduce_all(diffs > 0))

        print("✅ integrate_model_tf  test passed")
        print("Output Results：\n", result.numpy())

# 运行测试
if __name__ == '__main__':
    unittest.main()