#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 05:47:20 2025

@author: yanchuanmiao
"""

import unittest
import tensorflow as tf
import numpy as np

# 导入你写的 leapfrog_tf
# from your_module import leapfrog_tf  # 如果你有模块文件

# 被测试函数
def leapfrog_tf(dydt, tspan, y0, n, dim):
    t0, tstop = tspan
    dt = (tstop - t0) / tf.cast(n, tf.float32)

    t = tf.TensorArray(dtype=tf.float32, size=n + 1, dynamic_size=False, clear_after_read=False)
    y = tf.TensorArray(dtype=tf.float32, size=n + 1, dynamic_size=False, clear_after_read=False)

    t = t.write(0, t0)
    y = y.write(0, y0)
    anew = dydt(t.read(0), y.read(0))

    def loop_body(i, t, y, anew):
        t_next = t.read(i - 1) + dt
        aold = anew
        y_prev = y.read(i - 1)
        y_next = tf.identity(y_prev)

        for j in range(0, dim // 2):
            y_next = tf.tensor_scatter_nd_update(
                y_next, [[j]], [y_prev[j] + dt * (y_prev[j + dim // 2] + 0.5 * dt * aold[j + dim // 2])]
            )

        anew = dydt(t_next, y_next)

        for j in range(0, dim // 2):
            y_next = tf.tensor_scatter_nd_update(
                y_next, [[j + dim // 2]], [y_prev[j + dim // 2] + 0.5 * dt * (aold[j + dim // 2] + anew[j + dim // 2])]
            )

        t = t.write(i, t_next)
        y = y.write(i, y_next)
        return i + 1, t, y, anew

    _, t, y, _ = tf.while_loop(
        lambda i, t, y, anew: i < n + 1,
        loop_body,
        [1, t, y, anew]
    )

    return y.stack()

# 单元测试类
class TestLeapfrogTF(unittest.TestCase):
    def test_leapfrog_simple_oscillator(self):
        # 定义测试系统：dq/dt = p, dp/dt = -q
        def dydt(t, y):
            q, p = y[0], y[1]
            return tf.convert_to_tensor([p, -q], dtype=tf.float32)

        y0 = tf.constant([1.0, 0.0], dtype=tf.float32)  # 初始位置 q=1, p=0
        tspan = [0.0, 10.0]
        n = 1000
        dim = 2

        result = leapfrog_tf(dydt, tspan, y0, n, dim)

        # ✅ 断言 shape 正确
        self.assertEqual(result.shape, (n + 1, dim))

        # ✅ 每一个输出都应是 Tensor 类型
        self.assertIsInstance(result, tf.Tensor)

        # ✅ 能量守恒测试：H = 0.5 * (q^2 + p^2) 应该接近常数
        q_vals = result[:, 0]
        p_vals = result[:, 1]
        energy = 0.5 * (q_vals**2 + p_vals**2)

        energy_diff = tf.reduce_max(tf.abs(energy - energy[0]))
        self.assertLess(energy_diff.numpy(), 1e-2)  # 能量变化不能太大

        print("✅ leapfrog_tf test passed")
        print("Initial Energy:", energy[0].numpy())
        print("Maximum Energy Difference:", energy_diff.numpy())

# 运行测试
if __name__ == '__main__':
    unittest.main()