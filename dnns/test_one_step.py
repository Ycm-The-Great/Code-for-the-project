#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 03:20:39 2025

@author: yanchuanmiao
"""
import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import namedtuple


# 模拟参数类
class Args:
    input_dim = 4  # q + p 的维度总和

args = Args()

# 假的 integrate_model（只返回平移后的 y0）
def fake_integrate_model(hnn_model, t_span, y0, steps, args, **kwargs):
    y0 = tf.cast(y0, tf.float32)
    # y0 = tf.reshape(y0, [-1])  # flatten
    return tf.stack([y0 + 0.1 * i for i in range(steps + 1)], axis=0)  

# 假的 HNN 模型（不使用）
hnn_model = None

# 假的哈密顿量函数（平方和）
def fake_hamiltonian(x):
    return tf.reduce_sum(tf.square(x), axis=0)
HNNKernelResults = namedtuple("HNNKernelResults", [
    "accept",          # [chains] 是否接受
    "log_accept_ratio",# [chains] log(α)
    "grad_evals"       # 标量：累积梯度调用次数
])


class HNN_HMC_Kernel(tfp.mcmc.TransitionKernel):
    def __init__(self, hnn_model, integrate_model, functions, step_size, args, **kwargs):
        self.hnn_model = hnn_model
        self.integrate_model = integrate_model
        self.functions = functions
        self.step_size = step_size
        self.args = args
        self.kwargs = kwargs

    @property
    def is_calibrated(self):
        return False

    def one_step(self, current_state, previous_kernel_results, seed=None):
        q, p = current_state
        input_dim = self.args.input_dim
        y0 = tf.concat([q, p], axis=-1)

        L = self.kwargs.get("L", 10)
        steps = 2
        t_span = [0, L]
        self.kwargs.update({'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-8})

        hnn_ivp = self.integrate_model(self.hnn_model, t_span, y0, steps - 1, self.args, **self.kwargs)
        chains = q.shape[0]
        yhamil = hnn_ivp[-1, :]
        yhamil1 = tf.reshape(yhamil, [args.input_dim, -1])
        y0 = tf.reshape(y0, [args.input_dim, -1])
        
        H_star = self.functions(yhamil1)
        H_prev = self.functions(y0)
        log_accept_ratio = tf.clip_by_value(H_prev - H_star, -50.0, 50.0)
        alpha = tf.math.exp(log_accept_ratio)
        
        alpha = tf.minimum(1, alpha)
        tf_seed = tfp.random.sanitize_seed(seed)
        rand_val = tf.random.stateless_uniform(shape=[], seed=tf_seed, dtype=tf.float32)
        accept = alpha > rand_val.numpy()
        accept_transposed = tf.transpose(accept)
        grad_evals_this_step = steps
        total_grad_evals = previous_kernel_results.grad_evals + grad_evals_this_step

        next_q = tf.where(accept_transposed, tf.convert_to_tensor(yhamil[:,:input_dim // 2], dtype=tf.float32), q)
        next_p = tf.random.normal(shape=[chains,input_dim // 2], mean=0.0, stddev=1.0, dtype=tf.float32)
        assert next_q.shape == (chains, input_dim // 2), f"❌ next_q shape mismatch: expected {(chains, input_dim // 2)}, got {next_q.shape}"
        assert next_p.shape == (chains, input_dim // 2), f"❌ next_p shape mismatch: expected {(chains, input_dim // 2)}, got {next_p.shape}"
        assert accept.shape == (chains,), f"accept shape mismatch: {accept.shape}"

        return [next_q, next_p], HNNKernelResults(
            accept=accept,
            log_accept_ratio=log_accept_ratio,
            grad_evals=total_grad_evals
        )

    def bootstrap_results(self, init_state):
        chains = init_state[0].shape[0]
        return HNNKernelResults(
            accept=tf.zeros([chains], dtype=tf.bool),
            log_accept_ratio=tf.zeros([chains], dtype=tf.float32),
            grad_evals=tf.zeros([], dtype=tf.int32)
        )
    
class TestHNNHMCKernel(unittest.TestCase):
    def test_one_step(self):
        # 生成初始状态 q, p
        num_chains = 2
        q0 = tf.random.normal([num_chains, args.input_dim // 2])
        p0 = tf.random.normal([num_chains, args.input_dim // 2])
        initial_state = [q0, p0]
        
        # 构造 kernel
        kernel = HNN_HMC_Kernel(
            hnn_model=hnn_model,
            integrate_model=fake_integrate_model,
            functions=fake_hamiltonian,
            step_size=0.1,
            args=args,
            L=1  # 设定时间跨度 L
        )
        
        # 获取初始 kernel results
        kernel_results = kernel.bootstrap_results(initial_state)
        
        # 设置种子
        seed = tf.constant([123, 456], dtype=tf.int32)
        
        # 调用 one_step
        [next_q, next_p], next_results = kernel.one_step(initial_state, kernel_results, seed=seed)
        
        # ✅ 断言 shape 正确
        self.assertEqual(next_q.shape, (num_chains, args.input_dim // 2))
        self.assertEqual(next_p.shape, (num_chains, args.input_dim // 2))
 
        # ✅ 断言 accept 是 bool 类型
        self.assertTrue(next_results.accept.dtype == tf.bool)
        self.assertEqual(next_results.accept.shape, (num_chains,))
 
        # ✅ 断言 log_accept_ratio 在 [-50, 50] 范围内
        self.assertTrue(tf.reduce_all(next_results.log_accept_ratio >= -50.0))
        self.assertTrue(tf.reduce_all(next_results.log_accept_ratio <= 50.0))
 
        # ✅ 断言 grad_evals 等于 2
        self.assertEqual(next_results.grad_evals.numpy(), 2)
   
        # ✅ 打印结果
        print("✅ one_step executed successfully")
        print("Initial q:\n", q0.numpy())
        print("Next q:\n", next_q.numpy())
        print("Next p:\n", next_p.numpy())
        print("Accept:\n", next_results.accept.numpy())
        print("Log acceptance ratio:\n", next_results.log_accept_ratio.numpy())
        print("Grad evals:\n", next_results.grad_evals.numpy())
       
        
        
if __name__ == '__main__':
    unittest.main()
    