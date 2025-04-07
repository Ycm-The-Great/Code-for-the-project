#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 02:07:03 2025

@author: yanchuanmiao
"""



import unittest
import tensorflow as tf
import numpy as np

# === Mock 依赖函数 ===
def compute_weights_and_x_gaussian(y, mu_theta, u, sigma_x, sigma_y):
    # 返回模拟权重 (batch_size, num_particles)
    weights = tf.ones((tf.shape(y)[0], tf.shape(u)[0]))
    return weights, None

def compute_weights_and_x_diffraction(y, mu, u, sigma, lambda_):
    weights = tf.ones((tf.shape(y)[0], tf.shape(u)[0]))
    return weights, None

def compute_weights_and_x_glmm(y, theta, u, sigma_x=0.1):
    weights = tf.ones((tf.shape(y)[0], tf.shape(u)[0]))
    return weights, None

# === 被测试函数 ===
def compute_hamiltonian(theta, rho, u, p, y, model, **kwargs):
    if model == 'gaussian':
        mu_theta = kwargs.get("mu_theta", 0.0)
        sigma_x = tf.exp(theta[0])
        sigma_y = tf.exp(theta[1])
        sigma_theta = kwargs.get('sigma_theta', 10.0)
        log_prior = -0.5 * ((mu_theta**2) / sigma_theta**2 + tf.math.log(2 * np.pi * sigma_theta**2))
        weights, _ = compute_weights_and_x_gaussian(y, mu_theta, u, sigma_x, sigma_y)
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1)))

    elif model == 'diffraction':
        mu = theta[0]
        log_sigma = theta[1]
        log_lambda = theta[2]
        sigma = tf.exp(log_sigma)
        lambda_ = tf.exp(log_lambda)
        log_prior = -0.5 * (
            mu**2 / 100.0 + log_sigma**2 / 100.0 + log_lambda**2 / 100.0
        )
        weights, _ = compute_weights_and_x_diffraction(y, mu, u, sigma, lambda_)
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1)))

    elif model == 'glmm':
        d = tf.shape(theta)[0]
        beta = theta[:-5]
        mu1, mu2 = theta[-5], theta[-4]
        log_lambda1, log_lambda2 = theta[-3], theta[-2]
        logit_w1 = theta[-1]
        lambda1, lambda2 = tf.exp(log_lambda1), tf.exp(log_lambda2)
        w1 = tf.sigmoid(logit_w1)
        log_prior = -0.5 * (
            tf.reduce_sum(beta**2) / 100.0 +
            mu1**2 / 100.0 +
            mu2**2 / 100.0 +
            log_lambda1**2 / 100.0 +
            log_lambda2**2 / 100.0 +
            logit_w1**2 / 100.0
        )
        weights, _ = compute_weights_and_x_glmm(y, theta, u, sigma_x=kwargs.get('sigma_x', 0.1))
        log_likelihood = tf.reduce_sum(tf.math.log(tf.reduce_mean(weights, axis=1) + 0.001))
    else:
        raise ValueError("Invalid model. Choose from 'gaussian', 'diffraction', or 'glmm'.")

    kinetic_energy = 0.5 * (tf.reduce_sum(rho**2) + tf.reduce_sum(u**2) + tf.reduce_sum(p**2))
    H = -log_prior - log_likelihood + kinetic_energy
    return H

# === 单元测试类 ===
class TestHamiltonianModels(unittest.TestCase):

    def setUp(self):
        # 通用输入
        self.rho = tf.constant([1.0, 1.0], dtype=tf.float32)
        self.u = tf.constant([1.0, 1.0], dtype=tf.float32)
        self.p = tf.constant([1.0], dtype=tf.float32)
        self.y = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)  # batch_size = 3

    def test_all_models(self):
        # === 1. Gaussian 模型 ===
        theta_gaussian = tf.math.log(tf.constant([1.0, 1.0], dtype=tf.float32))
        H_gaussian = compute_hamiltonian(
            theta_gaussian, self.rho, self.u, self.p, self.y,
            model='gaussian', mu_theta=0.0, sigma_theta=10.0
        )

        self.assertIsInstance(H_gaussian, tf.Tensor)
        self.assertEqual(tf.rank(H_gaussian), 0)

        # 手动计算期望值
        log_prior = -0.5 * np.log(2 * np.pi * (10.0**2))  # ≈ -3.221
        kinetic = 0.5 * (2 + 2 + 1)  # = 2.5
        expected_H_gaussian = -log_prior + kinetic  # ≈ 5.721

        self.assertAlmostEqual(H_gaussian.numpy(), expected_H_gaussian, places=1)
        print(f"[PASS]  Gaussian Hamiltonian = {H_gaussian.numpy():.3f} ≈ {expected_H_gaussian:.3f}")

        # === 2. Diffraction 模型 ===
        theta_diff = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)
        H_diff = compute_hamiltonian(
            theta_diff, self.rho, self.u, self.p, self.y,
            model='diffraction', sigma_x=0.1
        )

        self.assertIsInstance(H_diff, tf.Tensor)
        self.assertEqual(tf.rank(H_diff), 0)
        expected_H_diff = 2.5
        self.assertAlmostEqual(H_diff.numpy(), expected_H_diff, places=1)

        print(f"[PASS]  Diffraction Hamiltonian = {H_diff.numpy():.3f} ≈ {expected_H_diff:.3f}")

        # === 3. GLMM 模型 ===
        theta_glmm = tf.constant([0.1] * 8, dtype=tf.float32)
        H_glmm = compute_hamiltonian(
            theta_glmm, self.rho, self.u, self.p, self.y,
            model='glmm', sigma_x=0.1
        )

        self.assertIsInstance(H_glmm, tf.Tensor)
        self.assertEqual(tf.rank(H_glmm), 0)
        expected_H_glmm = 2.5
        self.assertAlmostEqual(H_glmm.numpy(), expected_H_glmm, places=1)
        print(f"[PASS] GLMM Hamiltonian = {H_glmm.numpy():.3f}  ≈ {expected_H_glmm:.3f} ")



if __name__ == '__main__':
    unittest.main()

