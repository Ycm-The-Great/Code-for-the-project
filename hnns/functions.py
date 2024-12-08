#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:33:31 2024

@author: yanchuanmiao
"""

import numpy as np
import tensorflow as tf
import argparse
from get_args import get_args

args = get_args()

def functions(coords):
    # 将 coords 转为 TensorFlow 张量
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    
    # ******** 1D Gaussian Mixture ********
    if (args.dist_name == '1D_Gauss_mix'):
        q, p = tf.split(coords, 2, axis=0)
        mu1 = 1.0
        mu2 = -1.0
        sigma = 0.35
        term1 = -tf.math.log(0.5 * (tf.math.exp(-(q - mu1) ** 2 / (2 * sigma ** 2))) + 
                             0.5 * (tf.math.exp(-(q - mu2) ** 2 / (2 * sigma ** 2))))
        H = term1 + p ** 2 / 2  # Normal PDF

    # ******** 2D Gaussian Four Mixtures ********
    elif(args.dist_name == '2D_Gauss_mix'):
        q1, q2, p1, p2 = tf.split(coords, 4, axis=0)
        sigma_inv = tf.constant([[1., 0.], [0., 1.]], dtype=tf.float32)
        term1 = 0.

        for mu in [[3., 0.], [-3., 0.], [0., 3.], [0., -3.]]:
            mu = tf.constant(mu, dtype=tf.float32)
            y = [q1 - mu[0], q2 - mu[1]]
            tmp1 = [sigma_inv[0, 0] * y[0] + sigma_inv[0, 1] * y[1],
                    sigma_inv[1, 0] * y[0] + sigma_inv[1, 1] * y[1]]
            tmp1 = tf.reshape(tmp1, (2,))
            term1 += 0.25 * tf.math.exp(-y[0] * tmp1[0] - y[1] * tmp1[1])

        term1 = -tf.math.log(term1)
        term2 = p1 ** 2 / 2 + p2 ** 2 / 2
        H = term1 + term2

    # ******** 5D Ill-Conditioned Gaussian ********
    elif(args.dist_name == '5D_illconditioned_Gaussian'):
        dic1 = tf.split(coords, args.input_dim, axis=0)
        var1 = tf.constant([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02], dtype=tf.float32)
        term1 = dic1[0] ** 2 / (2 * var1[0])
        for ii in range(1, 5):
            term1 += dic1[ii] ** 2 / (2 * var1[ii])
        term2 = dic1[5] ** 2 / 2
        for ii in range(6, 10):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    # ******** nD Funnel ********
    elif(args.dist_name == 'nD_Funnel'):
        dic1 = tf.split(coords, args.input_dim, axis=0)
        term1 = dic1[0] ** 2 / (2 * 3 ** 2)
        for ii in range(1, int(args.input_dim / 2)):
            term1 += dic1[ii] ** 2 / (2 * (tf.math.exp(dic1[0] / 2)) ** 2)
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    # ******** nD Rosenbrock ********
    elif(args.dist_name == 'nD_Rosenbrock'):
        dic1 = tf.split(coords, args.input_dim, axis=0)
        term1 = 0.0
        for ii in range(0, int(args.input_dim / 2) - 1):
            term1 += (100.0 * (dic1[ii + 1] - dic1[ii] ** 2) ** 2 + (1 - dic1[ii]) ** 2) / 20.0
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    # ******** nD standard Gaussian ********
    elif(args.dist_name == 'nD_standard_Gaussian'):
        dic1 = tf.split(coords, args.input_dim, axis=0)
        var1 = tf.ones(int(args.input_dim), dtype=tf.float32)
        term1 = dic1[0] ** 2 / (2 * var1[0])
        for ii in range(1, int(args.input_dim / 2)):
            term1 += dic1[ii] ** 2 / (2 * var1[ii])
        term2 = 0.0
        for ii in range(int(args.input_dim / 2), args.input_dim):
            term2 += dic1[ii] ** 2 / 2
        H = term1 + term2

    else:
        raise ValueError("probability distribution name not recognized")

    return H