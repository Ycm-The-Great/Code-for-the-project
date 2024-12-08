#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:47:19 2024

@author: yanchuanmiao
"""
import tensorflow as tf
import numpy as np
import argparse

from utils import lfrog
from get_args import get_args
args = get_args()

class HNN(tf.keras.Model):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)  # Levi-Civita permutation tensor
        self.field_type = field_type

    def call(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert len(y.shape) == 2 and y.shape[1] == args.input_dim, \
            "Output tensor should have shape [batch_size, 2]"
        # Split output into two parts
        dic1 = tf.split(y, num_or_size_splits=args.input_dim, axis=1)
        answer1 = tf.concat(dic1[0:int(args.input_dim / 2)], axis=1), \
                  tf.concat(dic1[int(args.input_dim / 2):args.input_dim], axis=1)
        return answer1

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        return self.differentiable_model(x)

    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            # Canonical coordinate assumption
            M = tf.eye(n)
            M = tf.concat([M[n // 2:], -M[:n // 2]], axis=0)
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = tf.ones((n, n))  # Matrix of ones
            M *= 1 - tf.eye(n)  # Clear diagonals
            M = tf.tensor_scatter_nd_update(M, [[i, i] for i in range(n)], tf.zeros(n))  # Clear diagonal entries
            M = tf.transpose(M) * -1 if n % 2 == 0 else M  # Adjust for sign pattern
            for i in range(n):  # Make asymmetric
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M
