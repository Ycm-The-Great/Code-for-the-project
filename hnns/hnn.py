#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:47:19 2024

@author: yanchuanmiao
"""
import tensorflow as tf
import numpy as np
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
        # Traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert len(y.shape) == 2 and y.shape[1] == args.input_dim, "Output tensor should have shape [batch_size, 2]"
       
        # Split and return as tuple
        # dic1 = tf.split(y, num_or_size_splits=2, axis=1)
        # 切分 dic1 为两部分
       
        # dic1 = tf.split(y, num_or_size_splits=args.input_dim, axis=1)
        
        # #  拼接前半部分
        # first_half = tf.concat(dic1[0:int(args.input_dim / 2)], axis=1)
        
        # #  拼接后半部分
        # second_half = tf.concat(dic1[int(args.input_dim / 2):args.input_dim], axis=1)
        
        first_half = y[:, :args.input_dim // 2]  # 前一半切片
        second_half = y[:, args.input_dim // 2:]  # 后一半切片

        # 打印调试信息，检查切片结果是否正确
        # print("y:", y)
        # print("first_half:", first_half)
        # print("second_half:", second_half)
        answer1 = (first_half, second_half)

        # # 拼接前半部分
        # if first_half:
        #     first_half_concat = first_half[0]  # 以第一个张量为起点
        #     for tensor in first_half[1:]:  # 逐步拼接剩余的张量
        #         first_half_concat = tf.concat([first_half_concat, tensor], axis=1)
        # else:
        #     first_half_concat = None  # 如果前半部分为空
     
        # # 拼接后半部分
        # if second_half:
        #     second_half_concat = second_half[0]  # 以第一个张量为起点
        #     for tensor in second_half[1:]:  # 逐步拼接剩余的张量
        #         second_half_concat = tf.concat([second_half_concat, tensor], axis=1)
        # else:
        #     second_half_concat = None  # 如果后半部分为空
        
        # answer1 = (first_half_concat, second_half_concat)
        # print(answer1)

        # answer1 = (tf.concat(dic1[:args.input_dim // 2], axis=1),
        #            tf.concat(dic1[args.input_dim // 2:], axis=1))
        # print(answer1)
        return tf.split(y, num_or_size_splits=2, axis=1)

    def lfrog_time_derivative(self, x, dt):
        return lfrog(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        if len(x.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            x = tf.expand_dims(x, axis=0)  # Add batch dimension, shape becomes (1, input_dim)
        '''NEURAL ODE-STYLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STYLE VECTOR FIELD'''
        #F1, F2 = self.call(x)  # Traditional forward pass
        ''''F1, F2 = self.call(x) 写在tf.GradientTape后才有用'''
        conservative_field = tf.zeros_like(x)  # Start out with both components set to 0
        solenoidal_field = tf.zeros_like(x)

        if self.field_type != 'solenoidal':
            with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
                tape.watch(x)
                F1, F2 = self.call(x) 
                F1_sum = tf.reduce_sum(F1)
            dF1 = tape.gradient(F1_sum, x)  # Gradients for conservative field
            conservative_field = tf.linalg.matmul(dF1, tf.eye(self.M.shape[0]))

        if self.field_type != 'conservative':
            with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
                tape.watch(x)
                F1, F2 = self.call(x) 
                F2_sum = tf.reduce_sum(F2)
                #print(F2,F2_sum)
            dF2 = tape.gradient(F2_sum, x)  # Gradients for solenoidal field
            solenoidal_field = tf.linalg.matmul(dF2, tf.transpose(self.M))
            

        if separate_fields:
            return [conservative_field, solenoidal_field]
        del tape  
        return conservative_field + solenoidal_field

    def permutation_tensor(self, n):
        """
        Constructs the Levi-Civita permutation tensor or identity matrix based on input dimensions.
        """
        if self.assume_canonical_coords:
            M = tf.eye(n)
            M = tf.concat([M[n // 2:], -M[:n // 2]], axis=0)
        else:
            # Constructs the Levi-Civita permutation tensor
            M = tf.ones((n, n))  # Matrix of ones
            M *= (1 - tf.eye(n))  # Clear diagonals
            M = tf.tensor_scatter_nd_update(M, [[i, i] for i in range(n)], tf.zeros(n))  # Clear diagonals
            M = tf.where(tf.range(n) % 2 == 0, -M, M)  # Pattern of signs
            M = tf.where(tf.range(n)[:, None] % 2 == 0, -M, M)  # Pattern of signs (columns)

            for i in range(n):  # Make asymmetric
                for j in range(i + 1, n):
                    M = tf.tensor_scatter_nd_update(M, [[i, j]], [-M[i, j]])
        return M