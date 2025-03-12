#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:59:32 2024

@author: yanchuanmiao
"""

import numpy as np
import os
import pickle
import zipfile
import imageio
import shutil
import tensorflow as tf

def leapfrog(dydt, tspan, y0, n, dim):
    """
    Implements the leapfrog integrator for solving ODEs.
    """
    t0 = tspan[0]
    tstop = tspan[1]
    dt = (tstop - t0) / n

    t = np.zeros(n + 1)
    y = np.zeros([dim, n + 1])

    for i in range(0, n + 1):
        if i == 0:
            t[0] = t0
            for j in range(0, dim):
                y[j, 0] = y0[j]
            anew = dydt(t, y[:, i])
        else:
            t[i] = t[i - 1] + dt
            aold = anew
            for j in range(0, int(dim / 2)):
                y[j, i] = y[j, i - 1] + dt * (
                    y[(j + int(dim / 2)), i - 1] + 0.5 * dt * aold[(j + int(dim / 2))]
                )
            anew = dydt(t, y[:, i])
            for j in range(0, int(dim / 2)):
                y[(j + int(dim / 2)), i] = y[(j + int(dim / 2)), i - 1] + 0.5 * dt * (
                    aold[(j + int(dim / 2))] + anew[(j + int(dim / 2))]
                )
    return y
def leapfrog_tf(dydt, tspan, y0, n, dim):
    """
    Implements the leapfrog integrator for solving ODEs in TensorFlow.
    """
    t0, tstop = tspan
    dt = (tstop - t0) / tf.cast(n, tf.float32)

    # 初始化时间和状态变量
    t = tf.TensorArray(dtype=tf.float32, size=n + 1, dynamic_size=False,clear_after_read=False)
    y = tf.TensorArray(dtype=tf.float32, size=n + 1, dynamic_size=False,clear_after_read=False)

    # 初始条件
    t = t.write(0, t0)
    y = y.write(0, y0)

    # 计算初始加速度
    anew = dydt(t.read(0), y.read(0))

    # 迭代 Leapfrog 方法
    def loop_body(i, t, y, anew):
        t_next = t.read(i - 1) + dt
        aold = anew
        y_prev = y.read(i - 1)

        # 计算位置更新
        y_next = tf.identity(y_prev)
        for j in range(0, dim // 2):
            # print(j,y_next)
            y_next = tf.tensor_scatter_nd_update(
                y_next, [[j]], [y_prev[j] + dt * (y_prev[j + dim // 2] + 0.5 * dt * aold[j + dim // 2])]
            )

        anew = dydt(t_next, y_next)

        # 计算速度更新
        for j in range(0, dim // 2):
            y_next = tf.tensor_scatter_nd_update(
                y_next, [[j + dim // 2]], [y_prev[j + dim // 2] + 0.5 * dt * (aold[j + dim // 2] + anew[j + dim // 2])]
            )

        t = t.write(i, t_next)
        y = y.write(i, y_next)

        return i + 1, t, y, anew

    # 运行 Leapfrog 迭代
    _, t, y, _ = tf.while_loop(
        lambda i, t, y, anew: i < n + 1,
        loop_body,
        [1, t, y, anew]
    )

    return y.stack()  # 以 Tensor 形式返回所有状态

def lfrog(fun, y0, t, dt, *args, **kwargs):
    """
    Implements a simple leapfrog integrator step.
    """
    k1 = fun(y0, t - dt, *args, **kwargs)
    k2 = fun(y0, t + dt, *args, **kwargs)
    dy = (k2 - k1) / (2 * dt)
    return dy


def L2_loss(u, v):
    """
    Computes the L2 loss between two tensors.
    """
    return tf.reduce_mean(tf.square(u - v))


def to_pickle(thing, path):
    """
    Save an object to a pickle file.
    """
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):
    """
    Load an object from a pickle file.
    """
    thing = None
    with open(path, "rb") as handle:
        thing = pickle.load(handle)
    return thing


def choose_nonlinearity(name):
    """
    Returns the specified nonlinearity function.
    """
    if name == "tanh":
        nl = tf.math.tanh
    elif name == "relu":
        nl = tf.nn.relu
    elif name == "sigmoid":
        nl = tf.nn.sigmoid
    elif name == "softplus":
        nl = tf.nn.softplus
    elif name == "selu":
        nl = tf.nn.selu
    elif name == "elu":
        nl = tf.nn.elu
    elif name == "swish":
        nl = lambda x: x * tf.nn.sigmoid(x)
    elif name == "sine":
        nl = tf.math.sin
    else:
        raise ValueError("Nonlinearity not recognized")
    return nl
