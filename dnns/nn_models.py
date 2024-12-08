#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 03:57:35 2024

@author: yanchuanmiao
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils import choose_nonlinearity
from get_args import get_args
args = get_args()

class MLP(tf.keras.Model):
    '''Just a salt-of-the-earth MLP'''
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='sine'):
        super(MLP, self).__init__()
        # Define layers
        self.linear1 = layers.Dense(hidden_dim, use_bias=True, input_shape=(input_dim,))
        self.linear2 = layers.Dense(hidden_dim, use_bias=True)
        self.linear3 = layers.Dense(hidden_dim, use_bias=True)
        self.linear4 = layers.Dense(hidden_dim, use_bias=True)
        self.linear5 = layers.Dense(output_dim, use_bias=False)

        # Orthogonal initialization
        for layer in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
            self._apply_orthogonal_initializer(layer)

        # Choose nonlinearity
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def _apply_orthogonal_initializer(self, layer):
        """Apply orthogonal initialization to the kernel of the given layer."""
        initializer = tf.keras.initializers.Orthogonal()
        layer.kernel_initializer = initializer

    def call(self, x, separate_fields=False):
        if len(x.shape) == 1:  # Input is 1D, e.g., (input_dim,)
            x = tf.expand_dims(x, axis=0)  # Add batch dimension, shape becomes (1, input_dim)
        # Forward pass through the layers with nonlinearity
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        h = self.nonlinearity(self.linear4(h))
        return self.linear5(h)