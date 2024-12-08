#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:56:13 2024

@author: yanchuanmiao
"""

import json
import os

#***** Names of coded probability distribution functions: *****
# - '1D_Gauss_mix'
# - '2D_Gauss_mix'
# - '5D_illconditioned_Gaussian'
# - 'nD_Funnel'
# - 'nD_Rosenbrock'
# - 'nD_standard_Gaussian'

###  the input_dim should be 2*d_sample
 
# 定义参数
config = {
    "input_dim": 4,  # Dimensionality of input tensor (position + momentum variables)
    "num_samples": 10,  # Number of training samples simulated using Hamiltonian Monte Carlo
    "len_sample": 50,  # Length of Hamiltonian trajectory for each training sample
    "dist_name": "nD_Funnel",  # Name of the probability distribution function
    "save_dir": os.path.dirname(os.path.abspath(__file__)),  # Where to save the trained model
    "load_dir": os.path.dirname(os.path.abspath(__file__)),  # Where to load the training data from
    "should_load": False,  # Should load training data?
    "load_file_name": "nD_Funnel",  # If load training data, the file name (.pkl format)
    "total_steps": 5000,  # Number of gradient steps
    "hidden_dim": 100,  # Hidden dimension of MLP
    "num_layers": 3,  # Hidden layers of MLP
    "learn_rate": 0.0005,  # Learning rate
    "batch_size": 1000,  # Batch size
    "nonlinearity": "tanh",  # Neural net nonlinearity
    "test_fraction": 0.0,  # Fraction of testing samples
    "step_size": 0.025,  # Step size for time integration
    "print_every": 200,  # Number of gradient steps between prints
    "verbose": False,  # Verbose?
    "field_type": "solenoidal",  # Type of vector field to learn
    "seed": 0  # Random seed
}
# 保存为 config.json 文件
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

print("config.json 文件已生成！")

# 定义默认的参数配置
config_hmc = {
    "chains": 1,          # Number of Markov chains
    "epsilon": 0.025,     # Time integration step size
    "N": 100,            # Number of samples per chain
    "L": 10,              # Length of Hamiltonian trajectories
    "burn": 10          # Number of burn-in samples to discard
}

# 写入到 config_hmc.json 文件
with open("config_hmc.json", "w") as f:
    json.dump(config_hmc, f, indent=4)

print("config_hmc.json 文件已生成！")

# 定义默认的参数配置
config_lmc = {
    "chains": 1,          # Number of Markov chains
    "epsilon": 0.025,     # Time integration step size
    "N": 1000,           # Number of samples per chain
    "L": 10,              # Length of Hamiltonian trajectories
    "burn": 100          # Number of burn-in samples to discard
}

# 写入到 config_lmc.json 文件
with open("config_lmc.json", "w") as f:
    json.dump(config_lmc, f, indent=4)

print("config_lmc.json 文件已生成！")

# 定义默认的采样参数配置
config_nuts = {
    "N": 100,  # Number of samples
    "burn": 10,  # Number of burn-in samples
    "epsilon": 0.025,  # Step size
    "N_lf": 20,  # Number of cool-down samples when DNN integration errors are high
    "hnn_threshold": 10.0,  # DNN integration error threshold
    "lf_threshold": 1000.0  # Numerical gradient integration error threshold
}

# 写入到 config_nuts.json 文件
with open("config_nuts.json", "w") as f:
    json.dump(config_nuts, f, indent=4)

print("config_nuts.json 文件已生成！")

