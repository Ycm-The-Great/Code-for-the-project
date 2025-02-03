#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 04:14:37 2025

@author: yanchuanmiao
"""
import json
config={
  'model':  'glmm'   ,          # 模型类型 ('gaussian', 'diffraction', 'glmm')
  'num_samples':  200,           # 生成样本数
  "test_split":  0.0 ,         # 测试集比例
  "should_load":  True ,       # 是否加载数据
  'save_path':  'glmm.pkl',      # 数据保存路径
  'load_path':  'glmm.pkl',      # 数据加载路径
  
  
  # 轨迹生成参数
  't_span': [0, 0.5],          # 时间范围
  'timescale':  10  ,            # 时间刻度（总步数 = timescale * (t_end - t_start)）
  'sigma_x':  0.1 ,            # 隐变量噪声标准差
  
  # 模型特定参数
  ## Gaussian 模型
  'mu_theta': 0.0 ,           # 参数 theta 的先验均值
  'sigma_theta': 10.0,         # 参数 theta 的先验标准差
  'sigma_y': 1.0,             # 观测噪声标准差
  
  ## Diffraction 模型
  'mu': 1.0  ,                # 隐变量均值
  'sigma': 1.0,               # 隐变量标准差
  'lambda_':  0.1,             # 衍射参数
  
  ## GLMM 模型
  'p': 8,                       # 协变量维度
  'mu1':  0.0 ,                # 混合分布均值1
  'mu2':  3.0 ,                # 混合分布均值2
  'lambda1': 10.0 ,           # 混合分布精度1
  'lambda2':  3.0  ,           # 混合分布精度2
  'w1': 0.5,                  # 混合分布权重
  'n_i':  6  ,                   # 每个主体的样本数
  
  'hidden_dim': 128
  }
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

print("config.json 文件已生成！")