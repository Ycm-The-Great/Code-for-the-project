#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 06:44:04 2025

@author: yanchuanmiao
"""

import unittest
from Args import Args  # 从独立文件导入参数类
from PHNN_train import train
from phnn_hmc import phnn_sampling
import os

# python -m unittest test_integration.py

class TestIntegration(unittest.TestCase):
    def test_workflow(self):
        """最简集成测试：仅按顺序调用你的原始代码"""
        # 训练阶段（完全使用你的原始参数解析）
        args = Args().parse_args()
        model, stats = train(args)  # 不检查输出
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        # 保存模型
        path = os.path.join(THIS_DIR, f"{args.model}.h5")  # 模型保存路径
        model.save_weights(path)  # 保存模型的权重
       
        phnn_sampling(args)  # 不检查输出
    def test_train(self):
        """只调用训练函数，不验证任何结果"""
        args = Args().parse_args()  # 你的原始参数解析
        model, stats = train(args)  # 只要不报错即通过
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))

        # 保存模型
        path = os.path.join(THIS_DIR, f"{args.model}.h5")  # 模型保存路径
        model.save_weights(path)  # 保存模型的权重
    def test_sampling(self):
        """只调用采样函数，不验证任何结果"""
        args = Args().parse_args()  # 你的原始参数解析
        phnn_sampling(args)  # 只要不报错即通过




if __name__ == "__main__":
    unittest.main()