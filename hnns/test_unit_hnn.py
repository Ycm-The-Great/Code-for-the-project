#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:43:34 2024

@author: yanchuanmiao
"""

# test_integration.py
import unittest

# 正确引入四个文件中的所有需要测试的函数
from get_args import get_args
from train_hnn import train
from hnn_hmc import  get_control_group_hmc, hnn_sampling
from hnn_lmc import  get_control_group_lmc, hnn_lmc_sampling
from hnn_nuts_online import get_model, get_args_NUTs, hnn_nuts_sampling
import os

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        在测试开始前，获取所有模块的 args 对象。
        """
        # cls.args = get_args()  # 获取 train_hnn.py 的参数
        # cls.hmc_args = get_control_group_hmc()  # 获取 hnn_hmc.py 的参数
        # cls.lmc_args = get_control_group_lmc()  # 获取 hnn_lmc.py 的参数
        # cls.nuts_args = get_args_NUTs()  # 获取 hnn_nuts_online.py 的参数
        try:
            cls.args = get_args()  # 获取 train_hnn 的参数
        except SystemExit:
            cls.args = None  # 如果命令行参数不匹配，忽略该模块
        
        try:
            cls.hmc_args = get_control_group_hmc()  # 获取 hnn_hmc 的参数
        except SystemExit:
            cls.hmc_args = None
  
        try:
            cls.lmc_args = get_control_group_lmc()  # 获取 hnn_lmc 的参数
        except SystemExit:
            cls.lmc_args = None
  
        try:
            cls.nuts_args = get_args_NUTs()  # 获取 hnn_nuts_online 的参数
        except SystemExit:
            cls.nuts_args = None

    def test_00_train_hnn(self):
        """
        测试 train_hnn.py 的逻辑
        """
        print("Unitest: Training")
        # print(self.args)
        model, stats = train(self.args)  # 调用训练逻辑
        # 验证模型是否正确保存（路径来自 args）
        model_path = f"{self.args.save_dir}/{self.args.dist_name}.h5"
        model.save_weights(model_path)  # 保存模型的权重
        self.assertTrue(
            os.path.exists(model_path), f"Model not found at {model_path}"
        )

    def test_01_hnn_hmc(self):
        """
        测试 hnn_hmc.py 的逻辑
        """
        print("Unitest: hnn_hmc")
        # print(self.args)

        hnn_sampling(self.args, self.hmc_args)  # 调用 hnn_hmc 的采样逻辑

    def test_02_hnn_lmc(self):
        """
        测试 hnn_lmc.py 的逻辑
        """
        print("Unitest: hnn_lmc")
        # print(self.args)
        hnn_lmc_sampling(self.args,self.lmc_args)  # 调用 hnn_lmc 的采样逻辑

    def test_03_hnn_nuts_online(self):
        """
        测试 hnn_nuts_online.py 的逻辑
        """
        print("Unitest: hnn_nuts_online")
        # print(self.args)
        hnn_model = get_model(self.args, baseline=True)
        hnn_nuts_sampling(self.args, self.nuts_args)  # 调用 hnn_nuts_online 的采样逻辑


if __name__ == "__main__":
    unittest.main()