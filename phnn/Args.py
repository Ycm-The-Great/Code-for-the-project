#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 03:53:33 2025

@author: yanchuanmiao
"""

import argparse
from dataclasses import dataclass
import json
from typing import Optional, Dict, Any
# python your_script.py --json_config config.json --num_samples 2000 --lambda1 20.0
# python script.py --json_config config.json --model glmm --num_samples 1000

@dataclass
class Args:
    """参数管理类 (包含默认值)"""
    # 全局参数
    model: str = 'glmm'               # 模型类型 ('gaussian', 'diffraction', 'glmm')
    num_samples: int = 500           # 生成样本数
    test_split: float = 0.0          # 测试集比例
    should_load: bool = True        # 是否加载数据
    save_path: str = 'glmm.pkl'      # 数据保存路径
    load_path: str = 'glmm.pkl'      # 数据加载路径
    seed: int = 42
    T: int = 500
    N: int = 10
    y_path: str = 'y_glmm.pkl'
    
    # 轨迹生成参数
    t_span: list = (0, 0.5)          # 时间范围
    timescale: int = 10              # 时间刻度（总步数 = timescale * (t_end - t_start)）
    sigma_x: float = 0.1             # 隐变量噪声标准差
    
    # 模型特定参数
    ## Gaussian 模型
    mu_theta: float = 0.0            # 参数 theta 的先验均值
    sigma_theta: float = 10.0         # 参数 theta 的先验标准差
    sigma_y: float = 1.0             # 观测噪声标准差
    
    ## Diffraction 模型
    mu: float = 1.0                  # 隐变量均值
    sigma: float = 1.0               # 隐变量标准差
    lambda_: float = 0.1             # 衍射参数
    
    ## GLMM 模型
    p: int = 8                       # 协变量维度
    mu1: float = 0.0                 # 混合分布均值1
    mu2: float = 3.0                 # 混合分布均值2
    lambda1: float = 10.0            # 混合分布精度1
    lambda2: float = 3.0             # 混合分布精度2
    w1: float = 0.5                  # 混合分布权重
    n_i: int = 6                     # 每个主体的样本数
    
    
    # 新增网络结构参数
    hidden_dim: int = 128              # PHNN 隐藏层维度（核心新增参数）
    learn_rate: float = 0.001
    batch_size: int = 32
    num_steps: int = 3000
    print_every: int = 100
    
    chains: int = 1  # 马尔科夫链数量
    N :   int = 1000  # 每条链的采样数
    L : int= 10  # 每条链的哈密顿轨迹长度
    burn: int = 100  # 丢弃的burn-in样本数量
    epsilon : float= 0.025  # 时间积分步长
    
    @property
    def input_theta_dim(self) -> int:
        """theta 的维度 (根据模型动态计算)"""
        if self.model == 'glmm':
            return self.p + 5  # beta (p) + 5个混合分布参数
        elif self.model == 'diffraction':
            return 3       
        return 2  # Gaussian/Diffraction 默认为2维
    
    @property
    def aux_dim(self) -> int:
        """隐变量维度 (根据模型动态计算)"""
        if self.model == 'glmm':
            return self.T * self.N * self.n_i* self.input_theta_dim  # 隐变量维度示例
        return self.T * self.N* self.input_theta_dim
    
    # def parse_args(self):
    #     """解析命令行参数 (覆盖默认值)"""
    #     parser = argparse.ArgumentParser()
    #     for field in self.__dataclass_fields__:
    #         parser.add_argument(f'--{field}', type=type(getattr(self, field)), default=getattr(self, field))
    #     args = parser.parse_args()
    #     self.__dict__.update(vars(args))
    #     return self
    
    def parse_args(self) -> 'Args':
        """解析参数 (优先级：命令行 > JSON > 默认值)"""
        # 1. 初步解析以获取 JSON 文件路径
        prelim_parser = argparse.ArgumentParser(add_help=False)
        prelim_parser.add_argument('--json_config', type=str, default=None)
        prelim_args, _ = prelim_parser.parse_known_args()

        # 2. 加载 JSON 配置
        if prelim_args.json_config:
            self.load_json(prelim_args.json_config)

        # 3. 解析所有命令行参数（覆盖 JSON 和默认值）
        parser = self._create_parser()
        cmd_args = parser.parse_args()
        self.__dict__.update(vars(cmd_args))
        return self

    def _create_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器"""
        parser = argparse.ArgumentParser()
        for field in self.__dataclass_fields__:
            if field == 'json_config':
                continue  # 已在初步解析中处理
            parser.add_argument(
                f'--{field}',
                type=type(getattr(self, field)),
                default=getattr(self, field)
            )
        return parser

    def load_json(self, json_path: str) -> None:
        """从 JSON 文件加载配置"""
        with open(json_path, 'r') as f:
            config: Dict[str, Any] = json.load(f)
            for key, value in config.items():
                if key in self.__dataclass_fields__:
                    setattr(self, key, value)

    def to_json(self, json_path: str) -> None:
        """保存当前配置到 JSON 文件"""
        with open(json_path, 'w') as f:
            config = {k: v for k, v in self.__dict__.items() if k != 'json_config'}
            json.dump(config, f, indent=2)

# 示例用法
if __name__ == "__main__":
    args = Args().parse_args()
    print(args)
    # 创建参数实例
    args = Args()
    
    # 加载 JSON 配置文件
    json_path = "config.json"
    args.load_json(json_path)
    
    # 可选：手动覆盖某些参数
    
    # 打印结果
    print("当前配置:")
    print(f"模型: {args.model}")
    print(f"样本数: {args.num_samples}")
    print(f"时间范围: {args.t_span}")
    print(f"隐变量噪声标准差: {args.sigma_x}")
    print(f"mu_theta: {args.mu_theta}")
    print(f"sigma_theta: {args.sigma_theta}")
    print(f"sigma_y: {args.sigma_y}")
    print(f"timescale (手动覆盖): {args.timescale}")
