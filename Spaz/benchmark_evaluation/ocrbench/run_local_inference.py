#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用本地JSON数据运行OCR推理的便捷脚本
避免内存溢出问题

使用方法:
1. 首先运行数据准备脚本：
   python prepare_ocrbench_data.py

2. 然后运行推理：
   python run_local_inference.py --model qwen3b     # 3B Qwen模型
   python run_local_inference.py --model qwen7b     # 7B Qwen模型  
   python run_local_inference.py --model openai     # OpenAI GPT模型
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入推理脚本
from infer_ocrbench_qwen import run_infer as run_infer_qwen
from infer_ocrbench_qwen7b import run_infer as run_infer_qwen7b
from infer_ocrbench_openai import run_infer as run_infer_openai

def main():
    parser = argparse.ArgumentParser(description="运行本地OCR推理")
    parser.add_argument("--model", type=str, choices=["qwen3b", "qwen7b", "openai"], 
                        default="qwen3b", help="选择模型: qwen3b/qwen7b/openai")
    parser.add_argument("--json-file", type=str, default="ocrbench_local_data.json",
                        help="本地JSON数据文件路径")
    parser.add_argument("--limit", type=int, default=-1, 
                        help="限制样本数量，-1表示全部")
    parser.add_argument("--log-every", type=int, default=50,
                        help="每多少个样本输出一次日志")
    
    args = parser.parse_args()
    
    # 检查JSON文件是否存在
    if not os.path.exists(args.json_file):
        print(f"❌ JSON数据文件不存在: {args.json_file}")
        print("请先运行: python prepare_ocrbench_data.py")
        return
    
    print(f"🚀 启动{args.model.upper()}模型推理...")
    print(f"📂 数据文件: {args.json_file}")
    print(f"📊 样本限制: {'全部' if args.limit == -1 else args.limit}")
    
    # 根据模型选择运行不同的推理脚本
    if args.model == "qwen3b":
        output_dir = "outputs_ocrbench_qwen3b_local"
        config_name = "translator"
        asyncio.run(run_infer_qwen(
            data_dir="",  # 不使用
            output_dir=output_dir,
            config_name=config_name,
            limit=args.limit,
            log_every=args.log_every,
            local_json_file=args.json_file
        ))
    
    elif args.model == "qwen7b":
        output_dir = "outputs_ocrbench_qwen7b_local"
        config_name = "qwen2_5_vl_7b_dashscope"
        asyncio.run(run_infer_qwen7b(
            data_dir="",  # 不使用
            output_dir=output_dir,
            config_name=config_name,
            limit=args.limit,
            log_every=args.log_every,
            local_json_file=args.json_file
        ))
    
    elif args.model == "openai":
        output_dir = "outputs_ocrbench_openai_local"
        config_name = "default"
        asyncio.run(run_infer_openai(
            data_dir="",  # 不使用
            output_dir=output_dir,
            config_name=config_name,
            limit=args.limit,
            log_every=args.log_every,
            local_json_file=args.json_file
        ))
    
    print(f"\n✅ {args.model.upper()}模型推理完成!")
    print(f"📁 结果保存在: {output_dir}/")

if __name__ == "__main__":
    main()
