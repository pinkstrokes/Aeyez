#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预处理OCRBench-v2数据，将图片和题目信息保存到本地JSON文件中
避免在推理时加载大量数据导致内存溢出

用法:
1. 运行此脚本生成本地数据文件
2. 使用 infer_ocrbench_local.py 读取本地文件进行推理
"""

import os
import io
import json
import base64
from datetime import datetime
from typing import List, Dict, Any

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """将PIL图片转换为base64字符串"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64

def prepare_ocrbench_data(
    data_dir: str = "OCRBench-v2/data",
    output_file: str = "ocrbench_local_data.json",
    split: str = "train",
    samples_per_type: int = 30,
    max_types: int = 30
):
    """
    准备OCRBench数据，保存到本地JSON文件
    
    Args:
        data_dir: parquet文件目录
        output_file: 输出JSON文件路径
        split: 数据集分割
        samples_per_type: 每种问题类型的样本数
        max_types: 最大问题类型数
    """
    print(f"🔄 开始加载OCRBench数据从: {data_dir}")
    
    # 载入本地 parquet 分片
    ds = load_dataset("parquet", data_files=os.path.join(data_dir, "*.parquet"), split=split)
    
    print(f"📊 原始数据集大小: {len(ds)}")
    
    # 按问题类型采样
    if samples_per_type > 0:
        print(f"🎯 按问题类型采样: 每种类型 {samples_per_type} 个样本，最多 {max_types} 种类型...")
        
        # 按类型分组
        type_groups = {}
        for i, sample in enumerate(tqdm(ds, desc="分组数据")):
            q_type = sample.get('type', 'unknown')
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append(i)
        
        print(f"📋 发现 {len(type_groups)} 种问题类型:")
        for q_type, indices in type_groups.items():
            print(f"  {q_type}: {len(indices)} 个样本")
        
        # 选择前 max_types 种类型，每种采样 samples_per_type 个
        selected_indices = []
        selected_types = list(type_groups.keys())[:max_types]
        
        for q_type in selected_types:
            indices = type_groups[q_type]
            # 固定采样：选择前 samples_per_type 个样本（确保每次结果一致）
            sampled_indices = indices[:min(samples_per_type, len(indices))]
            selected_indices.extend(sampled_indices)
            print(f"  📝 从 {q_type} 类型中采样了 {len(sampled_indices)} 个样本")
        
        print(f"✅ 总共选择了 {len(selected_indices)} 个样本索引")
        
        # 处理选定的样本
        processed_data = []
        for idx in tqdm(selected_indices, desc="处理样本"):
            sample = ds[idx]
            
            # 提取基本信息
            pil_img: Image.Image = sample["image"]
            question: str = sample["question"]
            answers: List[str] = sample["answers"]
            sid = int(sample["id"]) if "id" in sample else idx
            question_type: str = sample.get("type", "basic")
            dataset_name: str = sample.get("dataset_name", "unknown")
            eval_method: str = sample.get("eval", None)
            
            # 将图片转换为base64
            img_base64 = pil_to_base64(pil_img, fmt="PNG")
            
            # 创建样本记录
            sample_record = {
                "index": idx,
                "id": sid,
                "question": question,
                "answers": answers,
                "type": question_type,
                "dataset_name": dataset_name,
                "eval": eval_method,
                "image_base64": img_base64,
                "image_format": "PNG"
            }
            
            processed_data.append(sample_record)
    
    else:
        # 处理所有数据
        processed_data = []
        for i, sample in enumerate(tqdm(ds, desc="处理所有样本")):
            pil_img: Image.Image = sample["image"]
            question: str = sample["question"]
            answers: List[str] = sample["answers"]
            sid = int(sample["id"]) if "id" in sample else i
            question_type: str = sample.get("type", "basic")
            dataset_name: str = sample.get("dataset_name", "unknown")
            eval_method: str = sample.get("eval", None)
            
            # 将图片转换为base64
            img_base64 = pil_to_base64(pil_img, fmt="PNG")
            
            sample_record = {
                "index": i,
                "id": sid,
                "question": question,
                "answers": answers,
                "type": question_type,
                "dataset_name": dataset_name,
                "eval": eval_method,
                "image_base64": img_base64,
                "image_format": "PNG"
            }
            
            processed_data.append(sample_record)
    
    # 创建元数据
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_data_dir": data_dir,
        "split": split,
        "samples_per_type": samples_per_type,
        "max_types": max_types,
        "total_samples": len(processed_data),
        "question_types": list(set(item["type"] for item in processed_data)),
        "dataset_names": list(set(item["dataset_name"] for item in processed_data))
    }
    
    # 保存到JSON文件
    output_data = {
        "metadata": metadata,
        "samples": processed_data
    }
    
    print(f"💾 保存数据到: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("✅ 数据准备完成!")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 总样本数: {len(processed_data)}")
    print(f"🔖 问题类型数: {len(metadata['question_types'])}")
    print(f"📋 数据集数: {len(metadata['dataset_names'])}")
    
    # 计算文件大小
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"💿 文件大小: {file_size_mb:.2f} MB")
    print("="*60)
    
    return output_file, metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备OCRBench本地数据文件")
    parser.add_argument("--data-dir", type=str, default="OCRBench-v2/data",
                        help="parquet文件目录路径")
    parser.add_argument("--output-file", type=str, default="ocrbench_local_data.json",
                        help="输出JSON文件路径")
    parser.add_argument("--split", type=str, default="train",
                        help="数据集分割")
    parser.add_argument("--samples-per-type", type=int, default=30,
                        help="每种问题类型的样本数")
    parser.add_argument("--max-types", type=int, default=30,
                        help="最大问题类型数")
    
    args = parser.parse_args()
    
    prepare_ocrbench_data(
        data_dir=args.data_dir,
        output_file=args.output_file,
        split=args.split,
        samples_per_type=args.samples_per_type,
        max_types=args.max_types
    )
