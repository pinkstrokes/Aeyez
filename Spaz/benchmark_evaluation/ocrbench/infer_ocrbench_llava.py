#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLaVA-1.5-7B inference on OCRBench-v2 using Transformers
Combines the LLaVA model wrapper with OCRBench evaluation pipeline
"""

import os
import io
import json
import base64
import argparse
from datetime import datetime
from typing import List
from PIL import Image
from tqdm import tqdm
import torch

# Disable flash attention to avoid GLIBC version issues
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

from transformers import AutoProcessor, LlavaForConditionalGeneration


class LLaVAModel:
    """Simple LLaVA model wrapper using Transformers"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"📦 加载 LLaVA 模型: {model_path}")
        print(f"🖥️  使用设备: {device}")

        self.device = device
        
        # Try to load model with low_cpu_mem_usage, fallback if accelerate is not available
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Disable flash attention
            ).to(device)
        except ImportError as e:
            print("⚠️ accelerate 库未安装，使用标准加载方式（可能需要更多内存）")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                attn_implementation="eager"  # Disable flash attention
            ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_path)

        print("✅ 模型加载成功!")

    def generate(self, text, image, max_new_tokens=512, temperature=0.01):
        """Generate response from text and image"""
        # Prepare conversation format
        content = [
            {"type": "image"},
            {"type": "text", "text": text}
        ]

        conversation = [
            {
                "role": "user",
                "content": content,
            },
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode
        generated_text = self.processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()


def base64_to_pil(b64_string: str) -> Image.Image:
    """将base64字符串转换为PIL图片"""
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data)).convert('RGB')


def load_local_data(json_file: str) -> tuple:
    """从本地JSON文件加载数据"""
    print(f"📂 从本地JSON文件加载数据: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    samples = data['samples']
    
    print(f"✅ 加载完成: {len(samples)} 个样本")
    print(f"📊 问题类型: {len(metadata['question_types'])} 种")
    print(f"📋 数据集: {len(metadata['dataset_names'])} 个")
    
    return samples, metadata


def is_correct_simple(pred: str, gt_list: List[str]) -> bool:
    """简单的答案匹配"""
    pred_normalized = pred.strip().lower()
    for gt in gt_list:
        if gt.strip().lower() in pred_normalized:
            return True
    return False


def run_inference(args):
    """在OCRBench上运行LLaVA推理"""
    print("=" * 60)
    print("LLaVA-1.5-7B OCRBench 推理")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = os.path.join(args.output_dir, f"ocrbench_llava_pred_{timestamp}.jsonl")
    log_path = os.path.join(args.output_dir, f"ocrbench_llava_run_{timestamp}.log")

    # 加载数据
    samples, metadata = load_local_data(args.data_file)
    n_total = len(samples) if args.limit < 0 else min(args.limit, len(samples))
    
    # 限制样本数量
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"🎯 限制处理样本数: {n_total}")

    # 记录配置到日志文件
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("========== LLaVA OCRBench 评估开始 ==========\n")
        lf.write(f"开始时间: {start_time}\n")
        lf.write(f"模型路径: {args.model_path}\n")
        lf.write(f"数据文件: {args.data_file}\n")
        lf.write(f"输出目录: {args.output_dir}\n")
        lf.write(f"样本数量: {n_total}\n")
        lf.write(f"样本限制: {args.limit if args.limit > 0 else '无限制'}\n")
        lf.write(f"最大生成tokens: {args.max_tokens}\n")
        lf.write(f"温度: {args.temperature}\n")
        lf.write("\n========== 输出文件 ==========\n")
        lf.write(f"JSONL: {jsonl_path}\n")
        lf.write(f"LOG: {log_path}\n")
        lf.write("\n========== 开始处理 ==========\n")

    # 加载LLaVA模型
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = LLaVAModel(args.model_path, device=device)

    # 系统提示词
    system_prompt = (
        "You are an OCR QA assistant. Read the provided image and answer the user's question. "
        "Answer with the minimal text needed, no extra words or punctuation."
    )

    # 结果统计
    n_ok, n_fail = 0, 0
    type_stats = {}  # 统计问题类型
    results = []

    # 跳过的字典相关题目类型
    dict_related_types = [
        "key information extraction cn", "key information extraction en",
        "key information mapping en", "chart parsing en", "document parsing cn",
        "document parsing en", "handwritten answer extraction cn",
        "table parsing cn", "table parsing en"
    ]

    # 打开输出文件
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, sample in enumerate(tqdm(samples, desc="推理进度")):
            try:
                # 提取样本信息
                question: str = sample["question"]
                answers: List[str] = sample["answers"]
                sid = int(sample.get("id", i))
                question_type: str = sample.get("type", "basic")
                dataset_name: str = sample.get("dataset_name", "unknown")
                
                # 跳过字典相关题目
                if question_type in dict_related_types:
                    print(f"⏭️  跳过字典题目 (ID: {sid}) - {question_type}")
                    continue
                
                # 解码图片
                img_base64 = sample["image_base64"]
                image = base64_to_pil(img_base64)
                
                # 构造提示词
                prompt = f"{system_prompt}\n\nQuestion: {question}\nAnswer briefly with only the exact content found in the image."
                
                # 生成预测
                pred_text = model.generate(
                    prompt,
                    image,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                
                # 评估结果
                is_correct_pred = is_correct_simple(pred_text, answers)
                
                # 统计
                if is_correct_pred:
                    n_ok += 1
                else:
                    n_fail += 1
                
                type_stats[question_type] = type_stats.get(question_type, 0) + 1
                
                # 保存结果
                result = {
                    "id": sid,
                    "question": question,
                    "answers": answers,
                    "prediction": pred_text,
                    "correct": is_correct_pred,
                    "type": question_type,
                    "dataset_name": dataset_name,
                    "current_accuracy": n_ok / (n_ok + n_fail) if (n_ok + n_fail) > 0 else 0.0
                }
                results.append(result)
                jf.write(json.dumps(result, ensure_ascii=False) + "\n")
                jf.flush()
                
                # 定期输出进度
                if (i + 1) % args.log_every == 0:
                    print(f"\n📊 进度: {i + 1}/{n_total}, 当前准确率: {n_ok/(n_ok+n_fail):.4f}")
                
            except Exception as e:
                print(f"❌ 样本 {i} 处理失败: {e}")
                n_fail += 1
                continue

    # 最终摘要
    final_acc = n_ok / max(1, n_total)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n========== SUMMARY ==========\n")
        lf.write(f"结束时间: {end_time}\n")
        lf.write(f"总样本数: {n_total}\n")
        lf.write(f"正确数量: {n_ok}\n")
        lf.write(f"错误数量: {n_fail}\n")
        lf.write(f"最终正确率: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        
        lf.write("\n========== 问题类型统计 ==========\n")
        for q_type, count in type_stats.items():
            lf.write(f"{q_type}: {count}个样本\n")
        
        lf.write(f"\njsonl: {jsonl_path}\n")

    print("\n" + "=" * 60)
    print("✅ 评估完成!")
    print(f"总样本数: {n_total}")
    print(f"正确数量: {n_ok}")
    print(f"错误数量: {n_fail}")
    print(f"最终正确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print("\n问题类型统计:")
    for q_type, count in type_stats.items():
        print(f"  {q_type}: {count}个样本")
    print("=" * 60)
    print(f"📄 JSONL: {jsonl_path}")
    print(f"📄 LOG  : {log_path}")


def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5-7B OCRBench 推理")
    parser.add_argument("--model-path", type=str,
                       default="/projects/bdpn/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
                       help="LLaVA 模型路径")
    parser.add_argument("--data-file", type=str,
                       default="ocrbench_local_data.json",
                       help="OCRBench JSONL 数据文件")
    parser.add_argument("--output-dir", type=str,
                       default="outputs_ocrbench_llava",
                       help="输出目录")
    parser.add_argument("--limit", type=int, default=-1,
                       help="限制处理的样本数量 (-1表示全部)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.01,
                       help="采样温度")
    parser.add_argument("--cpu", action="store_true",
                       help="强制使用CPU (默认: 如果可用则使用GPU)")
    parser.add_argument("--log-every", type=int, default=50,
                       help="每N个样本输出一次进度")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()

