#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference on OCRBench-v2 parquet shards with your LLM interface.

- Loads data from local parquet files: data/*.parquet
- For each sample: (image, question) -> model -> prediction
- Compares against answers (list[str]) with normalization
- Writes per-sample JSONL + CSV and a .log summary
"""

import os
import io
import re
import json
import base64
import argparse
import ast
from datetime import datetime
from typing import List

from PIL import Image
from tqdm import tqdm
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# 导入统一的评估脚本
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "OCRBench_v2_eval/eval_scripts"))
from eval import process_predictions

# === 关键：导入你给的 LLM 接口 ===
# 假设 infer_ocrbench.py 与 src/ 在同一层；按你的工程实际结构调整。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.llm import LLM, Message  # 修改为你项目里 llm.py 的实际路径

# ---------- 评估函数 ----------

def get_evaluation_score(pred: str, answers: List[str], question_type: str, eval_method: str = None, question: str = "") -> float:
    """使用统一的eval.py评估方法"""
    try:
        # 构造符合eval.py期望的数据格式
        data_item = {
            "predict": pred,
            "answers": answers,
            "type": question_type,
            "question": question
        }
        
        # 如果有eval方法，添加到数据项中
        if eval_method:
            data_item["eval"] = eval_method
        
        # 使用eval.py中的评估逻辑
        # 这里我们需要创建一个临时的评估函数，基于eval.py的逻辑
        return evaluate_single_sample(data_item)
    
    except Exception as e:
        print(f"评估错误 ({question_type}): {e}")
        return 0.0

def evaluate_single_sample(data_item):
    """直接使用eval.py的评估逻辑"""
    try:
        # 创建一个临时的评估函数，直接调用eval.py中的process_predictions逻辑
        # 但只处理单个样本
        import tempfile
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump([data_item], temp_file, ensure_ascii=False, indent=2)
            temp_input_path = temp_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            temp_output_path = temp_file.name
        
        try:
            # 调用eval.py的process_predictions函数
            process_predictions(temp_input_path, temp_output_path)
            
            # 读取结果
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            if result and len(result) > 0:
                return result[0].get('score', 0.0)
            else:
                return 0.0
                
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except OSError:
                pass
            
    except Exception as e:
        print(f"单样本评估错误: {e}")
        return 0.0

def is_correct(pred: str, gt_list: List[str], question_type: str = "basic", eval_method: str = None, question: str = "") -> bool:
    """使用综合评估函数判断对错"""
    score = get_evaluation_score(pred, gt_list, question_type, eval_method, question)
    return score >= 0.5

# ---------- 图片处理函数 ----------
def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    try:
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    finally:
        buf.close()

def base64_to_data_url(b64_string: str, fmt: str = "PNG") -> str:
    """将base64字符串转换为data URL"""
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64_string}"

def base64_to_pil(b64_string: str) -> Image.Image:
    """将base64字符串转换为PIL图片"""
    img_data = base64.b64decode(b64_string)
    buf = io.BytesIO(img_data)
    try:
        img = Image.open(buf)
        img.load()
        return img
    finally:
        buf.close()

def stream_json_samples(json_file: str, limit: int = -1):
    """流式读取JSON文件中的样本，避免一次性加载所有数据到内存"""
    print(f"📂 流式读取JSON文件: {json_file}")
    
    # 检查文件格式：如果是标准JSON格式，需要特殊处理
    with open(json_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # 重置文件指针
        
        if first_line.startswith('{') and not first_line.startswith('{"id"'):
            # 标准JSON格式，包含metadata和samples
            print("📋 检测到标准JSON格式，使用ijson进行流式解析...")
            try:
                import ijson
                # 使用ijson进行流式解析，只解析samples数组
                count = 0
                with open(json_file, 'rb') as f_binary:
                    for sample in ijson.items(f_binary, 'samples.item'):
                        if limit > 0 and count >= limit:
                            break
                        count += 1
                        yield sample
            except ImportError:
                print("⚠️ ijson未安装，回退到传统加载方式...")
                print("💡 建议安装ijson: pip install ijson")
                data = json.load(f)
                samples = data.get('samples', [])
                
                count = 0
                for sample in samples:
                    if limit > 0 and count >= limit:
                        break
                    count += 1
                    yield sample
        else:
            # JSONL格式，每行一个JSON对象
            print("📋 检测到JSONL格式，逐行读取...")
            count = 0
            for line_num, line in enumerate(f, 1):
                if limit > 0 and count >= limit:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    sample = json.loads(line)
                    count += 1
                    yield sample
                except json.JSONDecodeError as e:
                    print(f"⚠️ 跳过第{line_num}行，JSON解析错误: {e}")
                    continue
    
    print(f"✅ 流式读取完成，共处理 {count} 个样本")

async def process_sample(i, sample, question, answers, sid, question_type, dataset_name, eval_method, data_url, llm, jf, eval_method_stats, type_stats, dataset_stats, log_every):
    """处理单个样本"""
    
    # 构造消息与图片（使用 data URL）
    user_text = USER_PROMPT_TEMPLATE.format(q=question)
    messages = [
        Message.system_message(SYSTEM_PROMPT),
        Message.user_message(user_text)
    ]

    try:
        # 直接调用LLM，不设置超时
        pred_text = await llm.ask_with_images(
            messages=messages,
            images=[data_url],   # 也可传入多个
            system_msgs=None,
            stream=False,
            temperature=0.01
        )
        
        # 评估结果
        is_correct_pred = is_correct(pred_text, answers, question_type, eval_method, question)
        
        # 统计信息
        method_key = eval_method or "default"
        if method_key not in eval_method_stats:
            eval_method_stats[method_key] = {"count": 0, "correct": 0, "total_score": 0.0}
        eval_method_stats[method_key]["count"] += 1
        if is_correct_pred:
            eval_method_stats[method_key]["correct"] += 1
        eval_method_stats[method_key]["total_score"] += (1.0 if is_correct_pred else 0.0)
        
        type_stats[question_type] = type_stats.get(question_type, 0) + 1
        dataset_stats[dataset_name] = dataset_stats.get(dataset_name, 0) + 1
        
        # 计算当前正确率
        current_correct = sum(stats["correct"] for stats in eval_method_stats.values())
        current_total = sum(stats["count"] for stats in eval_method_stats.values())
        current_accuracy = current_correct / max(1, current_total)
        
        # 获取当前总token消耗（从LLM的token计数器）
        token_summary = llm.token_tracker.get_usage_summary()
        total_tokens = token_summary.get('total_tokens', 0)
        
        # 写入结果
        result = {
            "id": sid,
            "question": question,
            "answers": answers,
            "prediction": pred_text,
            "correct": is_correct_pred,
            "type": question_type,
            "dataset_name": dataset_name,
            "eval_method": eval_method,
            "current_accuracy": current_accuracy,
            "current_total_tokens": total_tokens
        }
        jf.write(json.dumps(result, ensure_ascii=False) + "\n")
        jf.flush()  # 立即写入，避免数据丢失
        
        # 定期输出日志
        if (i + 1) % log_every == 0:
            total_processed = i + 1
            print(f"📊 进度: {total_processed}, 已处理样本")
            
    except Exception as e:
        print(f"❌ 样本 {i} (ID: {sid}) 处理失败: {e}")
        return False
    finally:
        # 强制垃圾回收，释放图片内存
        import gc
        gc.collect()
    
    return True

# ---------- 构造提示词 ----------
SYSTEM_PROMPT = (
    "You are an OCR QA assistant. Read the provided image and answer the user's question. "
    "Answer with the minimal text needed, no extra words or punctuation."
)

USER_PROMPT_TEMPLATE = (
    "Question: {q}\n"
    "Answer briefly with only the exact content found in the image."
)

async def run_infer(
    data_dir: str,
    output_dir: str,
    config_name: str = "default",
    split: str = "train",
    limit: int = -1,
    log_every: int = 100,
    samples_per_type: int = 10,
    max_types: int = 10,
    local_json_file: str = None,
    streaming: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = os.path.join(output_dir, f"ocrbench_pred_{timestamp}.jsonl")
    log_path   = os.path.join(output_dir, f"ocrbench_run_{timestamp}.log")
    
    # 显示所有输出文件路径
    print("="*60)
    print("📁 输出文件路径:")
    print(f"📄 JSONL结果文件: {jsonl_path}")
    print(f"📋 运行日志文件 : {log_path}")
    print(f"📂 输出目录     : {output_dir}")
    print("="*60)

    # 选择数据源：本地JSON文件 或 parquet文件
    if local_json_file and os.path.exists(local_json_file):
        # 强制使用流式处理模式，避免内存溢出
        print(f"🎯 使用流式处理模式，避免内存溢出")
        n_total = limit if limit > 0 else 900  # 假设总共900个样本
        samples = None  # 不预加载
    else:
        # 从parquet文件加载（原始方式）
        if not DATASETS_AVAILABLE:
            raise ImportError("需要安装datasets库来加载parquet文件: pip install datasets")
        
        print(f"📂 从parquet文件加载数据: {data_dir}")
        ds = load_dataset("parquet", data_files=os.path.join(data_dir, "*.parquet"), split=split)
        
        # 按问题类型采样
        if samples_per_type > 0:
            print(f"正在按问题类型采样: 每种类型 {samples_per_type} 个样本，最多 {max_types} 种类型...")
            
            # 按类型分组
            type_groups = {}
            for i, sample in enumerate(ds):
                q_type = sample.get('type', 'unknown')
                if q_type not in type_groups:
                    type_groups[q_type] = []
                type_groups[q_type].append(i)
            
            print(f"发现 {len(type_groups)} 种问题类型:")
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
                print(f"从 {q_type} 类型中采样了 {len(sampled_indices)} 个样本")
            
            # 创建采样后的数据集
            sampled_data = []
            for idx in selected_indices:
                sampled_data.append(ds[idx])
            
            print(f"总共采样了 {len(sampled_data)} 个样本")
            samples = sampled_data
            n_total = len(samples)
        else:
            samples = list(ds)
            n_total = len(samples) if limit < 0 else min(limit, len(samples))

    # 初始化 LLM（用你的配置名；在 config 中选择支持图像的模型）
    llm = LLM(config_name=config_name)
    
    # 记录开始时间和配置到日志文件
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("========== OCRBench 评估开始 ==========\n")
        lf.write(f"开始时间: {start_time}\n")
        lf.write(f"配置名称: {config_name}\n")
        lf.write(f"输出目录: {output_dir}\n")
        lf.write(f"样本限制: {limit if limit > 0 else '无限制'}\n")
        lf.write(f"日志频率: {log_every}\n")
        lf.write(f"每种类型样本数: {samples_per_type}\n")
        lf.write(f"最大类型数: {max_types}\n")
        lf.write(f"流式处理: {'是' if streaming else '否'}\n")
        lf.write(f"本地JSON文件: {local_json_file or '无'}\n")
        lf.write("\n========== 输出文件 ==========\n")
        lf.write(f"JSONL结果文件: {jsonl_path}\n")
        lf.write(f"运行日志文件 : {log_path}\n")
        lf.write("="*50 + "\n")

    # 结果统计
    n_ok, n_fail = 0, 0
    eval_method_stats = {}  # 统计各种评估方法的使用情况
    type_stats = {}  # 统计问题类型
    dataset_stats = {}  # 统计数据集

    # 打开输出文件
    jf = open(jsonl_path, "w", encoding="utf-8")

    # 选择处理方式：流式处理 或 批量处理
    if local_json_file and os.path.exists(local_json_file):
        # 流式处理：逐个读取样本
        sample_iter = stream_json_samples(local_json_file, limit)
        progress_bar = tqdm(sample_iter, desc="Infer (Streaming)")
        
        for i, sample in enumerate(progress_bar):
            # 本地JSON格式
            question: str = sample["question"]
            answers: List[str] = sample["answers"]
            sid = int(sample["id"]) if "id" in sample else i
            question_type: str = sample.get("type", "basic")
            dataset_name: str = sample.get("dataset_name", "unknown")
            eval_method: str = sample.get("eval", None)
            
            # 跳过所有字典相关的题目类型，避免内存溢出
            dict_related_types = [
                "key information extraction cn", "key information extraction en", 
                "key information mapping en", "chart parsing en", "document parsing cn", 
                "document parsing en", "handwritten answer extraction cn", 
                "table parsing cn", "table parsing en"
            ]
            
            if question_type in dict_related_types:
                print(f"⏭️  跳过字典题目 (ID: {sid}) - {question_type}")
                continue
            
            # 从base64还原图片
            img_base64 = sample["image_base64"]
            data_url = base64_to_data_url(img_base64, sample.get("image_format", "PNG"))
            
            # 处理样本...
            success = await process_sample(i, sample, question, answers, sid, question_type, dataset_name, eval_method, data_url, llm, jf, eval_method_stats, type_stats, dataset_stats, log_every)
            
            if success:
                n_ok += 1
            else:
                n_fail += 1
            
            # 每10个样本强制垃圾回收一次
            if (i + 1) % 10 == 0:
                import gc
                gc.collect()
    else:
        # 批量处理：使用预加载的数据（仅用于parquet文件）
        for i in tqdm(range(n_total), desc="Infer"):
            sample = samples[i]

            # 原始datasets格式
            pil_img: Image.Image = sample["image"]  # datasets 会自动解码为 PIL Image
            question: str = sample["question"]
            answers: List[str] = sample["answers"]
            sid = int(sample["id"]) if "id" in sample else i
            question_type: str = sample.get("type", "basic")
            dataset_name: str = sample.get("dataset_name", "unknown")
            eval_method: str = sample.get("eval", None)
            
            # 转换为data URL
            data_url = pil_to_data_url(pil_img, fmt="PNG")
            
            # 处理样本...
            success = await process_sample(i, sample, question, answers, sid, question_type, dataset_name, eval_method, data_url, llm, jf, eval_method_stats, type_stats, dataset_stats, log_every)
            
            if success:
                n_ok += 1
            else:
                n_fail += 1

    jf.close()

    # 最终摘要日志
    final_acc = n_ok / max(1, n_total)
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n========== SUMMARY ==========\n")
        lf.write(f"结束时间: {end_time}\n")
        lf.write(f"总样本数: {n_total}\n")
        lf.write(f"正确数量: {n_ok}\n")
        lf.write(f"错误数量: {n_fail}\n")
        lf.write(f"最终正确率: {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        
        # 评估方法统计
        lf.write("\n========== 评估方法统计 ==========\n")
        for method, stats in eval_method_stats.items():
            method_acc = stats["correct"] / stats["count"] if stats["count"] > 0 else 0
            avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            lf.write(f"{method}: {stats['count']}个样本, 正确率: {method_acc:.4f}, 平均分数: {avg_score:.4f}\n")
        
        lf.write(f"\n========== 输出文件 ==========\n")
        lf.write(f"JSONL结果文件: {jsonl_path}\n")
        lf.write(f"运行日志文件 : {log_path}\n")
        lf.write(f"输出目录     : {output_dir}\n")

    print("\n" + "="*50)
    print("评估完成!")
    print(f"总样本数: {n_total}")
    print(f"正确数量: {n_ok}")
    print(f"错误数量: {n_fail}")
    print(f"最终正确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print("\n评估方法统计:")
    for method, stats in eval_method_stats.items():
        method_acc = stats["correct"] / stats["count"] if stats["count"] > 0 else 0
        avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {method}: {stats['count']}个样本, 正确率: {method_acc:.4f}, 平均分数: {avg_score:.4f}")
    print("="*60)
    print("📁 生成的文件:")
    print(f"📄 JSONL结果文件: {jsonl_path}")
    print(f"📋 运行日志文件 : {log_path}")
    print("="*60)
    
    # 显示文件大小信息
    try:
        jsonl_size = os.path.getsize(jsonl_path) / 1024 / 1024  # MB
        log_size = os.path.getsize(log_path) / 1024  # KB
        print("📊 文件大小:")
        print(f"   JSONL: {jsonl_size:.2f} MB")
        print(f"   LOG  : {log_size:.2f} KB")
    except Exception as e:
        print(f"⚠️ 无法获取文件大小: {e}")
    
    # 使用eval.py进行统一评估
    print("\n" + "="*60)
    print("📊 开始使用eval.py进行统一评估...")
    try:
        # 将JSONL转换为JSON格式，供eval.py使用
        eval_json_path = jsonl_path.replace('.jsonl', '_eval.json')
        convert_jsonl_to_json(jsonl_path, eval_json_path)
        
        # 使用eval.py进行评估
        eval_result_path = jsonl_path.replace('.jsonl', '_eval_result.json')
        process_predictions(eval_json_path, eval_result_path)
        
        print("✅ 统一评估完成!")
        print(f"📄 评估结果文件: {eval_result_path}")
        
        # 读取并显示评估结果摘要
        with open(eval_result_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        print("\n📊 各任务类型评估结果:")
        task_scores = {}
        for item in eval_results:
            task_type = item.get('type', 'unknown')
            score = item.get('score', 0)
            if task_type not in task_scores:
                task_scores[task_type] = {'scores': [], 'count': 0}
            task_scores[task_type]['scores'].append(score)
            task_scores[task_type]['count'] += 1
        
        for task_type, stats in task_scores.items():
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            print(f"  {task_type}: {stats['count']}个样本, 平均分数: {avg_score:.4f}")
        
    except Exception as e:
        print(f"⚠️ 统一评估失败: {e}")
        print("💡 可以使用原始JSONL文件进行后续分析")
    
    print("="*60)

def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    """将JSONL文件转换为JSON格式，供eval.py使用"""
    print(f"🔄 转换JSONL到JSON格式: {jsonl_path} -> {json_path}")
    
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    sample = json.loads(line.strip())
                    # 转换为eval.py期望的格式
                    eval_sample = {
                        "id": sample.get("id", 0),
                        "predict": sample.get("prediction", ""),
                        "answers": sample.get("answers", []),
                        "type": sample.get("type", "basic"),
                        "question": sample.get("question", ""),
                    }
                    
                    # 如果有eval方法，添加进去
                    if "eval_method" in sample:
                        eval_sample["eval"] = sample["eval_method"]
                    
                    samples.append(eval_sample)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 跳过无效JSON行: {e}")
                    continue
    
    # 保存为JSON格式
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成: {len(samples)} 个样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="OCRBench-v2/data",
                        help="Path to the local parquet shards folder (contains *.parquet)")
    parser.add_argument("--output-dir", type=str, default="outputs_ocrbench")
    parser.add_argument("--config-name", type=str, default="default",
                        help="llm config name you use in your config")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=-1, help="limit samples for a quick run; -1 for all")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--samples-per-type", type=int, default=30, help="number of samples per question type")
    parser.add_argument("--max-types", type=int, default=30, help="maximum number of question types to sample")
    parser.add_argument("--local-json-file", type=str, default="ocrbench_local_data.json",
                        help="Path to local JSON data file (优先使用，避免内存溢出)")
    parser.add_argument("--streaming", action="store_true", default=False,
                        help="使用流式处理模式，避免内存溢出")
    args = parser.parse_args()

    import asyncio
    asyncio.run(run_infer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_name=args.config_name,
        split=args.split,
        limit=args.limit,
        log_every=args.log_every,
        samples_per_type=args.samples_per_type,
        max_types=args.max_types,
        local_json_file=args.local_json_file,
        streaming=args.streaming
    ))
