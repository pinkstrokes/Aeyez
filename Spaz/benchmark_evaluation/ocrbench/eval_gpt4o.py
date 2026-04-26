#!/usr/bin/env python3
"""
使用OpenAI GPT-4o-mini评估OCRBench预测结果
- 选择题：严格按照选项字母判断（A/B/C/D）
- 开放题：基于语义相似度判断，意思基本一样即可
"""

import json
import argparse
import os
import sys
import re
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

# 添加项目路径以使用 LLM 类
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from app.llm import LLM


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ 跳过第{line_num}行无效JSON: {e}")
                    continue
    return data


def is_multiple_choice(question: str, answers: List[str]) -> bool:
    """判断是否为选择题（答案是单个字母A/B/C/D）"""
    if not answers:
        return False
    
    # 检查所有答案是否都是单个字母
    for ans in answers:
        ans_stripped = ans.strip().upper()
        if len(ans_stripped) == 1 and ans_stripped in ['A', 'B', 'C', 'D', 'E', 'F']:
            return True
    
    return False


def extract_choice_letter(text: str) -> str:
    """从文本中提取选择题答案字母"""
    text = text.strip().upper()
    
    # 直接匹配单个字母
    if len(text) == 1 and text in ['A', 'B', 'C', 'D', 'E', 'F']:
        return text
    
    # 匹配 "A)", "A.", "A:", "(A)", "【A】" 等格式
    patterns = [
        r'^([A-F])[):\.]',  # A) A: A.
        r'^\(([A-F])\)',     # (A)
        r'^【([A-F])】',     # 【A】
        r'答案[是为]?\s*[：:]\s*([A-F])',  # 答案是：A
        r'^选项?\s*([A-F])',  # 选项A
        r'\b([A-F])\b',      # 独立的字母
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # 如果文本很短且包含字母，尝试提取第一个字母
    if len(text) <= 10:
        for char in text:
            if char in ['A', 'B', 'C', 'D', 'E', 'F']:
                return char
    
    return ""


def evaluate_multiple_choice(prediction: str, answers: List[str]) -> bool:
    """
    评估选择题：严格按照答案字母判断
    """
    # 提取预测的答案字母
    pred_letter = extract_choice_letter(prediction)
    
    if not pred_letter:
        return False
    
    # 检查是否匹配任何标准答案
    for ans in answers:
        ans_letter = extract_choice_letter(ans)
        if pred_letter == ans_letter:
            return True
    
    return False


def create_semantic_evaluation_prompt(question: str, answers: List[str], prediction: str) -> str:
    """创建语义评估提示词（用于开放题）"""
    answers_str = "\n".join([f"- {ans}" for ans in answers])
    
    prompt = f"""你是一个OCR问答评估专家。请判断模型的预测答案是否正确。

问题：{question}

标准答案（以下任意一个都算对）：
{answers_str}

模型预测：{prediction}

评估标准：
1. 如果预测的**核心含义**与标准答案一致，即使表述不同，也判断为正确
2. 如果预测包含标准答案的**关键信息**，允许有合理的格式差异（如大小写、标点、空格等）
3. 对于数字、日期、时间等，允许不同格式但含义相同（如"12:30 PM"和"12:30 p.m."）
4. 如果预测的核心信息明显错误或缺失关键内容，判断为错误


请严格按照以下格式回答，只输出一个词：
- 如果正确，回答：CORRECT
- 如果错误，回答：INCORRECT

你的判断："""

    return prompt


def call_gpt4o_mini(prompt: str, llm: LLM) -> str:
    """调用OpenAI GPT-4o-mini API（使用项目LLM类）"""
    try:
        # 使用项目的 LLM 类进行同步调用
        from app.llm import Message
        
        messages = [
            Message.system_message("You are a precise evaluation assistant."),
            Message.user_message(prompt)
        ]
        
        # 同步调用（如果 ask 是 async，需要使用 asyncio.run）
        import asyncio
        response = asyncio.run(llm.ask(messages, temperature=0.0))
        
        return response.strip()
        
    except Exception as e:
        print(f"❌ API调用错误: {e}")
        return ""


def evaluate_semantic(question: str, answers: List[str], prediction: str, llm: LLM) -> bool:
    """使用GPT-4o-mini进行语义评估（开放题）"""
    try:
        prompt = create_semantic_evaluation_prompt(question, answers, prediction)
        response = call_gpt4o_mini(prompt, llm)
        
        if not response:
            print("⚠️ 模型响应为空")
            return False
        
        # 解析响应
        response_upper = response.upper().strip()
        
        if "CORRECT" in response_upper:
            return True
        elif "INCORRECT" in response_upper:
            return False
        else:
            # 如果响应不明确，再尝试判断
            print(f"⚠️ 模型响应不明确: {response}")
            return False
            
    except Exception as e:
        print(f"❌ 评估错误: {e}")
        return False


def evaluate_sample(sample: Dict[str, Any], llm: LLM) -> Dict[str, Any]:
    """评估单个样本"""
    question = sample.get("question", "")
    answers = sample.get("answers", [])
    prediction = sample.get("prediction", "")
    sample_type = sample.get("type", "unknown")
    
    if not question or not answers or not prediction:
        return {
            "is_correct": False,
            "eval_method": "invalid",
            "error": "数据不完整"
        }
    
    # 判断题目类型
    is_choice = is_multiple_choice(question, answers)
    
    if is_choice:
        # 选择题：严格按字母判断
        is_correct = evaluate_multiple_choice(prediction, answers)
        eval_method = "multiple_choice_strict"
    else:
        # 开放题：语义判断
        is_correct = evaluate_semantic(question, answers, prediction, llm)
        eval_method = "semantic_similarity"
    
    return {
        "is_correct": is_correct,
        "eval_method": eval_method,
        "is_choice_question": is_choice,
        "extracted_prediction": extract_choice_letter(prediction) if is_choice else prediction
    }


def main():
    parser = argparse.ArgumentParser(description="使用GPT-4o-mini评估OCRBench预测结果")
    parser.add_argument("--input", type=str, required=True, help="输入的JSONL文件路径")
    parser.add_argument("--output", type=str, help="输出结果文件路径（可选）")
    parser.add_argument("--config-name", type=str, default="gpt4o_mini", 
                       help="LLM配置名称（使用config.toml中的配置，默认: gpt4o_mini）")
    parser.add_argument("--max-samples", type=int, help="最大评估样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 初始化LLM（使用config.toml中的配置）
    try:
        llm = LLM(config_name=args.config_name)
        print(f"✅ 成功加载LLM配置: {args.config_name}")
    except Exception as e:
        print(f"❌ 加载LLM配置失败: {e}")
        return
    
    print(f"🚀 开始使用GPT-4o-mini评估: {args.input}")
    print(f"📝 使用配置: {args.config_name}")
    
    # 加载数据
    print("📖 加载数据...")
    data = load_jsonl_data(args.input)
    print(f"📊 总共加载 {len(data)} 个样本")
    
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"🔬 限制评估样本数: {len(data)}")
    
    # 开始评估
    print("\n" + "="*60)
    print("🔍 开始评估...")
    print("="*60)
    
    correct_count = 0
    total_count = len(data)
    results = []
    
    # 统计不同评估方法的结果
    choice_correct = 0
    choice_total = 0
    semantic_correct = 0
    semantic_total = 0
    
    for i, sample in enumerate(tqdm(data, desc="评估进度"), 1):
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        prediction = sample.get("prediction", "")
        
        # 评估样本
        eval_result = evaluate_sample(sample, llm)
        is_correct = eval_result["is_correct"]
        
        if is_correct:
            correct_count += 1
        
        # 统计不同类型的准确率
        if eval_result.get("is_choice_question"):
            choice_total += 1
            if is_correct:
                choice_correct += 1
        else:
            semantic_total += 1
            if is_correct:
                semantic_correct += 1
        
        # 保存结果
        result = {
            "id": sample.get("id", i-1),
            "question": question[:100] + "..." if len(question) > 100 else question,
            "answers": answers,
            "prediction": prediction,
            "is_correct": is_correct,
            "eval_method": eval_result["eval_method"],
            "is_choice_question": eval_result.get("is_choice_question", False),
            "type": sample.get("type", "unknown"),
            "original_correct": sample.get("correct", None)
        }
        results.append(result)
        
        # 每50个样本显示一次进度
        if i % 50 == 0:
            current_accuracy = correct_count / i
            print(f"\n📈 样本 {i}/{total_count} - 当前准确率: {current_accuracy:.4f} ({correct_count}/{i})")
    
    # 最终统计
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    choice_accuracy = choice_correct / choice_total if choice_total > 0 else 0
    semantic_accuracy = semantic_correct / semantic_total if semantic_total > 0 else 0
    
    print("\n" + "="*60)
    print("📊 评估完成!")
    print("="*60)
    print(f"✅ 总体正确: {correct_count}/{total_count} = {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"\n📋 分类统计:")
    print(f"   🔤 选择题: {choice_correct}/{choice_total} = {choice_accuracy:.4f} ({choice_accuracy*100:.2f}%)")
    print(f"   💬 开放题: {semantic_correct}/{semantic_total} = {semantic_accuracy:.4f} ({semantic_accuracy*100:.2f}%)")
    
    # 保存结果
    if args.output:
        output_data = {
            "summary": {
                "total_samples": total_count,
                "correct_count": correct_count,
                "accuracy": final_accuracy,
                "choice_questions": {
                    "total": choice_total,
                    "correct": choice_correct,
                    "accuracy": choice_accuracy
                },
                "semantic_questions": {
                    "total": semantic_total,
                    "correct": semantic_correct,
                    "accuracy": semantic_accuracy
                },
                "model": "gpt-4o-mini",
                "eval_strategy": {
                    "choice": "strict letter matching (A/B/C/D)",
                    "semantic": "meaning-based similarity with GPT-4o-mini"
                }
            },
            "results": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {args.output}")
    
    # 与原结果对比（如果有的话）
    original_correct = sum(1 for r in results if r.get("original_correct") is True)
    if original_correct > 0:
        original_accuracy = original_correct / total_count
        print("\n🔄 与原结果对比:")
        print(f"   原准确率: {original_accuracy:.4f} ({original_correct}/{total_count})")
        print(f"   GPT-4o-mini评估: {final_accuracy:.4f} ({correct_count}/{total_count})")
        print(f"   差异: {abs(final_accuracy - original_accuracy):.4f}")


if __name__ == "__main__":
    main()

