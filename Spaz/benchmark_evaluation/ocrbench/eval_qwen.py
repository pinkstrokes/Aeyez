#!/usr/bin/env python3
"""
使用Qwen32B模型评估OCRBench预测结果的脚本
比较[answers]和[prediction]，输出正确/错误，并统计正确率
直接使用DashScope API，不依赖项目LLM类
"""

import json
import argparse
import sys
import os
import requests
from typing import List, Dict, Any


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


def create_evaluation_prompt(question: str, answers: List[str], prediction: str) -> str:
    """创建评估提示词"""
    answers_str = ", ".join([f'"{ans}"' for ans in answers])
    
    prompt = f"""请评估以下问答对的正确性：


标准答案: {answers_str}

模型预测: "{prediction}"

请仔细比较模型预测和标准答案，判断预测是否正确。

评估标准：
1. 如果预测与任何一个标准答案完全匹配，则正确
2. 如果预测与标准答案在语义上等价（如"12:42 PM"和"12:42 p.m."），则正确
3. 如果预测包含标准答案的关键信息且格式合理，则正确
4. 如果是选择题，严格按字母判断
5. 其他情况则错误

请只回答"正确"或"错误"，不要解释原因。"""

    return prompt


def call_qwen32b_api(prompt: str, api_key: str) -> str:
    """调用Qwen32B DashScope API"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "qwen2.5-vl-32b-instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 0.001,
        "top_k": 1
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API调用错误: {e}")
        return ""
    except (KeyError, IndexError) as e:
        print(f"❌ 响应解析错误: {e}")
        return ""


def evaluate_with_qwen(question: str, answers: List[str], prediction: str, api_key: str) -> bool:
    """使用Qwen32B模型评估单个样本"""
    try:
        prompt = create_evaluation_prompt(question, answers, prediction)
        
        # 调用Qwen32B API
        response = call_qwen32b_api(prompt, api_key)
        
        if not response:
            print("⚠️ 模型响应为空")
            return False
        
        # 解析响应
        response_text = response.lower()
        
        # 判断正确性
        if "正确" in response_text or "correct" in response_text or "true" in response_text:
            return True
        elif "错误" in response_text or "incorrect" in response_text or "false" in response_text:
            return False
        else:
            # 如果响应不明确，尝试从内容判断
            print(f"⚠️ 模型响应不明确: {response}")
            return False
            
    except (ConnectionError, TimeoutError, ValueError) as e:
        print(f"❌ 评估错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="使用Qwen32B评估OCRBench预测结果")
    parser.add_argument("--input", type=str, required=True, help="输入的JSONL文件路径")
    parser.add_argument("--output", type=str, help="输出结果文件路径（可选）")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", ""),
                       help="DashScope API密钥")
    parser.add_argument("--max-samples", type=int, help="最大评估样本数（用于测试）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    print(f"🚀 开始使用Qwen32B评估: {args.input}")
    print(f"🔑 使用API密钥: {args.api_key[:10]}...")
    
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
    
    for i, sample in enumerate(data, 1):
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        prediction = sample.get("prediction", "")
        
        if not question or not answers or not prediction:
            print(f"⚠️ 样本 {i} 数据不完整，跳过")
            continue
        
        print(f"\n📝 样本 {i}/{total_count}")
        print(f"问题: {question}")
        print(f"标准答案: {answers}")
        print(f"模型预测: {prediction}")
        
        # 使用Qwen32B评估
        is_correct = evaluate_with_qwen(question, answers, prediction, args.api_key)
        
        if is_correct:
            print("✅ 正确")
            correct_count += 1
        else:
            print("❌ 错误")
        
        # 保存结果
        result = {
            "id": sample.get("id", i-1),
            "question": question,
            "answers": answers,
            "prediction": prediction,
            "qwen_evaluation": is_correct,
            "original_correct": sample.get("correct", None)
        }
        results.append(result)
        
        # 显示当前准确率
        current_accuracy = correct_count / i
        print(f"📈 当前准确率: {current_accuracy:.4f} ({correct_count}/{i})")
    
    # 最终统计
    final_accuracy = correct_count / total_count
    
    print("\n" + "="*60)
    print("📊 评估完成!")
    print("="*60)
    print(f"✅ 正确: {correct_count}")
    print(f"❌ 错误: {total_count - correct_count}")
    print(f"📈 总准确率: {final_accuracy:.4f} ({correct_count}/{total_count})")
    
    # 保存结果
    if args.output:
        output_data = {
            "summary": {
                "total_samples": total_count,
                "correct_count": correct_count,
                "accuracy": final_accuracy,
                "model": "qwen2.5-vl-32b-instruct"
            },
            "results": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print("💾 结果已保存到: " + args.output)
    
    # 与原结果对比（如果有的话）
    original_correct = sum(1 for r in results if r.get("original_correct") is True)
    if original_correct > 0:
        original_accuracy = original_correct / total_count
        print("\n🔄 与原结果对比:")
        print(f"   原准确率: {original_accuracy:.4f} ({original_correct}/{total_count})")
        print(f"   Qwen评估: {final_accuracy:.4f} ({correct_count}/{total_count})")
        print(f"   差异: {abs(final_accuracy - original_accuracy):.4f}")


if __name__ == "__main__":
    main()
