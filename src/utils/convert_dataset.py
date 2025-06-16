#!/usr/bin/env python
import os
import sys
import argparse

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(root_dir)

# 导入转换模块
from src.utils.dataset_converter import convert_cot_dataset

def main():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录
    root_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # 默认路径使用绝对路径
    default_input = os.path.join(root_dir, "dataset", "reasoning", "default_light_reasoning_dataset", "CoT_collection_light_10.json")
    default_output = os.path.join(root_dir, "dataset", "reasoning", "default_light_reasoning_dataset", "cot_10_en_converted.jsonl")
    
    parser = argparse.ArgumentParser(description="转换CoT数据集为评估器所需格式")
    parser.add_argument("--input", "-i", type=str, required=False,
                      help="输入JSON文件路径", 
                      default=default_input)
    parser.add_argument("--output", "-o", type=str, required=False,
                      help="输出JSONL文件路径",
                      default=default_output)
    parser.add_argument("--language", "-l", type=str, choices=["en", "zh"], default="en",
                      help="转换语言 (默认: en)")
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在!")
        # 尝试查找文件
        for root, dirs, files in os.walk(os.path.join(root_dir, "dataset")):
            for file in files:
                if file == "CoT_collection_light_10.json" or "CoT" in file:
                    print(f"找到可能的文件: {os.path.join(root, file)}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"语言: {args.language}")
    
    # 执行转换
    try:
        items = convert_cot_dataset(args.input, args.output, args.language)
        print(f"转换成功! 共转换 {len(items)} 条数据")
        return 0
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 