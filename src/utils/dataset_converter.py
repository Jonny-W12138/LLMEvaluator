import json
import os
import hashlib
import re
from typing import Dict, List, Union, Optional

def convert_cot_dataset(input_file: str, output_file: str, language: str = "en"):
    """
    将新的CoT集合数据集格式转换为旧数据集格式
    
    Args:
        input_file: 输入文件路径(新格式)
        output_file: 输出文件路径(旧格式)
        language: 语言，默认为英文
    """
    print(f"正在转换数据集: {input_file} -> {output_file}")
    
    # 读取新格式数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    
    converted_items = []
    
    # 遍历并转换每个数据项
    for item_id, item in data_dict.items():
        # 创建唯一标识符，使用原始ID或基于任务和内容生成
        qid = item_id if item_id else generate_qid(item.get('source', ''), item.get('task', ''))
        
        # 转换为旧格式
        converted_item = {
            "qid": qid,
            "contexts": [item.get('source', '')],
            "reference": {
                "answer": parse_answer(item.get('target', '')),
                "process": parse_steps(item.get('rationale', ''))
            },
            "category": item.get('task', '未分类')
        }
        
        # 尝试提取难度级别
        level = extract_difficulty_level(item.get('source', ''))
        if level is not None:
            converted_item["level"] = level
        else:
            # 如果无法从问题中提取难度，根据步骤数量估计
            steps = converted_item["reference"]["process"]
            if isinstance(steps, list):
                if len(steps) <= 2:
                    converted_item["level"] = 0
                elif len(steps) <= 4:
                    converted_item["level"] = 1
                elif len(steps) <= 6:
                    converted_item["level"] = 2
                else:
                    converted_item["level"] = 3
            else:
                converted_item["level"] = 0
        
        # 示例字段留空，可以在需要时添加
        converted_item["examples"] = []
        
        converted_items.append(converted_item)
    
    # 写入转换后的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，共转换 {len(converted_items)} 条数据")
    return converted_items

def generate_qid(source: str, task: str) -> str:
    """生成唯一的问题ID"""
    # 使用问题文本和任务类型的哈希创建ID
    combined = (source + task).encode('utf-8')
    hash_obj = hashlib.md5(combined)
    # 返回格式：任务类型_前8位哈希
    task_prefix = re.sub(r'\W+', '_', task.lower()) if task else "general"
    return f"{task_prefix}_{hash_obj.hexdigest()[:8]}"

def parse_answer(target: str) -> List[str]:
    """
    解析目标答案
    解析各种可能的答案格式，将它们转换为字符串列表
    """
    if not target:
        return []
        
    # 尝试解析为JSON
    try:
        # 如果已经是列表格式，直接返回
        if isinstance(target, list):
            return [str(item) for item in target]
            
        # 尝试解析JSON字符串
        if isinstance(target, str):
            # 检查是否是JSON数组字符串
            if target.strip().startswith('[') and target.strip().endswith(']'):
                try:
                    parsed = json.loads(target)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except:
                    pass
                    
            # 检查是否是JSON对象字符串
            if target.strip().startswith('{') and target.strip().endswith('}'):
                try:
                    parsed = json.loads(target)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        # 如果JSON对象包含answer字段
                        answer = parsed["answer"]
                        if isinstance(answer, list):
                            return [str(item) for item in answer]
                        else:
                            return [str(answer)]
                except:
                    pass
    except:
        pass
    
    # 默认情况：将答案作为单个字符串返回
    return [target]

def parse_steps(rationale: str) -> List[str]:
    """
    解析推理步骤
    将原始思维链(CoT)分解为步骤列表
    """
    if not rationale:
        return []
    
    # 尝试从JSON中提取步骤
    try:
        # 尝试查找JSON部分
        json_match = re.search(r'\{.*\}', rationale, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "process" in parsed:
                if isinstance(parsed["process"], list):
                    return parsed["process"]
    except:
        pass
    
    # 尝试使用常见的步骤分隔符分割文本
    # 1. 尝试使用"步骤X:"或"Step X:"格式
    steps = re.findall(r'(?:步骤|Step)\s*\d+[:：]?(.*?)(?=(?:步骤|Step)\s*\d+[:：]|$)', rationale, re.DOTALL | re.IGNORECASE)
    if steps:
        return [step.strip() for step in steps]
    
    # 2. 尝试使用数字+点+空格格式 (如 "1. 第一步")
    steps = re.findall(r'\d+\.\s+(.*?)(?=\d+\.\s+|$)', rationale, re.DOTALL)
    if steps:
        return [step.strip() for step in steps]
    
    # 3. 尝试使用换行符分割
    steps = [line.strip() for line in rationale.split('\n') if line.strip()]
    if len(steps) > 1:
        return steps
    
    # 如果无法识别明确的步骤，将整个推理作为一个步骤
    return [rationale.strip()]

def extract_difficulty_level(text: str) -> Optional[int]:
    """尝试从问题文本中提取难度级别"""
    level_match = re.search(r'[Ll]evel\s*[:：]?\s*(\d)', text)
    if level_match:
        return int(level_match.group(1))
    
    difficulty_match = re.search(r'难度[：:]\s*(\d)', text)
    if difficulty_match:
        return int(difficulty_match.group(1))
    
    return None

if __name__ == "__main__":
    # 使用项目配置中的路径
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # 使用绝对路径
    input_file = os.path.join(ROOT_PATH, "dataset/reasoning/default_light_reasoning_dataset/CoT_collection_light_10.json")
    output_file_en = os.path.join(ROOT_PATH, "dataset/reasoning/default_light_reasoning_dataset/cot_10_en_converted.jsonl")
    output_file_zh = os.path.join(ROOT_PATH, "dataset/reasoning/default_light_reasoning_dataset/cot_10_zh_converted.jsonl")
    
    print(f"输入文件路径: {input_file}")
    print(f"输出文件路径: {output_file_en}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        # 尝试查找文件
        for root, dirs, files in os.walk(os.path.join(ROOT_PATH, "dataset")):
            for file in files:
                if file == "CoT_collection_light_10.json" or "CoT" in file:
                    print(f"找到可能的文件: {os.path.join(root, file)}")
    else:
        # 转换英文数据集
        convert_cot_dataset(input_file, output_file_en, "en")
        # 如果有中文数据集，可以使用相同的函数转换
        # convert_cot_dataset(input_file, output_file_zh, "zh") 