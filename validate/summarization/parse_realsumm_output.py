import json
import re


def parse_judge_output(judge_output_str):
    # 提取judge_output中的JSON部分
    json_str = re.search(r'```json\n([\s\S]*?)\n```', judge_output_str).group(1)
    return json.loads(json_str)


def process_json_line(json_line):
    # 解析原始JSON行
    data = json.loads(json_line)

    # 解析judge_output
    judge_output = parse_judge_output(data["judge_output"])
    key_facts = data["key_facts"]
    sentences = data["sentences"]

    # 初始化key_fact_labels
    key_fact_labels = [0] * len(key_facts)

    # 检查每个key_fact是否被标记为"Yes"
    for item in judge_output:
        try:
            idx = key_facts.index(item["key fact"])
            if item["response"] == "Yes":
                key_fact_labels[idx] = 1
        except ValueError:
            # 如果key fact不在key_facts列表中，跳过
            continue

    # 初始化sentence_labels
    sentence_labels = [0] * len(sentences)

    # 检查每条句子是否被命中
    for item in judge_output:
        if "line number" in item and item["line number"]:
            for line_num in item["line number"]:
                # line number是从1开始的，转换为0-based索引
                sentence_idx = line_num - 1
                if 0 <= sentence_idx < len(sentences):
                    sentence_labels[sentence_idx] = 1

    # 添加parsed_judge字段
    data["parsed_judge"] = {
        "key_fact_labels": key_fact_labels,
        "sentence_labels": sentence_labels
    }

    return json.dumps(data, ensure_ascii=False)


def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():  # 跳过空行
                processed_line = process_json_line(line.strip())
                outfile.write(processed_line + '\n')


# 使用示例
input_file = '/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/validate/summarization/output/realsumm-500-output.json'  # 替换为您的输入文件路径
output_file = '/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/validate/summarization/output/realsumm-500-output-parsed.json'  # 替换为您想要的输出文件路径
process_json_file(input_file, output_file)