import json


def parse_model_output(model_output):
    """
    解析 model_output 字段，返回 pred_labels, pred_types, pred_quote
    """
    pred_labels, pred_types, pred_quote = [], [], []
    try:
        # 尝试找到 JSON 数组或对象的起始和结束位置
        start_idx = model_output.find('[')
        if start_idx != -1:
            end_idx = model_output.find(']', start_idx)
        else:
            start_idx = model_output.find('{')
            end_idx = model_output.find('}', start_idx)

        if start_idx != -1 and end_idx != -1:
            # 提取 JSON 字符串
            json_str = model_output[start_idx:end_idx + 1]
            # 使用 json.loads 解析 JSON
            output = json.loads(json_str)

            # 处理解析后的数据
            if isinstance(output, list):  # 如果是数组
                for out in output:
                    category = out["category"].replace('\n', '').replace('[', '').replace(']', '')
                    pred_labels.append(0 if category.lower() == "no error" else 1)
                    pred_types.append(category)
                    pred_quote.append(out.get("quote", ""))
            elif isinstance(output, dict):  # 如果是对象
                category = output["category"].replace('\n', '').replace('[', '').replace(']', '')
                pred_labels.append(0 if category.lower() == "no error" else 1)
                pred_types.append(category)
                pred_quote.append(output.get("quote", ""))

            return pred_labels, pred_types, pred_quote, True
        else:
            raise ValueError("No valid JSON array or object found in the input text.")
    except Exception as e:
        print(f'Parsing error: {e}')
        return [], [], [], False


def process_json_file(file_path):
    """
    处理 JSON 文件，重新解析 success 为 false 的记录
    """
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每一条记录
    for record in data:
        if not record["success"]:
            print(f"Processing record {record['record_index']}...")
            # 重新解析 model_output
            pred_labels, pred_types, pred_quote, success = parse_model_output(record["model_output"])
            if success:
                # 替换原有的解析结果
                record["pred_labels"] = pred_labels
                record["pred_types"] = pred_types
                record["pred_quote"] = pred_quote
                record["success"] = success
                print(f"Record {record['record_index']} updated successfully.")
            else:
                print(f"Failed to parse record {record['record_index']}.")

    # 保存更新后的 JSON 文件
    output_file_path = file_path.replace('.json', '_updated.json')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Updated JSON saved to {output_file_path}.")


# 运行脚本
if __name__ == "__main__":
    file_path = "/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/tasks/qwen2.5-14b-instruct-1m/summarization/evaluation/llm_judge/keyfact_check/keyfact_check-2025-03-13_20-18-10.json"  # 替换为你的 JSON 文件路径
    process_json_file(file_path)