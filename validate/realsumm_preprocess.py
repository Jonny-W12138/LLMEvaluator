import json


def process_json_files(file_a_path, file_b_path, file_c_path):
    # 1. 读取文件A的所有记录
    with open(file_a_path, 'r', encoding='utf-8') as f_a:
        records_a = [json.loads(line) for line in f_a]

    # 2. 读取文件B的所有记录并按doc_id分组
    doc_id_to_records_b = {}
    with open(file_b_path, 'r', encoding='utf-8') as f_b:
        for line in f_b:
            record = json.loads(line)
            doc_id = record.get('doc_id')
            if doc_id not in doc_id_to_records_b:
                doc_id_to_records_b[doc_id] = []
            doc_id_to_records_b[doc_id].append(record)

    # 3. 处理每条A记录，查找匹配的B记录，生成新记录
    results = []
    for record_a in records_a:
        doc_id = record_a.get('doc_id')
        matched_records_b = doc_id_to_records_b.get(doc_id, [])[:5]  # 最多取5条匹配记录

        # 对每条匹配的B记录，创建新记录
        for record_b in matched_records_b:
            new_record = record_b.copy()  # 复制A的所有字段
            new_record['key_facts'] = record_a['key_facts']
            results.append(new_record)

    # 4. 将结果写入文件C
    with open(file_c_path, 'w', encoding='utf-8') as f_c:
        for result in results:
            f_c.write(json.dumps(result, ensure_ascii=False) + '\n')

# 使用示例
file_a_path = '/Users/jonnyw/Documents/nju/graduation_proj/opencodes/summarization/FineSurE-ACL24/dataset/realsumm/human-keyfact-list.json'  # 替换为你的文件A路径
file_b_path = '/Users/jonnyw/Documents/nju/graduation_proj/opencodes/summarization/FineSurE-ACL24/reproduce/results/realsumm-result-by-gpt4-w-finesure.json'  # 替换为你的文件B路径
file_c_path = '/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/validate/summarization/data/realsumm-500.json'  # 替换为你想要保存的结果文件路径

process_json_files(file_a_path, file_b_path, file_c_path)