import json
import pandas as pd
from bert_score import score
import os


def evaluate_and_save_to_json(dataframe, json_file_path, reference_columns, output_json_path,
                              custom_model):
    """
    use BertScore to evaluate the model output against the reference summaries and save the results to a JSON file.
    :param dataframe: pd.DataFrame, contains the evaluation dataset with reference summaries.
    :param json_file_path: str, JSON file path containing the model-generated summaries.
    :param reference_columns: list, list of column names in pd.DataFrame storing the reference summaries.
    :param output_json_path: str, JSON file path to save the results.
    :param custom_model: str, custom BertScore evaluation model, default is "bert-base-uncased".
    """

    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    generated_texts = {result["record_index"]: result["generated_text"] for result in json_data["results"]}

    dataframe["generated_text"] = dataframe.index.map(generated_texts.get)

    if dataframe["generated_text"].isnull().any():
        raise ValueError("Some records do not have generated summaries. Check the JSON file.")

    # 合并参考摘要列
    dataframe["reference_texts"] = dataframe[reference_columns].apply(
        lambda x: [text for text in x if pd.notnull(text)], axis=1)

    # 计算 BertScore
    all_generated = dataframe["generated_text"].tolist()
    all_references = dataframe["reference_texts"].tolist()

    # 每条记录有多个参考摘要时，需展开
    all_references_flat = []
    for refs in all_references:
        all_references_flat.extend([refs] * len(refs))

    all_generated_flat = []
    for gen_text, refs in zip(all_generated, all_references):
        all_generated_flat.extend([gen_text] * len(refs))

    # 计算 BertScore
    P, R, F1 = score(
        cands=all_generated_flat,
        refs=all_references_flat,
        model_type=custom_model,
        lang="en",  # 根据摘要语言调整
        verbose=True
    )

    # 保存得分
    dataframe["bertscore_precision"] = P[:len(all_generated)]
    dataframe["bertscore_recall"] = R[:len(all_generated)]
    dataframe["bertscore_f1"] = F1[:len(all_generated)]

    # 构建输出 JSON 数据
    metadata = {
        "selected_data_path": json_data["metadata"]["selected_data_path"],
        "reference_columns": reference_columns,
        "custom_model": custom_model
    }

    results = []
    for idx, row in dataframe.iterrows():
        results.append({
            "record_index": idx,
            "generated_text": row["generated_text"],
            "reference_texts": row["reference_texts"],
            "bertscore_precision": row["bertscore_precision"],
            "bertscore_recall": row["bertscore_recall"],
            "bertscore_f1": row["bertscore_f1"]
        })

    output_data = {
        "metadata": metadata,
        "results": results
    }

    # 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Evaluation result saved to {output_json_path}")


# 示例调用
df = pd.DataFrame({
    "summary_1": ["This is a reference summary.", "Another reference summary."],
    "summary_2": ["Alternate reference summary.", None]
})
json_path = "path_to_json_file.json"
output_path = "path_to_output_file.json"
reference_cols = ["summary_1", "summary_2"]

evaluate_and_save_to_json(df, json_path, reference_cols, output_path, custom_model="microsoft/deberta-v3-large")
