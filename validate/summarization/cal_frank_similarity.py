import json
import numpy as np
from collections import defaultdict

# 大模型类别与人工类别的映射
type_mapping = {
    "RelE": "predicate error",
    "EntE": "entity error",
    "CircE": "circumstantial error",
    "CorefE": "coreference error",
    "LinkE": "linking error",
    "OutE": "out of context error",
    "GramE": "grammatical error",
    "NoE": "no error",
    "OtherE": "other error"
}

# 所有可能的错误类型
all_error_types = set(type_mapping.values())


def parse_judge_output(judge_output):
    """解析大模型评估结果，提取错误类别"""
    try:
        judge_data = json.loads(judge_output.strip('```json\n'))
        return set(entry['category'].replace('-', ' ') for entry in judge_data)
    except Exception as e:
        print(f"解析 judge_output 失败: {e}")
        return set()


def parse_annotations(raw_annotations):
    """解析人工标注结果，提取错误类别"""
    annotator_labels = {}
    for key in ["annotator_0", "annotator_1", "annotator_2"]:
        if key in raw_annotations:
            types = set(type_mapping.get(label, label) for label in raw_annotations[key]["factuality_types"])
            annotator_labels[key] = types
        else:
            annotator_labels[key] = set()
    return annotator_labels


def get_majority_error_types(annotator_labels, exclude_key=None):
    """获取多数人同意的错误类型（排除指定的标注者）"""
    # 收集所有标注者（排除指定的）
    remaining_annotators = [k for k in annotator_labels.keys() if k != exclude_key]

    # 统计每个错误类型的出现次数
    error_counts = defaultdict(int)
    for annotator in remaining_annotators:
        for error_type in annotator_labels[annotator]:
            error_counts[error_type] += 1

    if not error_counts:
        return set()

    # 找出出现次数最多的错误类型
    max_count = max(error_counts.values())
    majority_errors = {k for k, v in error_counts.items() if v == max_count}

    return majority_errors


def jaccard_similarity(set1, set2):
    """计算 Jaccard 相似度"""
    if not set1 and not set2:
        return 1.0  # 两者都为空，认为完全匹配
    if not set1 or not set2:
        return 0.0  # 其中一个为空，则相似度为 0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def calculate_correctness(stats, jaccard_values, set1, set2, group_name):
    """计算正确率和 Jaccard 相似度"""
    stats[group_name]['total_predictions'] += 1
    if set1 & set2:  # 只要有一个共同错误类型就算正确
        stats[group_name]['correct_predictions'] += 1

    # 计算 Jaccard
    jaccard_values[group_name].append(jaccard_similarity(set1, set2))


def print_accuracy_stats(stats, title):
    """打印准确率统计结果"""
    correct = stats['correct_predictions']
    total = stats['total_predictions']
    accuracy = correct / total if total > 0 else np.nan

    print(f"\n{title}:")
    print(f"正确率: {accuracy:.4f} (总预测次数: {total}, 正确预测次数: {correct})")
    return accuracy


def print_jaccard_stats(jaccard_values, title):
    """打印 Jaccard 相似度统计结果"""
    avg_jaccard = np.nanmean(jaccard_values) if jaccard_values else np.nan
    print(f"\n{title} 的平均 Jaccard 相似度: {avg_jaccard:.4f}")
    return avg_jaccard


def print_error_type_stats(error_type_stats):
    """打印每种错误类型的预测情况"""
    print("\n模型在各错误类型下的预测情况:")
    print("{:<25} {:<10} {:<10} {:<10} {:<10}".format(
        "错误类型", "总出现次数", "正确预测", "错误预测", "准确率"))

    for error_type in sorted(all_error_types):
        stats = error_type_stats[error_type]
        total = stats['total']
        correct = stats['correct']
        accuracy = correct / total if total > 0 else 0.0

        print("{:<25} {:<10} {:<10} {:<10} {:.4f}".format(
            error_type, total, correct, total - correct, accuracy))


def main(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 初始化统计字典
    model_stats = {
        "vs_majority": {'total_predictions': 0, 'correct_predictions': 0}
    }

    human_stats = {
        "annotator_0_vs_majority": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_1_vs_majority": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_2_vs_majority": {'total_predictions': 0, 'correct_predictions': 0},
        # 新增两两比较统计
        "annotator_0_vs_1": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_0_vs_2": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_1_vs_0": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_1_vs_2": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_2_vs_0": {'total_predictions': 0, 'correct_predictions': 0},
        "annotator_2_vs_1": {'total_predictions': 0, 'correct_predictions': 0}
    }

    model_jaccard = {key: [] for key in model_stats}
    human_jaccard = {key: [] for key in human_stats}

    # 初始化错误类型统计
    error_type_stats = {error_type: {'total': 0, 'correct': 0}
                        for error_type in all_error_types}

    for entry in data:
        model_labels = parse_judge_output(entry["judge_output"])
        annotator_labels = parse_annotations(entry["raw_annotations"])

        # 获取多数人同意的错误类型（排除模型）
        majority_errors = get_majority_error_types(annotator_labels)

        # 计算模型正确率 & Jaccard
        calculate_correctness(model_stats, model_jaccard, model_labels,
                              majority_errors, "vs_majority")

        # 计算人工正确率 & Jaccard (与多数人比较)
        for annotator in ["annotator_0", "annotator_1", "annotator_2"]:
            if annotator in annotator_labels:
                majority = get_majority_error_types(annotator_labels, exclude_key=annotator)
                key = f"{annotator}_vs_majority"
                calculate_correctness(human_stats, human_jaccard,
                                      annotator_labels[annotator], majority, key)

        # 计算人工两两之间的准确率
        if "annotator_0" in annotator_labels and "annotator_1" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_0"],
                                  annotator_labels["annotator_1"],
                                  "annotator_0_vs_1")

        if "annotator_0" in annotator_labels and "annotator_2" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_0"],
                                  annotator_labels["annotator_2"],
                                  "annotator_0_vs_2")

        if "annotator_1" in annotator_labels and "annotator_0" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_1"],
                                  annotator_labels["annotator_0"],
                                  "annotator_1_vs_0")

        if "annotator_1" in annotator_labels and "annotator_2" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_1"],
                                  annotator_labels["annotator_2"],
                                  "annotator_1_vs_2")

        if "annotator_2" in annotator_labels and "annotator_0" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_2"],
                                  annotator_labels["annotator_0"],
                                  "annotator_2_vs_0")

        if "annotator_2" in annotator_labels and "annotator_1" in annotator_labels:
            calculate_correctness(human_stats, human_jaccard,
                                  annotator_labels["annotator_2"],
                                  annotator_labels["annotator_1"],
                                  "annotator_2_vs_1")

        # 统计每种错误类型的预测情况
        for error_type in majority_errors:
            error_type_stats[error_type]['total'] += 1
            if error_type in model_labels:
                error_type_stats[error_type]['correct'] += 1

    # 打印模型正确率 & Jaccard
    model_accs = [print_accuracy_stats(model_stats[key], f"模型 {key} 正确率") for key in model_stats]
    model_avg_acc = np.nanmean(model_accs)
    print(f"\n模型平均正确率: {model_avg_acc:.4f}")

    model_jaccards = [print_jaccard_stats(model_jaccard[key], f"模型 {key} Jaccard") for key in model_jaccard]
    model_avg_jaccard = np.nanmean(model_jaccards)
    print(f"\n模型 Jaccard 平均相似度: {model_avg_jaccard:.4f}")

    # 打印人工正确率 & Jaccard (与多数人比较)
    print("\n人工标注者与多数人比较:")
    human_accs = [print_accuracy_stats(human_stats[key], f"人工 {key} 正确率")
                  for key in ["annotator_0_vs_majority", "annotator_1_vs_majority", "annotator_2_vs_majority"]]
    human_avg_acc = np.nanmean(human_accs)
    print(f"\n人工与多数人比较的平均正确率: {human_avg_acc:.4f}")

    human_jaccards = [print_jaccard_stats(human_jaccard[key], f"人工 {key} Jaccard")
                      for key in ["annotator_0_vs_majority", "annotator_1_vs_majority", "annotator_2_vs_majority"]]
    human_avg_jaccard = np.nanmean(human_jaccards)
    print(f"\n人工与多数人比较的 Jaccard 平均相似度: {human_avg_jaccard:.4f}")

    # 打印人工两两之间的准确率
    print("\n人工标注者两两比较:")
    pairwise_accs = [print_accuracy_stats(human_stats[key], f"人工 {key} 正确率")
                     for key in ["annotator_0_vs_1", "annotator_0_vs_2",
                                 "annotator_1_vs_0", "annotator_1_vs_2",
                                 "annotator_2_vs_0", "annotator_2_vs_1"]]
    pairwise_avg_acc = np.nanmean(pairwise_accs)
    print(f"\n人工两两比较的平均正确率: {pairwise_avg_acc:.4f}")

    pairwise_jaccards = [print_jaccard_stats(human_jaccard[key], f"人工 {key} Jaccard")
                         for key in ["annotator_0_vs_1", "annotator_0_vs_2",
                                     "annotator_1_vs_0", "annotator_1_vs_2",
                                     "annotator_2_vs_0", "annotator_2_vs_1"]]
    pairwise_avg_jaccard = np.nanmean(pairwise_jaccards)
    print(f"\n人工两两比较的 Jaccard 平均相似度: {pairwise_avg_jaccard:.4f}")

    # 打印错误类型统计
    print_error_type_stats(error_type_stats)


if __name__ == "__main__":
    main("/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/validate/summarization/output/frank_500_output.json")