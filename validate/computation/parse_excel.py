import pandas as pd
import re
from scipy.stats import pearsonr, spearmanr

def parse_nonempty_step_categories(file_path, col1, col2):
    df = pd.read_excel(file_path)

    results = []

    def extract_categories(cell):
        """从一个单元格中提取 Positive / Neutral / Negative"""
        categories = []
        if pd.isna(cell):
            return categories
        lines = str(cell).split('\n')
        for line in lines:
            match = re.search(r':\s*(Positive|Negative|Neutral)', line)
            if match:
                categories.append(match.group(1))
        return categories

    for idx, row in df.iterrows():
        cell1, cell2 = row.get(col1), row.get(col2)
        # 如果两个都为空，跳过
        if pd.isna(cell1) and pd.isna(cell2):
            continue

        parsed_row = {
            "row_index": idx + 2,  # 加2是因为Excel通常从第2行开始有数据
            col1: extract_categories(cell1),
            col2: extract_categories(cell2)
        }
        results.append(parsed_row)

    return results

def compute_scores(parsed_rows, col1, col2):
    all_scores = []

    for row in parsed_rows:
        for col in [col1, col2]:
            categories = row[col]
            if not categories:
                continue  # 跳过空的

            total = len(categories)
            positive = categories.count("Positive")
            neutral = categories.count("Neutral")

            validity = (positive + neutral) / total
            conciseness = neutral / total

            all_scores.append({
                "model": col,
                "row_index": row["row_index"],
                "validity_score": validity,
                "conciseness_score": conciseness
            })

    # 转换为 DataFrame，便于计算平均值和后续展示
    score_df = pd.DataFrame(all_scores)

    # 计算每个模型的平均得分
    avg_scores = score_df.groupby("model")[["validity_score", "conciseness_score"]].mean()

    return score_df, avg_scores

def compute_correlations(score_df, model1, model2):
    # 只保留需要的两组数据
    v1 = score_df[score_df["model"] == model1].sort_values("row_index")["validity_score"].values
    v2 = score_df[score_df["model"] == model2].sort_values("row_index")["validity_score"].values

    c1 = score_df[score_df["model"] == model1].sort_values("row_index")["conciseness_score"].values
    c2 = score_df[score_df["model"] == model2].sort_values("row_index")["conciseness_score"].values

    # 确保对齐行数
    if len(v1) != len(v2) or len(c1) != len(c2):
        raise ValueError("两模型的行数不一致，请检查数据对齐")

    # 皮尔森 & 斯皮尔曼系数
    corr_results = {
        "validity_pearson": pearsonr(v1, v2)[0],
        "validity_spearman": spearmanr(v1, v2)[0],
        "conciseness_pearson": pearsonr(c1, c2)[0],
        "conciseness_spearman": spearmanr(c1, c2)[0],
    }

    return corr_results


file_path = "merged_evaluation_output.xlsx"  # 你的Excel路径
# 要分析的列组（模型 vs Human）
comparison_models = ["deepseek-v3-steps", "qwen-plus-steps", "qwen-max-steps"]
col_human = "Human"

# 结果收集
all_avg_scores = {}
all_correlations = {}

for col_model in comparison_models:
    print(f"\n📊 对比模型: {col_model} vs {col_human}")

    # 解析两列非空行
    parsed_rows = parse_nonempty_step_categories(file_path, col_model, col_human)

    # 计算每一行的得分
    score_df, avg_scores = compute_scores(parsed_rows, col_model, col_human)
    all_avg_scores[col_model] = avg_scores

    # 打印每一行得分样例
    print("各行得分示例：")
    print(score_df.head())

    # 打印平均得分
    print("\n模型平均得分：")
    print(avg_scores)

    # 计算皮尔森 & 斯皮尔曼
    correlations = compute_correlations(score_df, col_model, col_human)
    all_correlations[col_model] = correlations

    # 打印相关性结果
    print("\n模型得分相关性分析：")
    for k, v in correlations.items():
        print(f"{k}: {v:.4f}")
