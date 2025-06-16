import json
import os
import pandas as pd
from collections import defaultdict
from glob import glob

# 替换为你的 JSON 文件路径列表或通配符路径
json_files = ["/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/tasks/deepseekv3/computation/math/evaluation/math_evaluation_2025-03-30_11-17-36.json",
              "/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/tasks/deepseekv3/computation/math/evaluation/math_evaluation_2025-04-04_11-15-21.json",
              "/Users/jonnyw/Documents/nju/graduation_proj/LLMEvaluator/tasks/deepseekv3/computation/math/evaluation/math_evaluation_2025-04-04_17-28-47.json"]


problems_data = defaultdict(lambda: {"llm_output": None, "evaluations": {}})

# 读取所有 JSON 文件
for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    judge_model = data.get("metadata", {}).get("model_engine", os.path.basename(file_path))
    evaluations = data.get("evaluation", [])

    for entry in evaluations:
        problem = entry.get("origin_response").get("problem")
        llm_output = entry.get("origin_response").get("llm_output")
        eval_steps = entry.get("evaluation").get("eval_steps", {})
        if_correct = entry.get("evaluation").get("if_correct")

        if problem == "" or llm_output == "" or eval_steps == {} or if_correct == "":
            continue

        if problems_data[problem]["llm_output"] is None:
            problems_data[problem]["llm_output"] = llm_output

        problems_data[problem]["evaluations"][judge_model] = {
            "eval_steps": eval_steps,
            "if_correct": if_correct
        }

# 构建最终 DataFrame
rows = []
for problem, pdata in problems_data.items():
    row = {
        "problem": problem,
        "llm_output": pdata["llm_output"]
    }

    # 收集所有模型名
    all_models = list(pdata["evaluations"].keys())

    # 收集所有出现过的步骤编号
    all_steps = set()
    for model_eval in pdata["evaluations"].values():
        all_steps.update(model_eval["eval_steps"].keys())

    # 判断哪些步骤在不同模型中有不同的categorie
    differing_steps = []
    for step in sorted(all_steps, key=int):
        categories = set()
        for model in all_models:
            model_steps = pdata["evaluations"].get(model, {}).get("eval_steps", {})
            step_info = model_steps.get(step)
            if step_info:
                cat = step_info.get("categorie")
                if cat == None:
                    cat = step_info.get("category")
                categories.add(cat)
        if len(categories) > 1:
            differing_steps.append(step)

    # 添加每个模型的 steps 和 correct 字段
    for model in all_models:
        eval_info = pdata["evaluations"][model]
        steps = eval_info.get("eval_steps", {})
        correct = eval_info.get("if_correct", "")

        step_lines = []
        for step_num in sorted(steps.keys(), key=int):
            step_data = steps[step_num]
            cat = step_data.get("categorie", "")
            if cat == "":
                cat = step_data.get("category", "")

            ref = step_data.get("reference step", "")
            sub_cat = step_data.get("sub-categories", "")
            if cat == "Neutral" or cat == "Negative":
                step_lines.append(f'{step_num}: {cat} ({sub_cat}) - {ref}')
            else:
                step_lines.append(f'{step_num}: {cat} - {ref}')
        row[f"{model}-steps"] = "\n".join(step_lines)
        row[f"{model}-correct"] = correct

    # 标注差异步骤
    row["different_steps"] = ", ".join(differing_steps) if differing_steps else "一致"

    rows.append(row)

# 转换为 DataFrame
df = pd.DataFrame(rows)

# 保存为 Excel 文件
df.to_excel("merged_evaluation_output.xlsx", index=False)
