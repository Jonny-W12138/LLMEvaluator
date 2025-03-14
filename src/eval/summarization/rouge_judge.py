from rouge import Rouge
import datetime
import streamlit as st
import json
import pandas as pd
import os
import torch

def rouge_judge(summary_file_path, task, field_mapping, selected_data_path):

    with (st.status("ROUGE judging...", expanded=True) as status):
        st.write("Checking the input data...")

        if (field_mapping["Field Type"] == "Ref Summary").sum() != 1 \
            or (field_mapping['Field Type'] == 'Transcript').sum() != 1:
            st.error("Field mapping must contain one 'Ref Summary' and one 'Transcript' field.")
            return

        ref_summary_field = field_mapping.loc[field_mapping["Field Type"] == "Ref Summary", "Dataset Field"].values[0]

        st.write("Reading data...")

        with open(summary_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summaries_to_judge = pd.DataFrame(data["results"])

        with open(selected_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ref_summaries = pd.DataFrame(data)

        if ref_summaries.shape[0] != summaries_to_judge.shape[0]:
            st.error("The number of reference summaries and summaries to judge must be the same.")
            return

        summaries_to_judge_list = []
        ref_summaries_list = []
        categories = []
        valid_indices = []

        total_num = ref_summaries.shape[0]

        for i in range(total_num):
            summary = summaries_to_judge.iloc[i]["summary"]
            ref = ref_summaries[ref_summary_field].iloc[i]

            # 过滤掉空的 summary
            if isinstance(summary, str) and summary.strip():
                summaries_to_judge_list.append(summary.strip())
                ref_summaries_list.append(ref.strip())
                valid_indices.append(i)
                categories.append(ref_summaries["category"].iloc[i])

        st.write(f"Valid summaries: {len(valid_indices)}/{total_num}")

        if not summaries_to_judge_list:
            st.error("No valid hypothesis summaries found. Check your input data.")
            return

        st.write("Evaluating summaries...")

        rouge = Rouge()

        scores = rouge.get_scores(summaries_to_judge_list, ref_summaries_list, avg=False)

        st.write("Saving results...")

        results = []

        for i in range(total_num):
            if i in valid_indices:
                idx = valid_indices.index(i)  # 获取对应的非空数据索引
                score = scores[idx]
                # 提取 rouge-1, rouge-2, rouge-l 的 f, p, r 值
                rouge_1_f = score["rouge-1"]["f"]
                rouge_1_p = score["rouge-1"]["p"]
                rouge_1_r = score["rouge-1"]["r"]

                rouge_2_f = score["rouge-2"]["f"]
                rouge_2_p = score["rouge-2"]["p"]
                rouge_2_r = score["rouge-2"]["r"]

                rouge_l_f = score["rouge-l"]["f"]
                rouge_l_p = score["rouge-l"]["p"]
                rouge_l_r = score["rouge-l"]["r"]

                results.append({
                    "ref_summary": ref_summaries[ref_summary_field].iloc[i],
                    "summary": summaries_to_judge.iloc[i]["summary"],
                    "category": categories[i],
                    "rouge-1-f": rouge_1_f,
                    "rouge-1-p": rouge_1_p,
                    "rouge-1-r": rouge_1_r,
                    "rouge-2-f": rouge_2_f,
                    "rouge-2-p": rouge_2_p,
                    "rouge-2-r": rouge_2_r,
                    "rouge-l-f": rouge_l_f,
                    "rouge-l-p": rouge_l_p,
                    "rouge-l-r": rouge_l_r,
                    "success": True
                })
            else:
                results.append({
                    "ref_summary": ref_summaries[ref_summary_field].iloc[i],
                    "summary": summaries_to_judge.iloc[i]["summary"],
                    "category": categories[i],
                    "rouge-1-f": None,
                    "rouge-1-p": None,
                    "rouge-1-r": None,
                    "rouge-2-f": None,
                    "rouge-2-p": None,
                    "rouge-2-r": None,
                    "rouge-l-f": None,
                    "rouge-l-p": None,
                    "rouge-l-r": None,
                    "success": False
                })

        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "rouge_results"), exist_ok=True)
        output_file = os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "rouge_results",
                                   f"rouge_results_{current_time}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        status.update(
            label=f"ROUGE judging completed. Results saved to {output_file}",
            state="complete",
            expanded=False
        )
