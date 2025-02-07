from rouge import Rouge
import datetime
import streamlit as st
import json
import pandas as pd
import os
import torch


def rouge_judge(summary_file_path, task, field_mapping, selected_data_path):

    with st.status("ROUGE judging...", expanded=True) as status:
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

        total_num = ref_summaries.shape[0]

        for i in range(total_num):
            summaries_to_judge_list.append(summaries_to_judge.iloc[i]["summary"].replace("\n", ""))
            ref_summaries_list.append(ref_summaries[ref_summary_field].iloc[i])

        hyps, refs = map(list, zip(*[[summary, ref] for summary, ref in zip(summaries_to_judge_list, ref_summaries_list)]))
        st.write("Evaluating summaries...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rouge = Rouge()

        scores = rouge.get_scores(hyps, refs, avg=False)

        st.write("Saving results...")

        results = []

        for i in range(total_num):
            results.append({
                "ref_summary": ref_summaries_list[i],
                "summary": summaries_to_judge_list[i],
                "rouge_score": scores[i]
            })

        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", f"rouge_results_{current_time}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        status.update(
            label=f"ROUGE judging completed. Results saved to {output_file}",
            state="complete",
            expanded=False
        )