import streamlit as st
import datetime
import torch
import json
import pandas as pd
import os
from bleurt import score
from fontTools.merge.util import current_time


def bleurt_judge(bleurt_model, summary_file_path, task, field_mapping, selected_data_path):

    with st.status("BLEURT judging...", expanded=True) as status:
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

        total_num = ref_summaries.shape[0]

        for i in range(total_num):
            summaries_to_judge_list.append(summaries_to_judge.iloc[i]["summary"].replace("\n", ""))
            ref_summaries_list.append(ref_summaries[ref_summary_field].iloc[i])
            categories.append(ref_summaries["category"].iloc[i])

        st.write("Evaluating summaries...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scorer = score.BleurtScorer(bleurt_model)
        scores = scorer.score(references=ref_summaries_list, candidates=summaries_to_judge_list)

        assert isinstance(scores, list) and len(scores) == total_num

        st.write("Saving results...")

        results = []
        for i in range(total_num):
            results.append({
                "ref_summary": ref_summaries_list[i],
                "summary": summaries_to_judge_list[i],
                "category": categories[i],
                "score": scores[i]
            })

        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "bleurt_results"), exist_ok=True)
        output_file = os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "bleurt_results",
                                   f"bleurt_results_{current_time}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        status.update(
            label=f"BLEURT judging completed. Results saved to {output_file}.",
            state="complete",
            expanded=False
        )

