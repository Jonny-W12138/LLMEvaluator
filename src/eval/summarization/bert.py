import datetime
import bert_score
import torch
import streamlit as st
import json
import pandas as pd
import os

from sympy.integrals.meijerint_doc import category


def bert_judge(bert_model, lang, num_layers,
               summary_file_path, task, field_mapping, selected_data_path):

    with st.status("Bert judging...", expanded=True) as status:
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

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        if num_layers == '' or num_layers == None or int(num_layers) == 0:
            P, R, F1 = bert_score.score(
                summaries_to_judge_list,
                ref_summaries_list,
                model_type=bert_model,
                lang=lang,
                device=device,
                verbose=True
            )
        else:
            P, R, F1 = bert_score.score(
                summaries_to_judge_list,
                ref_summaries_list,
                model_type=bert_model,
                lang=lang,
                num_layers=num_layers,
                device=device,
                verbose=True
            )

        st.write("Saving results...")

        results = []
        for i in range(total_num):
            results.append({
                "summary": summaries_to_judge_list[i],
                "ref_summary": ref_summaries_list[i],
                "category": categories[i],
                "bertscore_precision": P[i].item(),
                "bertscore_recall": R[i].item(),
                "bertscore_f1": F1[i].item()
            })

        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "bertscore_results"), exist_ok=True)
        output_file = os.path.join(os.getcwd(), "tasks", f"{task}", "summarization", "evaluation", "bertscore_results",
                                   f"bertscore_results_{current_time}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        status.update(
            label=f"Bert judge complete! Results saved to {output_file}.",
            state="complete",
            expanded=False
        )