import json
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff


def bertscore_results_render(file_path):
    st.write("bertscore_results")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bert_score_data = pd.DataFrame(data)

    if bert_score_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", bert_score_data.shape[0])
    with success_num_col:
        st.metric("Success number", bert_score_data["bertscore_f1"].notnull().sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{bert_score_data['bertscore_f1'].notnull().mean():.2%}")

    ave_score_col, max_score_col, min_score_col = st.columns(3)
    with ave_score_col:
        st.metric("Average score", f"{bert_score_data['bertscore_f1'].mean():.4f}")
    with max_score_col:
        st.metric("Max score", f"{bert_score_data['bertscore_f1'].max():.4f}")
    with min_score_col:
        st.metric("Min score", f"{bert_score_data['bertscore_f1'].min():.4f}")

    ave_prec_col, max_prec_col, min_prec_col = st.columns(3)
    with ave_prec_col:
        st.metric("Average precision", f"{bert_score_data['bertscore_precision'].mean():.4f}")
    with max_prec_col:
        st.metric("Max precision", f"{bert_score_data['bertscore_precision'].max():.4f}")
    with min_prec_col:
        st.metric("Min precision", f"{bert_score_data['bertscore_precision'].min():.4f}")

    ave_recall_col, max_recall_col, min_recall_col = st.columns(3)
    with ave_recall_col:
        st.metric("Average recall", f"{bert_score_data['bertscore_recall'].mean():.4f}")
    with max_recall_col:
        st.metric("Max recall", f"{bert_score_data['bertscore_recall'].max():.4f}")
    with min_recall_col:
        st.metric("Min recall", f"{bert_score_data['bertscore_recall'].min():.4f}")

    hist_data = [
        bert_score_data["bertscore_f1"].dropna(),
        bert_score_data["bertscore_precision"].dropna(),
        bert_score_data["bertscore_recall"].dropna()
    ]

    hist_labels = ["F1", "Precision", "Recall"]

    fig = ff.create_distplot(hist_data, hist_labels, show_hist=False)

    st.plotly_chart(fig)

def bleurt_results_render(file_path):
    st.write("bleurt_results")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bleurt_data = pd.DataFrame(data)

    if bleurt_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", bleurt_data.shape[0])
    with success_num_col:
        st.metric("Success number", bleurt_data["score"].notnull().sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{bleurt_data['score'].notnull().mean():.2%}")

    ave_score_col, max_score_col, min_score_col = st.columns(3)
    with ave_score_col:
        st.metric("Average score", f"{bleurt_data['score'].mean():.4f}")
    with max_score_col:
        st.metric("Max score", f"{bleurt_data['score'].max():.4f}")
    with min_score_col:
        st.metric("Min score", f"{bleurt_data['score'].min():.4f}")

    hist_data = [
        bleurt_data["score"].dropna(),
    ]

    hist_labels = ["BLEURT Score"]

    fig = ff.create_distplot(hist_data, hist_labels, show_hist=False)

    st.plotly_chart(fig)

def keyfact_alignment_render(file_path):
    st.write("keyfact_alignment")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keyfact_alignment_data = pd.DataFrame(data['results'])

    if keyfact_alignment_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", keyfact_alignment_data.shape[0])
    with success_num_col:
        st.metric("Success number", keyfact_alignment_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{keyfact_alignment_data['success'].mean():.2%}")

    success_data = keyfact_alignment_data[keyfact_alignment_data["success"]]

    all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]

    total_pred_labels = len(all_pred_labels)
    pred_labels_1_count = sum(1 for label in all_pred_labels if label == 1)
    pred_labels_1_ratio = pred_labels_1_count / total_pred_labels if total_pred_labels > 0 else 0

    total_pred_labels_col, pred_labels_1_count_col, pred_labels_1_ratio_col = st.columns(3)

    with total_pred_labels_col:
        st.metric("Total successful pred_labels", total_pred_labels)
    with pred_labels_1_count_col:
        st.metric("Correct pred_labels number", pred_labels_1_count)
    with pred_labels_1_ratio_col:
        st.metric("Correct pred_labels ratio", f"{pred_labels_1_ratio:.2%}")

def keyfact_check_render(file_path):
    st.write("keyfact_check")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keyfact_check_data = pd.DataFrame(data)

    if keyfact_check_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", keyfact_check_data.shape[0])
    with success_num_col:
        st.metric("Success number", keyfact_check_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{keyfact_check_data['success'].mean():.2%}")

    success_data = keyfact_check_data[keyfact_check_data["success"]]
    all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]

    total_pred_labels = len(all_pred_labels)
    fact_correct_labels_count = sum(1 for label in all_pred_labels if label == 0)

    fact_correct_labels_ratio = fact_correct_labels_count / total_pred_labels if total_pred_labels > 0 else 0

    total_pred_labels_col, fact_correct_labels_count_col, fact_correct_labels_ratio_col = st.columns(3)
    with total_pred_labels_col:
        st.metric("Total successful pred_labels", total_pred_labels)
    with fact_correct_labels_count_col:
        st.metric("Correct pred_labels number", fact_correct_labels_count)
    with fact_correct_labels_ratio_col:
        st.metric("Correct pred_labels ratio", f"{fact_correct_labels_ratio:.2%}")

def rouge_results_render(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rouge_data = pd.DataFrame(data)

    if rouge_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", rouge_data.shape[0])
    with success_num_col:
        st.metric("Success number", rouge_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{rouge_data['success'].mean():.2%}")

    ave_rouge_1_f_col, max_rouge_1_f_col, min_rouge_1_f_col, ave_rouge_1_p_col, max_rouge_1_p_col, min_rouge_1_p_col, ave_rouge_1_r_col, max_rouge_1_r_col, min_rouge_1_r_col = st.columns(
        9)
    with ave_rouge_1_f_col:
        st.metric("Average rouge-1-f", f"{rouge_data['rouge-1-f'].mean():.4f}")
    with max_rouge_1_f_col:
        st.metric("Max rouge-1-f", f"{rouge_data['rouge-1-f'].max():.4f}")
    with min_rouge_1_f_col:
        st.metric("Min rouge-1-f", f"{rouge_data['rouge-1-f'].min():.4f}")
    with ave_rouge_1_p_col:
        st.metric("Average rouge-1-p", f"{rouge_data['rouge-1-p'].mean():.4f}")
    with max_rouge_1_p_col:
        st.metric("Max rouge-1-p", f"{rouge_data['rouge-1-p'].max():.4f}")
    with min_rouge_1_p_col:
        st.metric("Min rouge-1-p", f"{rouge_data['rouge-1-p'].min():.4f}")
    with ave_rouge_1_r_col:
        st.metric("Average rouge-1-r", f"{rouge_data['rouge-1-r'].mean():.4f}")
    with max_rouge_1_r_col:
        st.metric("Max rouge-1-r", f"{rouge_data['rouge-1-r'].max():.4f}")
    with min_rouge_1_r_col:
        st.metric("Min rouge-1-r", f"{rouge_data['rouge-1-r'].min():.4f}")

    ave_rouge_2_f_col, max_rouge_2_f_col, min_rouge_2_f_col, ave_rouge_2_p_col, max_rouge_2_p_col, min_rouge_2_p_col, ave_rouge_2_r_col, max_rouge_2_r_col, min_rouge_2_r_col = st.columns(
        9)
    with ave_rouge_2_f_col:
        st.metric("Average rouge-2-f", f"{rouge_data['rouge-2-f'].mean():.4f}")
    with max_rouge_2_f_col:
        st.metric("Max rouge-2-f", f"{rouge_data['rouge-2-f'].max():.4f}")
    with min_rouge_2_f_col:
        st.metric("Min rouge-2-f", f"{rouge_data['rouge-2-f'].min():.4f}")
    with ave_rouge_2_p_col:
        st.metric("Average rouge-2-p", f"{rouge_data['rouge-2-p'].mean():.4f}")
    with max_rouge_2_p_col:
        st.metric("Max rouge-2-p", f"{rouge_data['rouge-2-p'].max():.4f}")
    with min_rouge_2_p_col:
        st.metric("Min rouge-2-p", f"{rouge_data['rouge-2-p'].min():.4f}")
    with ave_rouge_2_r_col:
        st.metric("Average rouge-2-r", f"{rouge_data['rouge-2-r'].mean():.4f}")
    with max_rouge_2_r_col:
        st.metric("Max rouge-2-r", f"{rouge_data['rouge-2-r'].max():.4f}")
    with min_rouge_2_r_col:
        st.metric("Min rouge-2-r", f"{rouge_data['rouge-2-r'].min():.4f}")

    ave_rouge_l_f_col, max_rouge_l_f_col, min_rouge_l_f_col, ave_rouge_l_p_col, max_rouge_l_p_col, min_rouge_l_p_col, ave_rouge_l_r_col, max_rouge_l_r_col, min_rouge_l_r_col = st.columns(
        9)
    with ave_rouge_l_f_col:
        st.metric("Average rouge-l-f", f"{rouge_data['rouge-l-f'].mean():.4f}")
    with max_rouge_l_f_col:
        st.metric("Max rouge-l-f", f"{rouge_data['rouge-l-f'].max():.4f}")
    with min_rouge_l_f_col:
        st.metric("Min rouge-l-f", f"{rouge_data['rouge-l-f'].min():.4f}")
    with ave_rouge_l_p_col:
        st.metric("Average rouge-l-p", f"{rouge_data['rouge-l-p'].mean():.4f}")
    with max_rouge_l_p_col:
        st.metric("Max rouge-l-p", f"{rouge_data['rouge-l-p'].max():.4f}")
    with min_rouge_l_p_col:
        st.metric("Min rouge-l-p", f"{rouge_data['rouge-l-p'].min():.4f}")
    with ave_rouge_l_r_col:
        st.metric("Average rouge-l-r", f"{rouge_data['rouge-l-r'].mean():.4f}")
    with max_rouge_l_r_col:
        st.metric("Max rouge-l-r", f"{rouge_data['rouge-l-r'].max():.4f}")
    with min_rouge_l_r_col:
        st.metric("Min rouge-l-r", f"{rouge_data['rouge-l-r'].min():.4f}")



    rouge_1_hist_data = [
        rouge_data["rouge-1-f"].dropna(),
        rouge_data["rouge-1-p"].dropna(),
        rouge_data["rouge-1-r"].dropna()
    ]
    rouge_1_hist_labels = ["F1", "Precision", "Recall"]
    rouge_1_fig = ff.create_distplot(rouge_1_hist_data, rouge_1_hist_labels, show_hist=False)

    st.write("rouge-1")
    st.plotly_chart(rouge_1_fig)

    rouge_2_hist_data = [
        rouge_data["rouge-2-f"].dropna(),
        rouge_data["rouge-2-p"].dropna(),
        rouge_data["rouge-2-r"].dropna()
    ]
    rouge_2_hist_labels = ["F1", "Precision", "Recall"]
    rouge_2_fig = ff.create_distplot(rouge_2_hist_data, rouge_2_hist_labels, show_hist=False)

    st.write("rouge-2")
    st.plotly_chart(rouge_2_fig)

    rouge_l_hist_data = [
        rouge_data["rouge-l-f"].dropna(),
        rouge_data["rouge-l-p"].dropna(),
        rouge_data["rouge-l-r"].dropna()
    ]
    rouge_l_hist_labels = ["F1", "Precision", "Recall"]
    rouge_l_fig = ff.create_distplot(rouge_l_hist_data, rouge_l_hist_labels, show_hist=False)

    st.write("rouge-l")
    st.plotly_chart(rouge_l_fig)

def summaries_render(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries_data = pd.DataFrame(data["results"])

    if summaries_data.shape[0] == 0:
        st.error("No data to display.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", summaries_data.shape[0])
    with success_num_col:
        st.metric("Success number", summaries_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{summaries_data['success'].mean():.2%}")

    success_data = summaries_data[summaries_data["success"]]
    summary_lengths = success_data["summary"].apply(len)

    ave_summary_length_col, max_summary_length_col, min_summary_length_col = st.columns(3)
    with ave_summary_length_col:
        st.metric("Average summary length", f"{summary_lengths.mean():.2f}")
    with max_summary_length_col:
        st.metric("Max summary length", f"{summary_lengths.max()}")
    with min_summary_length_col:
        st.metric("Min summary length", f"{summary_lengths.min()}")

    hist_data = [
        summary_lengths
    ]

    hist_labels = ["Summary Length"]

    fig = ff.create_distplot(hist_data, hist_labels, show_hist=False)

    st.plotly_chart(fig)

def llm_score_render(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_data = data['scores']

    faithfulness_col, completeness_col, conciseness_col = st.columns(3)
    with faithfulness_col:
        st.metric("Faithfulness", f"{llm_data['faithfulness_score']:.4f}")
    with completeness_col:
        st.metric("Completeness", f"{llm_data['completeness_score']:.4f}")
    with conciseness_col:
        st.metric("Conciseness", f"{llm_data['conciseness_score']:.4f}")