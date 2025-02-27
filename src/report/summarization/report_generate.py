import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

def init_template(task_name):
    html_template = '''\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERTScore Evaluation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }

        .container {
            max-width: 1200px;
            width: 100%;
            padding: 20px;
        }

        h1, h2, h3 {
            color: #007bff;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .display-6 {
            font-size: 30px;
        }

        .divider {
            border-top: 2px solid #007bff;
            margin: 20px 0;
        }
    </style>
</head>
<body>
<div class="container mt-5">
    <h1 style="text-align: center">Summarization Evaluation Report</h1>
    <p style="text-align: center">Task name: {{task_name_placeholder}}</p>
    <p style="text-align: center">Generated time: {{generated_time}}</p>
    
    '''
    html_template = html_template.replace("{{task_name_placeholder}}", task_name)
    html_template = html_template.replace("{{generated_time}}", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    return html_template

def generate_bertscore_report(json_path, report_template):
    report_template += '''\
    <h3>BERTScore Evaluation Report</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="bertscore_total_number" class="display-6">{{bertscore_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="bertscore_success_number" class="display-6">{{bertscore_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="bertscore_success_ratio" class="display-6">{{bertscore_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average F1 Score</h5>
                    <p id="bertscore_average_score" class="display-6">{{bertscore_average_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max F1 Score</h5>
                    <p id="bertscore_max_score" class="display-6">{{bertscore_max_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min F1 Score</h5>
                    <p id="bertscore_min_score" class="display-6">{{bertscore_min_score_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Precision</h5>
                    <p id="bertscore_average_precision" class="display-6">
                        {{bertscore_average_precision_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max Precision</h5>
                    <p id="bertscore_max_precision" class="display-6">{{bertscore_max_precision_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min Precision</h5>
                    <p id="bertscore_min_precision" class="display-6">{{bertscore_min_precision_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Recall</h5>
                    <p id="bertscore_average_recall" class="display-6">{{bertscore_average_recall_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max Recall</h5>
                    <p id="bertscore_max_recall" class="display-6">{{bertscore_max_recall_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min Recall</h5>
                    <p id="bertscore_min_recall" class="display-6">{{bertscore_min_recall_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col text-center">
                <h3>BERTScore Distribution</h3>
                <canvas id="bertscore_hist_chart"></canvas>
            </div>
        </div>
    </div>
    '''

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    bert_score_data = pd.DataFrame(data)

    # 处理数据为空的情况
    if bert_score_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = bert_score_data.shape[0]
    success_number = bert_score_data["bertscore_f1"].notnull().sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    average_score = f"{bert_score_data['bertscore_f1'].mean():.4f}"
    max_score = f"{bert_score_data['bertscore_f1'].max():.4f}"
    min_score = f"{bert_score_data['bertscore_f1'].min():.4f}"

    average_precision = f"{bert_score_data['bertscore_precision'].mean():.4f}"
    max_precision = f"{bert_score_data['bertscore_precision'].max():.4f}"
    min_precision = f"{bert_score_data['bertscore_precision'].min():.4f}"

    average_recall = f"{bert_score_data['bertscore_recall'].mean():.4f}"
    max_recall = f"{bert_score_data['bertscore_recall'].max():.4f}"
    min_recall = f"{bert_score_data['bertscore_recall'].min():.4f}"

    # 生成直方图
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)

    # 绘制条形图（直方图）
    sns.histplot(bert_score_data["bertscore_f1"].dropna(), kde=False, label="F1 Score", ax=ax1)
    sns.histplot(bert_score_data["bertscore_precision"].dropna(), kde=False, label="Precision", ax=ax1)
    sns.histplot(bert_score_data["bertscore_recall"].dropna(), kde=False, label="Recall", ax=ax1)
    ax1.set_title("BERTScore Histogram")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # 绘制核密度估计图 (KDE)
    sns.kdeplot(bert_score_data["bertscore_f1"].dropna(), label="F1 Score", fill=True, common_norm=False, ax=ax2)
    sns.kdeplot(bert_score_data["bertscore_precision"].dropna(), label="Precision", fill=True, common_norm=False,
                ax=ax2)
    sns.kdeplot(bert_score_data["bertscore_recall"].dropna(), label="Recall", fill=True, common_norm=False, ax=ax2)
    ax2.set_title("BERTScore KDE Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Density")
    ax2.legend()

    # 保存图像为 Base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)

    # 填充统计数据
    html_filled = report_template.replace("{{bertscore_total_number_placeholder}}", str(total_number)) \
        .replace("{{bertscore_success_number_placeholder}}", str(success_number)) \
        .replace("{{bertscore_success_ratio_placeholder}}", success_ratio) \
        .replace("{{bertscore_average_score_placeholder}}", average_score) \
        .replace("{{bertscore_max_score_placeholder}}", max_score) \
        .replace("{{bertscore_min_score_placeholder}}", min_score) \
        .replace("{{bertscore_average_precision_placeholder}}", average_precision) \
        .replace("{{bertscore_max_precision_placeholder}}", max_precision) \
        .replace("{{bertscore_min_precision_placeholder}}", min_precision) \
        .replace("{{bertscore_average_recall_placeholder}}", average_recall) \
        .replace("{{bertscore_max_recall_placeholder}}", max_recall) \
        .replace("{{bertscore_min_recall_placeholder}}", min_recall) \
        .replace('<canvas id="bertscore_hist_chart"></canvas>',
                 f'<img src="data:image/png;base64,{img_base64}" class="img-fluid" alt="BERTScore Distribution">')

    return html_filled


def generate_bleurt_report(json_path, report_template):
    report_template += '''\
    <h3>BLEURT Evaluation Report</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="bleurt_total_number" class="display-6">{{bleurt_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="bleurt_success_number" class="display-6">{{bleurt_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="bleurt_success_ratio" class="display-6">{{bleurt_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Score</h5>
                    <p id="bleurt_average_score" class="display-6">{{bleurt_average_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max Score</h5>
                    <p id="bleurt_max_score" class="display-6">{{bleurt_max_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min Score</h5>
                    <p id="bleurt_min_score" class="display-6">{{bleurt_min_score_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col text-center">
                <div class="col">
                    <h5>BLEURT Score Distribution</h5>
                    <canvas id="bleurt_hist_chart"></canvas>
                </div>
            </div>
        </div>
    </div>
    '''
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    bleurt_data = pd.DataFrame(data)

    # 处理数据为空的情况
    if bleurt_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = bleurt_data.shape[0]
    success_number = bleurt_data["score"].notnull().sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    average_score = f"{bleurt_data['score'].mean():.4f}"
    max_score = f"{bleurt_data['score'].max():.4f}"
    min_score = f"{bleurt_data['score'].min():.4f}"

    # 生成直方图
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)

    # 绘制BLEURT条形图（直方图）
    sns.histplot(bleurt_data["score"].dropna(), kde=False, label="BLEURT Score", ax=ax1)
    ax1.set_title("BLEURT Histogram")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # 绘制BLEURT核密度估计图 (KDE)
    sns.kdeplot(bleurt_data["score"].dropna(), label="BLEURT Score", fill=True, common_norm=False, ax=ax2)
    ax2.set_title("BLEURT KDE Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Density")
    ax2.legend()

    # 保存图像为 Base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)

    # 填充统计数据
    html_filled = report_template.replace("{{bleurt_total_number_placeholder}}", str(total_number)) \
        .replace("{{bleurt_success_number_placeholder}}", str(success_number)) \
        .replace("{{bleurt_success_ratio_placeholder}}", success_ratio) \
        .replace("{{bleurt_average_score_placeholder}}", average_score) \
        .replace("{{bleurt_max_score_placeholder}}", max_score) \
        .replace("{{bleurt_min_score_placeholder}}", min_score) \
        .replace('<canvas id="bleurt_hist_chart"></canvas>',
                 f'<img src="data:image/png;base64,{img_base64}" class="img-fluid" alt="BLEURT Distribution">')

    return html_filled


def generate_keyfact_alignment_report(json_path, report_template):
    report_template += '''\
    <h3>Keyfact Alignment Report</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="keyfact_alignment_total_number" class="display-6">
                        {{keyfact_alignment_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="keyfact_alignment_success_number" class="display-6">
                        {{keyfact_alignment_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="keyfact_alignment_success_ratio" class="display-6">
                        {{keyfact_alignment_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Pred Labels</h5>
                    <p id="keyfact_alignment_total_pred_labels" class="display-6">
                        {{keyfact_alignment_total_pred_labels_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Pred Labels 1 Count</h5>
                    <p id="keyfact_alignment_pred_labels_1_count" class="display-6">
                        {{keyfact_alignment_pred_labels_1_count_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Pred Labels 1 Ratio</h5>
                    <p id="keyfact_alignment_pred_labels_1_ratio" class="display-6">
                        {{keyfact_alignment_pred_labels_1_ratio_placeholder}}</p>
                </div>
            </div>
        </div>
    </div>

    '''
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    keyfact_alignment_data = pd.DataFrame(data['results'])

    # 处理数据为空的情况
    if keyfact_alignment_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = keyfact_alignment_data.shape[0]
    success_number = keyfact_alignment_data["success"].sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    success_data = keyfact_alignment_data[keyfact_alignment_data["success"]]

    all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]

    total_pred_labels = len(all_pred_labels)
    pred_labels_1_count = sum(1 for label in all_pred_labels if label == 1)
    pred_labels_1_ratio = pred_labels_1_count / total_pred_labels if total_pred_labels > 0 else 0

    # 填充统计数据
    html_filled = report_template.replace("{{keyfact_alignment_total_number_placeholder}}", str(total_number)) \
        .replace("{{keyfact_alignment_success_number_placeholder}}", str(success_number)) \
        .replace("{{keyfact_alignment_success_ratio_placeholder}}", success_ratio) \
        .replace("{{keyfact_alignment_total_pred_labels_placeholder}}", str(total_pred_labels)) \
        .replace("{{keyfact_alignment_pred_labels_1_count_placeholder}}", str(pred_labels_1_count)) \
        .replace("{{keyfact_alignment_pred_labels_1_ratio_placeholder}}", f"{pred_labels_1_ratio:.2%}")

    return html_filled


def generate_keyfact_check_report(json_path, report_template):
    report_template += '''\
    <h3>Keyfact Check Report</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="keyfact_check_total_number" class="display-6">{{keyfact_check_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="keyfact_check_success_number" class="display-6">
                        {{keyfact_check_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="keyfact_check_success_ratio" class="display-6">
                        {{keyfact_check_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Predicted Labels</h5>
                    <p id="keyfact_check_total_pred_labels" class="display-6">
                        {{keyfact_check_total_pred_labels_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Fact Correct Labels Count</h5>
                    <p id="keyfact_check_fact_correct_labels_count" class="display-6">
                        {{keyfact_check_fact_correct_labels_count_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Fact Correct Labels Ratio</h5>
                    <p id="keyfact_check_fact_correct_labels_ratio" class="display-6">
                        {{keyfact_check_fact_correct_labels_ratio_placeholder}}</p>
                </div>
            </div>
        </div>
    </div>
    '''

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    keyfact_check_data = pd.DataFrame(data)

    # 处理数据为空的情况
    if keyfact_check_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = keyfact_check_data.shape[0]
    success_number = keyfact_check_data["success"].sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    success_data = keyfact_check_data[keyfact_check_data["success"]]
    all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]

    total_pred_labels = len(all_pred_labels)
    fact_correct_labels_count = sum(1 for label in all_pred_labels if label == 0)

    fact_correct_labels_ratio = fact_correct_labels_count / total_pred_labels if total_pred_labels > 0 else 0

    # 填充统计数据
    html_filled = report_template.replace("{{keyfact_check_total_number_placeholder}}", str(total_number)) \
        .replace("{{keyfact_check_success_number_placeholder}}", str(success_number)) \
        .replace("{{keyfact_check_success_ratio_placeholder}}", success_ratio) \
        .replace("{{keyfact_check_total_pred_labels_placeholder}}", str(total_pred_labels)) \
        .replace("{{keyfact_check_fact_correct_labels_count_placeholder}}", str(fact_correct_labels_count)) \
        .replace("{{keyfact_check_fact_correct_labels_ratio_placeholder}}", f"{fact_correct_labels_ratio:.2%}")

    return html_filled


def generate_rouge_report(json_path, report_template):
    report_template += '''\
    <h3>ROUGE Report</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="rouge_total_number" class="display-6">{{rouge_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="rouge_success_number" class="display-6">{{rouge_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="rouge_success_ratio" class="display-6">{{rouge_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-1 F</h5>
                    <p id="rouge_average_rouge_1_f" class="display-6">{{rouge_average_rouge_1_f_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-1 F</h5>
                    <p id="rouge_max_rouge_1_f" class="display-6">{{rouge_max_rouge_1_f_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-1 F</h5>
                    <p id="rouge_min_rouge_1_f" class="display-6">{{rouge_min_rouge_1_f_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-1 P</h5>
                    <p id="rouge_average_rouge_1_p" class="display-6">{{rouge_average_rouge_1_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-1 P</h5>
                    <p id="rouge_max_rouge_1_p" class="display-6">{{rouge_max_rouge_1_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-1 P</h5>
                    <p id="rouge_min_rouge_1_p" class="display-6">{{rouge_min_rouge_1_p_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-1 R</h5>
                    <p id="rouge_average_rouge_1_r" class="display-6">{{rouge_average_rouge_1_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-1 R</h5>
                    <p id="rouge_max_rouge_1_r" class="display-6">{{rouge_max_rouge_1_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-1 R</h5>
                    <p id="rouge_min_rouge_1_r" class="display-6">{{rouge_min_rouge_1_r_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-2 F</h5>
                    <p id="rouge_average_rouge_2_f" class="display-6">{{rouge_average_rouge_2_f_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-2 F</h5>
                    <p id="rouge_max_rouge_2_f" class="display-6">{{rouge_max_rouge_2_f_placeholder}}</p>
                </div>
            </div>

            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-2 F</h5>
                    <p id="rouge_min_rouge_2_f" class="display-6">{{rouge_min_rouge_2_f_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-2 P</h5>
                    <p id="rouge_average_rouge_2_p" class="display-6">{{rouge_average_rouge_2_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-2 P</h5>
                    <p id="rouge_max_rouge_2_p" class="display-6">{{rouge_max_rouge_2_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-2 P</h5>
                    <p id="rouge_min_rouge_2_p" class="display-6">{{rouge_min_rouge_2_p_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-2 R</h5>
                    <p id="rouge_average_rouge_2_r" class="display-6">{{rouge_average_rouge_2_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-2 R</h5>
                    <p id="rouge_max_rouge_2_r" class="display-6">{{rouge_max_rouge_2_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-2 R</h5>
                    <p id="rouge_min_rouge_2_r" class="display-6">{{rouge_min_rouge_2_r_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-L F</h5>
                    <p id="rouge_average_rouge_l_f" class="display-6">{{rouge_average_rouge_l_f_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-L F</h5>
                    <p id="rouge_max_rouge_l_f" class="display-6">{{rouge_max_rouge_l_f_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-L F</h5>
                    <p id="rouge_min_rouge_l_f" class="display-6">{{rouge_min_rouge_l_f_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-L P</h5>
                    <p id="rouge_average_rouge_l_p" class="display-6">{{rouge_average_rouge_l_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-L P</h5>
                    <p id="rouge_max_rouge_l_p" class="display-6">{{rouge_max_rouge_l_p_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-L P</h5>
                    <p id="rouge_min_rouge_l_p" class="display-6">{{rouge_min_rouge_l_p_placeholder}}</p>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average ROUGE-L R</h5>
                    <p id="rouge_average_rouge_l_r" class="display-6">{{rouge_average_rouge_l_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max ROUGE-L R</h5>
                    <p id="rouge_max_rouge_l_r" class="display-6">{{rouge_max_rouge_l_r_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min ROUGE-L R</h5>
                    <p id="rouge_min_rouge_l_r" class="display-6">{{rouge_min_rouge_l_r_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col text-center">
                <h3>ROUGE-1 Distribution</h3>
                <canvas id="rouge_1_hist_chart"></canvas>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col text-center">
                <h3>ROUGE-2 Distribution</h3>
                <canvas id="rouge_2_hist_chart"></canvas>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col text-center">
                <h3>ROUGE-L Distribution</h3>
                <canvas id="rouge_l_hist_chart"></canvas>
            </div>
        </div>
    </div>
    '''
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    rouge_data = pd.DataFrame(data)

    # 处理数据为空的情况
    if rouge_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = rouge_data.shape[0]
    success_number = rouge_data["success"].sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    average_rouge_1_f = f"{rouge_data['rouge-1-f'].mean():.4f}"
    max_rouge_1_f = f"{rouge_data['rouge-1-f'].max():.4f}"
    min_rouge_1_f = f"{rouge_data['rouge-1-f'].min():.4f}"
    average_rouge_1_p = f"{rouge_data['rouge-1-p'].mean():.4f}"
    max_rouge_1_p = f"{rouge_data['rouge-1-p'].max():.4f}"
    min_rouge_1_p = f"{rouge_data['rouge-1-p'].min():.4f}"
    average_rouge_1_r = f"{rouge_data['rouge-1-r'].mean():.4f}"
    max_rouge_1_r = f"{rouge_data['rouge-1-r'].max():.4f}"
    min_rouge_1_r = f"{rouge_data['rouge-1-r'].min():.4f}"

    average_rouge_2_f = f"{rouge_data['rouge-2-f'].mean():.4f}"
    max_rouge_2_f = f"{rouge_data['rouge-2-f'].max():.4f}"
    min_rouge_2_f = f"{rouge_data['rouge-2-f'].min():.4f}"
    average_rouge_2_p = f"{rouge_data['rouge-2-p'].mean():.4f}"
    max_rouge_2_p = f"{rouge_data['rouge-2-p'].max():.4f}"
    min_rouge_2_p = f"{rouge_data['rouge-2-p'].min():.4f}"
    average_rouge_2_r = f"{rouge_data['rouge-2-r'].mean():.4f}"
    max_rouge_2_r = f"{rouge_data['rouge-2-r'].max():.4f}"
    min_rouge_2_r = f"{rouge_data['rouge-2-r'].min():.4f}"

    average_rouge_l_f = f"{rouge_data['rouge-l-f'].mean():.4f}"
    max_rouge_l_f = f"{rouge_data['rouge-l-f'].max():.4f}"
    min_rouge_l_f = f"{rouge_data['rouge-l-f'].min():.4f}"
    average_rouge_l_p = f"{rouge_data['rouge-l-p'].mean():.4f}"
    max_rouge_l_p = f"{rouge_data['rouge-l-p'].max():.4f}"
    min_rouge_l_p = f"{rouge_data['rouge-l-p'].min():.4f}"
    average_rouge_l_r = f"{rouge_data['rouge-l-r'].mean():.4f}"
    max_rouge_l_r = f"{rouge_data['rouge-l-r'].max():.4f}"
    min_rouge_l_r = f"{rouge_data['rouge-l-r'].min():.4f}"

    def generate_rouge_hist(rouge_data, metric, f_col, p_col, r_col):
        fig, (ax1, ax2) = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)

        # 获取 F1, Precision, Recall
        avg_f = f"{rouge_data[f_col].mean():.4f}"
        avg_p = f"{rouge_data[p_col].mean():.4f}"
        avg_r = f"{rouge_data[r_col].mean():.4f}"

        # 绘制条形图（直方图）
        sns.histplot(rouge_data[f"{metric}-f"].dropna(), kde=False, label="F1 Score", ax=ax1)
        sns.histplot(rouge_data[f"{metric}-p"].dropna(), kde=False, label="Precision", ax=ax1)
        sns.histplot(rouge_data[f"{metric}-r"].dropna(), kde=False, label="Recall", ax=ax1)
        ax1.set_title(f"ROUGE {metric} Histogram\nF1: {avg_f} | P: {avg_p} | R: {avg_r}")
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # 绘制核密度估计图 (KDE)
        sns.kdeplot(rouge_data[f"{metric}-f"].dropna(), label="F1 Score", fill=True, common_norm=False, ax=ax2)
        sns.kdeplot(rouge_data[f"{metric}-p"].dropna(), label="Precision", fill=True, common_norm=False, ax=ax2)
        sns.kdeplot(rouge_data[f"{metric}-r"].dropna(), label="Recall", fill=True, common_norm=False, ax=ax2)
        ax2.set_title(f"ROUGE {metric} KDE Distribution\nF1: {avg_f} | P: {avg_p} | R: {avg_r}")
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Density")
        ax2.legend()

        plt.tight_layout()

        # 保存图像为 Base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return img_base64

    # 生成ROUGE 1, 2, L的三个直方图，并传递对应的F, P, R列名
    rouge_1_img = generate_rouge_hist(rouge_data, 'rouge-1', 'rouge-1-f', 'rouge-1-p', 'rouge-1-r')
    rouge_2_img = generate_rouge_hist(rouge_data, 'rouge-2', 'rouge-2-f', 'rouge-2-p', 'rouge-2-r')
    rouge_l_img = generate_rouge_hist(rouge_data, 'rouge-l', 'rouge-l-f', 'rouge-l-p', 'rouge-l-r')

    # 填充统计数据
    html_filled = report_template.replace("{{rouge_total_number_placeholder}}", str(total_number)) \
        .replace("{{rouge_success_number_placeholder}}", str(success_number)) \
        .replace("{{rouge_success_ratio_placeholder}}", success_ratio) \
        .replace("{{rouge_average_rouge_1_f_placeholder}}", average_rouge_1_f) \
        .replace("{{rouge_max_rouge_1_f_placeholder}}", max_rouge_1_f) \
        .replace("{{rouge_min_rouge_1_f_placeholder}}", min_rouge_1_f) \
        .replace("{{rouge_average_rouge_1_p_placeholder}}", average_rouge_1_p) \
        .replace("{{rouge_max_rouge_1_p_placeholder}}", max_rouge_1_p) \
        .replace("{{rouge_min_rouge_1_p_placeholder}}", min_rouge_1_p) \
        .replace("{{rouge_average_rouge_1_r_placeholder}}", average_rouge_1_r) \
        .replace("{{rouge_max_rouge_1_r_placeholder}}", max_rouge_1_r) \
        .replace("{{rouge_min_rouge_1_r_placeholder}}", min_rouge_1_r) \
        .replace("{{rouge_average_rouge_2_f_placeholder}}", average_rouge_2_f) \
        .replace("{{rouge_max_rouge_2_f_placeholder}}", max_rouge_2_f) \
        .replace("{{rouge_min_rouge_2_f_placeholder}}", min_rouge_2_f) \
        .replace("{{rouge_average_rouge_2_p_placeholder}}", average_rouge_2_p) \
        .replace("{{rouge_max_rouge_2_p_placeholder}}", max_rouge_2_p) \
        .replace("{{rouge_min_rouge_2_p_placeholder}}", min_rouge_2_p) \
        .replace("{{rouge_average_rouge_2_r_placeholder}}", average_rouge_2_r) \
        .replace("{{rouge_max_rouge_2_r_placeholder}}", max_rouge_2_r) \
        .replace("{{rouge_min_rouge_2_r_placeholder}}", min_rouge_2_r) \
        .replace("{{rouge_average_rouge_l_f_placeholder}}", average_rouge_l_f) \
        .replace("{{rouge_max_rouge_l_f_placeholder}}", max_rouge_l_f) \
        .replace("{{rouge_min_rouge_l_f_placeholder}}", min_rouge_l_f) \
        .replace("{{rouge_average_rouge_l_p_placeholder}}", average_rouge_l_p) \
        .replace("{{rouge_max_rouge_l_p_placeholder}}", max_rouge_l_p) \
        .replace("{{rouge_min_rouge_l_p_placeholder}}", min_rouge_l_p) \
        .replace("{{rouge_average_rouge_l_r_placeholder}}", average_rouge_l_r) \
        .replace("{{rouge_max_rouge_l_r_placeholder}}", max_rouge_l_r) \
        .replace("{{rouge_min_rouge_l_r_placeholder}}", min_rouge_l_r) \
        .replace('<canvas id="rouge_1_hist_chart"></canvas>',
                 f'<img src="data:image/png;base64,{rouge_1_img}" class="img-fluid" alt="ROUGE-1 Distribution">') \
        .replace('<canvas id="rouge_2_hist_chart"></canvas>',
                 f'<img src="data:image/png;base64,{rouge_2_img}" class="img-fluid" alt="ROUGE-2 Distribution">') \
        .replace('<canvas id="rouge_l_hist_chart"></canvas>',
                 f'<img src="data:image/png;base64,{rouge_l_img}" class="img-fluid" alt="ROUGE-L Distribution">')

    return html_filled


def generate_summarization_report(json_path, report_template):
    report_template += '''\
    <h2>Summaries Report</h2>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Number</h5>
                    <p id="summaries_total_number" class="display-6">{{summaries_total_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Number</h5>
                    <p id="summaries_success_number" class="display-6">{{summaries_success_number_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Success Ratio</h5>
                    <p id="summaries_success_ratio" class="display-6">{{summaries_success_ratio_placeholder}}</p>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Summary Length</h5>
                    <p id="summaries_average_summary_length" class="display-6">
                        {{summaries_average_summary_length_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Max Summary Length</h5>
                    <p id="summaries_max_summary_length" class="display-6">
                        {{summaries_max_summary_length_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Min Summary Length</h5>
                    <p id="summaries_min_summary_length" class="display-6">
                        {{summaries_min_summary_length_placeholder}}</p>
                </div>
            </div>
        </div>
    </div>
    
    '''
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 DataFrame
    summaries_data = pd.DataFrame(data["results"])

    # 处理数据为空的情况
    if summaries_data.empty:
        print("No data available.")
        return report_template

    # 计算统计信息
    total_number = summaries_data.shape[0]
    success_number = summaries_data["success"].sum()
    success_ratio = f"{(success_number / total_number) * 100:.2f}%"

    success_data = summaries_data[summaries_data["success"]]
    summary_lengths = success_data["summary"].apply(len)

    average_summary_length = f"{summary_lengths.mean():.2f}"
    max_summary_length = f"{summary_lengths.max()}"
    min_summary_length = f"{summary_lengths.min()}"

    # 填充统计数据
    html_filled = report_template.replace("{{summaries_total_number_placeholder}}", str(total_number)) \
        .replace("{{summaries_success_number_placeholder}}", str(success_number)) \
        .replace("{{summaries_success_ratio_placeholder}}", success_ratio) \
        .replace("{{summaries_average_summary_length_placeholder}}", average_summary_length) \
        .replace("{{summaries_max_summary_length_placeholder}}", max_summary_length) \
        .replace("{{summaries_min_summary_length_placeholder}}", min_summary_length)

    return html_filled


def generate_llm_score_report(json_path, report_template):
    report_template += '''\
    <h2>LLM Based Judgement</h2>
    <h3>LLM score</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Faithfulness</h5>
                    <p id="llm_faithfulness_score" class="display-6">{{llm_faithfulness_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Completeness</h5>
                    <p id="llm_completeness_score" class="display-6">{{llm_completeness_score_placeholder}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Concisenesse</h5>
                    <p id="llm_conciseness_score" class="display-6">{{llm_conciseness_score_placeholder}}</p>
                </div>
            </div>
        </div>
    </div>
    
    '''

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_data = data['scores']

    # 计算统计信息
    faithfulness = f"{llm_data['faithfulness_score']:.4f}"
    completeness = f"{llm_data['completeness_score']:.4f}"
    conciseness = f"{llm_data['conciseness_score']:.4f}"

    # 填充统计数据
    html_filled = report_template.replace("{{llm_faithfulness_score_placeholder}}", faithfulness) \
        .replace("{{llm_completeness_score_placeholder}}", completeness) \
        .replace("{{llm_conciseness_score_placeholder}}", conciseness)

    return html_filled