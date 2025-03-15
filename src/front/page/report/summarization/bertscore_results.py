import json
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go


def bertscore_results_render(file_path):
    st.write("bertscore_results")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bert_score_data = pd.DataFrame(data)

    if bert_score_data.shape[0] == 0:
        st.error("No data to display.")
        return

    # 检查是否存在 category 字段
    if "category" not in bert_score_data.columns:
        st.warning("No 'category' field found in the data. Cannot display metrics by category.")
        return

    # 总体指标
    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", bert_score_data.shape[0])
    with success_num_col:
        st.metric("Success number", bert_score_data["bertscore_f1"].notnull().sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{bert_score_data['bertscore_f1'].notnull().mean():.2%}")

    # 显示 F1、Precision 和 Recall 的总体指标
    ave_score_col, max_score_col, min_score_col = st.columns(3)
    with ave_score_col:
        st.metric("Average F1", f"{bert_score_data['bertscore_f1'].mean():.4f}")
    with max_score_col:
        st.metric("Max F1", f"{bert_score_data['bertscore_f1'].max():.4f}")
    with min_score_col:
        st.metric("Min F1", f"{bert_score_data['bertscore_f1'].min():.4f}")

    ave_prec_col, max_prec_col, min_prec_col = st.columns(3)
    with ave_prec_col:
        st.metric("Average Precision", f"{bert_score_data['bertscore_precision'].mean():.4f}")
    with max_prec_col:
        st.metric("Max Precision", f"{bert_score_data['bertscore_precision'].max():.4f}")
    with min_prec_col:
        st.metric("Min Precision", f"{bert_score_data['bertscore_precision'].min():.4f}")

    ave_recall_col, max_recall_col, min_recall_col = st.columns(3)
    with ave_recall_col:
        st.metric("Average Recall", f"{bert_score_data['bertscore_recall'].mean():.4f}")
    with max_recall_col:
        st.metric("Max Recall", f"{bert_score_data['bertscore_recall'].max():.4f}")
    with min_recall_col:
        st.metric("Min Recall", f"{bert_score_data['bertscore_recall'].min():.4f}")

    # 绘制 F1、Precision 和 Recall 的分布图
    hist_data = [
        bert_score_data["bertscore_f1"].dropna(),
        bert_score_data["bertscore_precision"].dropna(),
        bert_score_data["bertscore_recall"].dropna()
    ]
    hist_labels = ["F1", "Precision", "Recall"]
    fig_distplot = ff.create_distplot(hist_data, hist_labels, show_hist=False)
    st.plotly_chart(fig_distplot)

    # 绘制每个类别的 F1、Precision 和 Recall 的小提琴图
    st.subheader("Violin Plots by Category")
    metrics = ["bertscore_f1", "bertscore_precision", "bertscore_recall"]
    for metric in metrics:
        fig_violin = px.violin(
            bert_score_data,
            x="category",
            y=metric,
            box=True,  # 显示箱线图
            color="category",
            points="all",  # 显示所有数据点
            title=f"{metric.capitalize()} by Category",
            labels={"category": "Category", metric: metric.capitalize()}
        )
        st.plotly_chart(fig_violin)

    # 计算每个类别的 F1、Precision 和 Recall 的平均值
    category_metrics = {}
    for category, group in bert_score_data.groupby("category"):
        category_metrics[category] = {
            "avg_f1": group["bertscore_f1"].mean(),
            "avg_precision": group["bertscore_precision"].mean(),
            "avg_recall": group["bertscore_recall"].mean()
        }

    # 绘制雷达图：显示各个类别的 F1、Precision 和 Recall 的平均值
    st.subheader("Radar Chart: Average F1, Precision, and Recall by Category")
    radar_data = []
    for category, metrics in category_metrics.items():
        radar_data.append({
            "Category": category,
            "F1": metrics["avg_f1"],
            "Precision": metrics["avg_precision"],
            "Recall": metrics["avg_recall"]
        })
    radar_df = pd.DataFrame(radar_data)
    fig_radar = go.Figure()
    for category in radar_df["Category"]:
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df[radar_df["Category"] == category][["F1", "Precision", "Recall"]].values.flatten(),
            theta=["F1", "Precision", "Recall"],
            fill='toself',
            name=category
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # 分数范围设置为 0 到 1
            ),
            bgcolor='rgba(0,0,0,0)'  # 设置雷达图背景为透明
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # 设置整个图表背景为透明
        plot_bgcolor='rgba(0,0,0,0)',   # 设置绘图区域背景为透明
        showlegend=True
    )
    st.plotly_chart(fig_radar)

    if st.download_button(
        label="Download HTML Report",
        data=generate_bertscore_results_html_report(file_path),
        file_name="bertscore_results_report.html",
        mime="text/html"
    ):
        st.success("Download link generated successfully.")

def generate_bertscore_results_html_report(file_path):
    """生成 BERTScore 数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bert_score_data = pd.DataFrame(data)

    if bert_score_data.shape[0] == 0:
        return "<h2>No data to display for BERTScore results.</h2>"

    # 检查是否存在 category 字段
    if "category" not in bert_score_data.columns:
        return "<h2>No 'category' field found in the data. Cannot display metrics by category.</h2>"

    # 计算关键指标
    total_tests = bert_score_data.shape[0]
    successful_tests = bert_score_data["bertscore_f1"].notnull().sum()
    success_ratio = successful_tests / total_tests if total_tests > 0 else 0

    # 生成关键指标卡片
    overall_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Total Tests: {total_tests}</div>
            <div class="metric-item">Successful Tests: {successful_tests}</div>
            <div class="metric-item">Success Ratio: {success_ratio:.2%}</div>
        </div>
    </div>
    """

    # 生成 F1、Precision 和 Recall 的指标卡片
    f1_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average F1: {bert_score_data['bertscore_f1'].mean():.4f}</div>
            <div class="metric-item">Max F1: {bert_score_data['bertscore_f1'].max():.4f}</div>
            <div class="metric-item">Min F1: {bert_score_data['bertscore_f1'].min():.4f}</div>
        </div>
    </div>
    """

    precision_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average Precision: {bert_score_data['bertscore_precision'].mean():.4f}</div>
            <div class="metric-item">Max Precision: {bert_score_data['bertscore_precision'].max():.4f}</div>
            <div class="metric-item">Min Precision: {bert_score_data['bertscore_precision'].min():.4f}</div>
        </div>
    </div>
    """

    recall_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average Recall: {bert_score_data['bertscore_recall'].mean():.4f}</div>
            <div class="metric-item">Max Recall: {bert_score_data['bertscore_recall'].max():.4f}</div>
            <div class="metric-item">Min Recall: {bert_score_data['bertscore_recall'].min():.4f}</div>
        </div>
    </div>
    """

    # 生成 F1、Precision 和 Recall 的分布图
    hist_data = [
        bert_score_data["bertscore_f1"].dropna(),
        bert_score_data["bertscore_precision"].dropna(),
        bert_score_data["bertscore_recall"].dropna()
    ]
    hist_labels = ["F1", "Precision", "Recall"]
    fig_distplot = ff.create_distplot(hist_data, hist_labels, show_hist=False)
    distplot_html = fig_distplot.to_html(full_html=False)

    # 生成每个类别的 F1、Precision 和 Recall 的小提琴图
    violin_plots_html = ""
    metrics = ["bertscore_f1", "bertscore_precision", "bertscore_recall"]
    for metric in metrics:
        fig_violin = px.violin(
            bert_score_data,
            x="category",
            y=metric,
            template="plotly_white",
            color="category",
            box=True,
            points="all",
            title=f"{metric.capitalize()} by Category",
            labels={"category": "Category", metric: metric.capitalize()}
        )
        violin_plots_html += fig_violin.to_html(full_html=False, include_plotlyjs="cdn")

    # 计算每个类别的 F1、Precision 和 Recall 的平均值
    category_metrics = {}
    for category, group in bert_score_data.groupby("category"):
        category_metrics[category] = {
            "avg_f1": group["bertscore_f1"].mean(),
            "avg_precision": group["bertscore_precision"].mean(),
            "avg_recall": group["bertscore_recall"].mean()
        }

    # 生成雷达图
    radar_chart_html = ""
    if category_metrics:
        radar_data = []
        for category, metrics in category_metrics.items():
            radar_data.append({
                "Category": category,
                "F1": metrics["avg_f1"],
                "Precision": metrics["avg_precision"],
                "Recall": metrics["avg_recall"]
            })
        radar_df = pd.DataFrame(radar_data)
        fig_radar = go.Figure()
        for category in radar_df["Category"]:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df[radar_df["Category"] == category][["F1", "Precision", "Recall"]].values.flatten(),
                theta=["F1", "Precision", "Recall"],
                fill='toself',
                name=category
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        radar_chart_html = fig_radar.to_html(full_html=False, include_plotlyjs="cdn")

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>BERTScore Results Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f4f4f4;
            }}
            .container {{
                max-width: 1000px;
                background: white;
                margin: 0 auto;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }}
            h1, h2, h3 {{
                text-align: center;
                color: #333;
            }}
            .metric {{
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin: 20px 0;
            }}
            .metric-row {{
                display: flex;
                justify-content: space-between;
                gap: 10px;
            }}
            .metric-item {{
                flex: 1;
                text-align: center;
                padding: 10px 20px;
                background: #007BFF;
                color: white;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
            }}
            .chart-container {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BERTScore Results Report</h1>

            <h2>Key Metrics</h2>
            {overall_metrics_html}

            <h2>F1 Metrics</h2>
            {f1_metrics_html}

            <h2>Precision Metrics</h2>
            {precision_metrics_html}

            <h2>Recall Metrics</h2>
            {recall_metrics_html}

            <h2>Distribution of F1, Precision, and Recall</h2>
            <div class="chart-container">
                {distplot_html}
            </div>

            <h2>Violin Plots by Category</h2>
            <div class="chart-container">
                {violin_plots_html}
            </div>

            {f"<h2>Radar Chart: Average F1, Precision, and Recall by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_metrics else ""}
        </div>
    </body>
    </html>
    """

    return html_template