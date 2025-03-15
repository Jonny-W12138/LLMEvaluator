import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from pygments.lexer import include


def rouge_results_render(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rouge_data = pd.DataFrame(data)

    if rouge_data.shape[0] == 0:
        st.error("No data to display.")
        return

    # 检查是否存在 category 字段
    if "category" not in rouge_data.columns:
        st.warning("No 'category' field found in the data. Cannot display metrics by category.")
        return

    # 总体指标
    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", rouge_data.shape[0])
    with success_num_col:
        st.metric("Success number", rouge_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{rouge_data['success'].mean():.2%}")

    # 显示 ROUGE-1 指标
    ave_rouge_1_f_col, max_rouge_1_f_col, min_rouge_1_f_col, ave_rouge_1_p_col, max_rouge_1_p_col, min_rouge_1_p_col, ave_rouge_1_r_col, max_rouge_1_r_col, min_rouge_1_r_col = st.columns(9)
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

    # 显示 ROUGE-2 指标
    ave_rouge_2_f_col, max_rouge_2_f_col, min_rouge_2_f_col, ave_rouge_2_p_col, max_rouge_2_p_col, min_rouge_2_p_col, ave_rouge_2_r_col, max_rouge_2_r_col, min_rouge_2_r_col = st.columns(9)
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

    # 显示 ROUGE-L 指标
    ave_rouge_l_f_col, max_rouge_l_f_col, min_rouge_l_f_col, ave_rouge_l_p_col, max_rouge_l_p_col, min_rouge_l_p_col, ave_rouge_l_r_col, max_rouge_l_r_col, min_rouge_l_r_col = st.columns(9)
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

    # 绘制 ROUGE-1、ROUGE-2 和 ROUGE-L 的分布图
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

    # 按类别分组计算指标
    category_metrics = {}
    for category, group in rouge_data.groupby("category"):
        category_metrics[category] = {
            "avg_rouge_1_f": group["rouge-1-f"].mean(),
            "avg_rouge_1_p": group["rouge-1-p"].mean(),
            "avg_rouge_1_r": group["rouge-1-r"].mean(),
            "avg_rouge_2_f": group["rouge-2-f"].mean(),
            "avg_rouge_2_p": group["rouge-2-p"].mean(),
            "avg_rouge_2_r": group["rouge-2-r"].mean(),
            "avg_rouge_l_f": group["rouge-l-f"].mean(),
            "avg_rouge_l_p": group["rouge-l-p"].mean(),
            "avg_rouge_l_r": group["rouge-l-r"].mean()
        }

    # 绘制每个类别的 ROUGE 指标的小提琴图
    st.subheader("Violin Plots by Category")
    metrics = ["rouge-1-f", "rouge-1-p", "rouge-1-r", "rouge-2-f", "rouge-2-p", "rouge-2-r", "rouge-l-f", "rouge-l-p", "rouge-l-r"]
    for metric in metrics:
        fig_violin = px.violin(
            rouge_data,
            x="category",
            y=metric,
            color="category",
            box=True,  # 显示箱线图
            points="all",  # 显示所有数据点
            title=f"{metric.capitalize()} by Category",
            labels={"category": "Category", metric: metric.capitalize()}
        )
        st.plotly_chart(fig_violin)

    # 绘制雷达图：显示各个类别的 ROUGE 指标的平均值
    st.subheader("Radar Chart: Average ROUGE Scores by Category")
    category_metrics = {}
    for category, group in rouge_data.groupby("category"):
        category_metrics[category] = {
            "avg_rouge_1_f": group["rouge-1-f"].mean(),
            "avg_rouge_1_p": group["rouge-1-p"].mean(),
            "avg_rouge_1_r": group["rouge-1-r"].mean(),
            "avg_rouge_2_f": group["rouge-2-f"].mean(),
            "avg_rouge_2_p": group["rouge-2-p"].mean(),
            "avg_rouge_2_r": group["rouge-2-r"].mean(),
            "avg_rouge_l_f": group["rouge-l-f"].mean(),
            "avg_rouge_l_p": group["rouge-l-p"].mean(),
            "avg_rouge_l_r": group["rouge-l-r"].mean()
        }

    # 准备雷达图数据
    radar_data = []
    for category, metrics in category_metrics.items():
        radar_data.append({
            "Category": category,
            "ROUGE-1 F1": metrics["avg_rouge_1_f"],
            "ROUGE-1 Precision": metrics["avg_rouge_1_p"],
            "ROUGE-1 Recall": metrics["avg_rouge_1_r"],
            "ROUGE-2 F1": metrics["avg_rouge_2_f"],
            "ROUGE-2 Precision": metrics["avg_rouge_2_p"],
            "ROUGE-2 Recall": metrics["avg_rouge_2_r"],
            "ROUGE-L F1": metrics["avg_rouge_l_f"],
            "ROUGE-L Precision": metrics["avg_rouge_l_p"],
            "ROUGE-L Recall": metrics["avg_rouge_l_r"]
        })
    radar_df = pd.DataFrame(radar_data)
    fig_radar = go.Figure()

    # 添加每个类别的雷达图数据
    for category in radar_df["Category"]:
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df[radar_df["Category"] == category][
                ["ROUGE-1 F1", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-2 F1", "ROUGE-2 Precision",
                 "ROUGE-2 Recall", "ROUGE-L F1", "ROUGE-L Precision", "ROUGE-L Recall"]].values.flatten(),
            theta=["ROUGE-1 F1", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-2 F1", "ROUGE-2 Precision",
                   "ROUGE-2 Recall", "ROUGE-L F1", "ROUGE-L Precision", "ROUGE-L Recall"],
            fill='toself',
            name=category
        ))

    # 设置雷达图布局
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # 分数范围设置为 0 到 1
            ),
            bgcolor='rgba(0,0,0,0)'  # 设置雷达图背景为透明
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # 设置整个图表背景为透明
        plot_bgcolor='rgba(0,0,0,0)',  # 设置绘图区域背景为透明
        showlegend=True  # 显示图例
    )

    # 显示雷达图
    st.plotly_chart(fig_radar)

    if st.download_button(
            label="Download HTML Report",
            data=generate_rouge_results_html_report(file_path),
            file_name="rouge_results_report.html",
            mime="text/html"
    ):
        st.success("Download link generated successfully.")



def generate_rouge_results_html_report(file_path):
    """生成 ROUGE 数据的 HTML 报告"""
    import json
    import pandas as pd
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rouge_data = pd.DataFrame(data)

    if rouge_data.shape[0] == 0:
        return "<h2>No data to display for ROUGE results.</h2>"

    # 检查是否存在 category 字段
    if "category" not in rouge_data.columns:
        return "<h2>No 'category' field found in the data. Cannot display metrics by category.</h2>"

    # 计算关键指标
    total_tests = rouge_data.shape[0]
    successful_tests = rouge_data["success"].sum()
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

    # 生成 ROUGE-1 指标卡片
    rouge_1_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-1 F1: {rouge_data['rouge-1-f'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-1 F1: {rouge_data['rouge-1-f'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-1 F1: {rouge_data['rouge-1-f'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-1 Precision: {rouge_data['rouge-1-p'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-1 Precision: {rouge_data['rouge-1-p'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-1 Precision: {rouge_data['rouge-1-p'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-1 Recall: {rouge_data['rouge-1-r'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-1 Recall: {rouge_data['rouge-1-r'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-1 Recall: {rouge_data['rouge-1-r'].min():.4f}</div>
        </div>
    </div>
    """

    # 生成 ROUGE-2 指标卡片
    rouge_2_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-2 F1: {rouge_data['rouge-2-f'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-2 F1: {rouge_data['rouge-2-f'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-2 F1: {rouge_data['rouge-2-f'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-2 Precision: {rouge_data['rouge-2-p'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-2 Precision: {rouge_data['rouge-2-p'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-2 Precision: {rouge_data['rouge-2-p'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-2 Recall: {rouge_data['rouge-2-r'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-2 Recall: {rouge_data['rouge-2-r'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-2 Recall: {rouge_data['rouge-2-r'].min():.4f}</div>
        </div>
    </div>
    """

    # 生成 ROUGE-L 指标卡片
    rouge_l_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-L F1: {rouge_data['rouge-l-f'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-L F1: {rouge_data['rouge-l-f'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-L F1: {rouge_data['rouge-l-f'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-L Precision: {rouge_data['rouge-l-p'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-L Precision: {rouge_data['rouge-l-p'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-L Precision: {rouge_data['rouge-l-p'].min():.4f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-item">Average ROUGE-L Recall: {rouge_data['rouge-l-r'].mean():.4f}</div>
            <div class="metric-item">Max ROUGE-L Recall: {rouge_data['rouge-l-r'].max():.4f}</div>
            <div class="metric-item">Min ROUGE-L Recall: {rouge_data['rouge-l-r'].min():.4f}</div>
        </div>
    </div>
    """

    # 生成 ROUGE-1、ROUGE-2 和 ROUGE-L 的分布图
    rouge_1_hist_data = [
        rouge_data["rouge-1-f"].dropna(),
        rouge_data["rouge-1-p"].dropna(),
        rouge_data["rouge-1-r"].dropna()
    ]
    rouge_1_hist_labels = ["F1", "Precision", "Recall"]
    fig_rouge_1 = ff.create_distplot(rouge_1_hist_data, rouge_1_hist_labels, show_hist=False)
    rouge_1_distplot_html = fig_rouge_1.to_html(full_html=False, include_plotlyjs="cdn")

    rouge_2_hist_data = [
        rouge_data["rouge-2-f"].dropna(),
        rouge_data["rouge-2-p"].dropna(),
        rouge_data["rouge-2-r"].dropna()
    ]
    rouge_2_hist_labels = ["F1", "Precision", "Recall"]
    fig_rouge_2 = ff.create_distplot(rouge_2_hist_data, rouge_2_hist_labels, show_hist=False)
    rouge_2_distplot_html = fig_rouge_2.to_html(full_html=False, include_plotlyjs="cdn")

    rouge_l_hist_data = [
        rouge_data["rouge-l-f"].dropna(),
        rouge_data["rouge-l-p"].dropna(),
        rouge_data["rouge-l-r"].dropna()
    ]
    rouge_l_hist_labels = ["F1", "Precision", "Recall"]
    fig_rouge_l = ff.create_distplot(rouge_l_hist_data, rouge_l_hist_labels, show_hist=False)
    rouge_l_distplot_html = fig_rouge_l.to_html(full_html=False, include_plotlyjs="cdn")

    # 计算每个类别的指标
    category_metrics = {}
    for category, group in rouge_data.groupby("category"):
        category_metrics[category] = {
            "avg_rouge_1_f": group["rouge-1-f"].mean(),
            "avg_rouge_1_p": group["rouge-1-p"].mean(),
            "avg_rouge_1_r": group["rouge-1-r"].mean(),
            "avg_rouge_2_f": group["rouge-2-f"].mean(),
            "avg_rouge_2_p": group["rouge-2-p"].mean(),
            "avg_rouge_2_r": group["rouge-2-r"].mean(),
            "avg_rouge_l_f": group["rouge-l-f"].mean(),
            "avg_rouge_l_p": group["rouge-l-p"].mean(),
            "avg_rouge_l_r": group["rouge-l-r"].mean()
        }

    # 生成类别指标表格
    category_table_html = ""
    if category_metrics:
        category_details = [
            {
                "Category": category,
                "ROUGE-1 F1": f"{metrics['avg_rouge_1_f']:.4f}",
                "ROUGE-1 Precision": f"{metrics['avg_rouge_1_p']:.4f}",
                "ROUGE-1 Recall": f"{metrics['avg_rouge_1_r']:.4f}",
                "ROUGE-2 F1": f"{metrics['avg_rouge_2_f']:.4f}",
                "ROUGE-2 Precision": f"{metrics['avg_rouge_2_p']:.4f}",
                "ROUGE-2 Recall": f"{metrics['avg_rouge_2_r']:.4f}",
                "ROUGE-L F1": f"{metrics['avg_rouge_l_f']:.4f}",
                "ROUGE-L Precision": f"{metrics['avg_rouge_l_p']:.4f}",
                "ROUGE-L Recall": f"{metrics['avg_rouge_l_r']:.4f}"
            }
            for category, metrics in category_metrics.items()
        ]
        df = pd.DataFrame(category_details)
        category_table_html = df.to_html(classes="table table-striped", index=False)

    # 生成每个类别的 ROUGE 指标的小提琴图
    violin_plot_html = ""
    if category_metrics:
        metrics = ["rouge-1-f", "rouge-1-p", "rouge-1-r", "rouge-2-f", "rouge-2-p", "rouge-2-r", "rouge-l-f", "rouge-l-p", "rouge-l-r"]
        violin_plots = []
        for metric in metrics:
            fig_violin = px.violin(
                rouge_data,
                x="category",
                y=metric,
                template="plotly_white",
                color="category",
                box=True,
                points="all",
                title=f"{metric.capitalize()} by Category",
                labels={"category": "Category", metric: metric.capitalize()}
            )
            violin_plots.append(fig_violin.to_html(full_html=False, include_plotlyjs="cdn"))
        violin_plot_html = "".join([f"<div class='chart-container'>{plot}</div>" for plot in violin_plots])

    # 生成雷达图：显示各个类别的 ROUGE 指标的平均值
    radar_chart_html = ""
    if category_metrics:
        radar_data = []
        for category, metrics in category_metrics.items():
            radar_data.append({
                "Category": category,
                "ROUGE-1 F1": metrics["avg_rouge_1_f"],
                "ROUGE-1 Precision": metrics["avg_rouge_1_p"],
                "ROUGE-1 Recall": metrics["avg_rouge_1_r"],
                "ROUGE-2 F1": metrics["avg_rouge_2_f"],
                "ROUGE-2 Precision": metrics["avg_rouge_2_p"],
                "ROUGE-2 Recall": metrics["avg_rouge_2_r"],
                "ROUGE-L F1": metrics["avg_rouge_l_f"],
                "ROUGE-L Precision": metrics["avg_rouge_l_p"],
                "ROUGE-L Recall": metrics["avg_rouge_l_r"]
            })
        radar_df = pd.DataFrame(radar_data)
        fig_radar = go.Figure()
        for category in radar_df["Category"]:
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df[radar_df["Category"] == category][
                    ["ROUGE-1 F1", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-2 F1", "ROUGE-2 Precision",
                     "ROUGE-2 Recall", "ROUGE-L F1", "ROUGE-L Precision", "ROUGE-L Recall"]].values.flatten(),
                theta=["ROUGE-1 F1", "ROUGE-1 Precision", "ROUGE-1 Recall", "ROUGE-2 F1", "ROUGE-2 Precision",
                       "ROUGE-2 Recall", "ROUGE-L F1", "ROUGE-L Precision", "ROUGE-L Recall"],
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
        <title>ROUGE Results Report</title>
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
            .table-container {{
                margin-top: 20px;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .table th, .table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .table th {{
                background-color: #007BFF;
                color: white;
            }}
            .chart-container {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ROUGE Results Report</h1>

            <h2>Key Metrics</h2>
            {overall_metrics_html}

            <h2>ROUGE-1 Metrics</h2>
            {rouge_1_metrics_html}

            <h2>ROUGE-2 Metrics</h2>
            {rouge_2_metrics_html}

            <h2>ROUGE-L Metrics</h2>
            {rouge_l_metrics_html}

            <h2>Distribution of ROUGE-1 Scores</h2>
            <div class="chart-container">
                {rouge_1_distplot_html}
            </div>

            <h2>Distribution of ROUGE-2 Scores</h2>
            <div class="chart-container">
                {rouge_2_distplot_html}
            </div>

            <h2>Distribution of ROUGE-L Scores</h2>
            <div class="chart-container">
                {rouge_l_distplot_html}
            </div>

            {f"<h2>Metrics by Category</h2><div class='table-container'>{category_table_html}</div>" if category_metrics else ""}

            {f"<h2>Violin Plots by Category</h2><div class='chart-container'>{violin_plot_html}</div>" if category_metrics else ""}

            {f"<h2>Radar Chart: Average ROUGE Scores by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_metrics else ""}
        </div>
    </body>
    </html>
    """

    return html_template
