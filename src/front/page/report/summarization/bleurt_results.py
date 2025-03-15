import json
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from htmlmin import minify

def bleurt_results_render(file_path):
    st.write("bleurt_results")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bleurt_data = pd.DataFrame(data)

    if bleurt_data.shape[0] == 0:
        st.error("No data to display.")
        return

    # 检查是否存在 category 字段
    if "category" not in bleurt_data.columns:
        st.warning("No 'category' field found in the data. Cannot display metrics by category.")
        return

    # 总体指标
    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", bleurt_data.shape[0])
    with success_num_col:
        st.metric("Success number", bleurt_data["score"].notnull().sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{bleurt_data['score'].notnull().mean():.2%}")

    # 显示 BLEURT Score 的总体指标
    ave_score_col, max_score_col, min_score_col = st.columns(3)
    with ave_score_col:
        st.metric("Average score", f"{bleurt_data['score'].mean():.4f}")
    with max_score_col:
        st.metric("Max score", f"{bleurt_data['score'].max():.4f}")
    with min_score_col:
        st.metric("Min score", f"{bleurt_data['score'].min():.4f}")

    # 绘制 BLEURT Score 的分布图
    hist_data = [
        bleurt_data["score"].dropna(),
    ]
    hist_labels = ["BLEURT Score"]
    fig_distplot = ff.create_distplot(hist_data, hist_labels, show_hist=False)
    st.plotly_chart(fig_distplot)

    # 按类别分组计算指标
    category_metrics = {}
    for category, group in bleurt_data.groupby("category"):
        category_metrics[category] = {
            "total_success": group["score"].notnull().sum(),
            "total_scores": group["score"].dropna().shape[0],
            "avg_score": group["score"].mean(),
            "max_score": group["score"].max(),
            "min_score": group["score"].min()
        }

    # 显示各个类别的指标
    st.subheader("Metrics by Category")
    for category, metrics in category_metrics.items():
        st.markdown(f"**{category}**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Success", metrics["total_success"])
        with col2:
            st.metric("Total Scores", metrics["total_scores"])
        with col3:
            st.metric("Average score", f"{metrics['avg_score']:.4f}")
        with col4:
            st.metric("Max score", f"{metrics['max_score']:.4f}")
        with col5:
            st.metric("Min score", f"{metrics['min_score']:.4f}")

    # 绘制每个类别的 BLEURT Score 的小提琴图
    st.subheader("Violin Plots by Category")
    fig_violin = px.violin(
        bleurt_data,
        x="category",
        y="score",
        color="category",
        box=True,  # 显示箱线图
        points="all",  # 显示所有数据点
        title="BLEURT Score by Category",
        labels={"category": "Category", "score": "BLEURT Score"}
    )
    st.plotly_chart(fig_violin)

    # 计算每个类别的 BLEURT Score 的平均值
    radar_data = []
    for category, metrics in category_metrics.items():
        radar_data.append({
            "Category": category,
            "BLEURT Score": metrics["avg_score"]
        })
    radar_df = pd.DataFrame(radar_data)

    # 计算 BLEURT Score 的最小值和最大值
    min_score = radar_df["BLEURT Score"].min()
    max_score = radar_df["BLEURT Score"].max()

    # 绘制雷达图：显示各个类别的 BLEURT Score 的平均值
    st.subheader("Radar Chart: Average BLEURT Score by Category")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["BLEURT Score"],
        theta=radar_df["Category"],
        fill='toself',
        name="BLEURT Score"
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min_score, max_score]  # 动态设置范围
            ),
            bgcolor='rgba(0,0,0,0)'  # 设置雷达图背景为透明
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # 设置整个图表背景为透明
        plot_bgcolor='rgba(0,0,0,0)',  # 设置绘图区域背景为透明
        showlegend=True
    )
    st.plotly_chart(fig_radar)

    if st.download_button(
        label="Download BLEURT Results Report",
        data=minify(generate_bleurt_results_html_report(file_path)),
        file_name="bleurt_results_report.html",
        mime="text/html"
    ):
        st.success("Report downloaded successfully.")

def generate_bleurt_results_html_report(file_path):
    """生成 BLEURT 数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bleurt_data = pd.DataFrame(data)

    if bleurt_data.shape[0] == 0:
        return "<h2>No data to display for BLEURT results.</h2>"

    # 检查是否存在 category 字段
    if "category" not in bleurt_data.columns:
        return "<h2>No 'category' field found in the data. Cannot display metrics by category.</h2>"

    # 计算关键指标
    total_tests = bleurt_data.shape[0]
    successful_tests = bleurt_data["score"].notnull().sum()
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

    # 生成 BLEURT Score 的指标卡片
    score_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Average Score: {bleurt_data['score'].mean():.4f}</div>
            <div class="metric-item">Max Score: {bleurt_data['score'].max():.4f}</div>
            <div class="metric-item">Min Score: {bleurt_data['score'].min():.4f}</div>
        </div>
    </div>
    """

    # 生成 BLEURT Score 的分布图
    hist_data = [
        bleurt_data["score"].dropna(),
    ]
    hist_labels = ["BLEURT Score"]
    fig_distplot = ff.create_distplot(hist_data, hist_labels, show_hist=False)
    distplot_html = fig_distplot.to_html(full_html=False)

    # 计算每个类别的指标
    category_metrics = {}
    for category, group in bleurt_data.groupby("category"):
        category_metrics[category] = {
            "total_success": group["score"].notnull().sum(),
            "total_scores": group["score"].dropna().shape[0],
            "avg_score": group["score"].mean(),
            "max_score": group["score"].max(),
            "min_score": group["score"].min()
        }

    # 生成类别指标表格
    category_table_html = ""
    if category_metrics:
        category_details = [
            {
                "Category": category,
                "Total Success": metrics["total_success"],
                "Total Scores": metrics["total_scores"],
                "Average Score": f"{metrics['avg_score']:.4f}",
                "Max Score": f"{metrics['max_score']:.4f}",
                "Min Score": f"{metrics['min_score']:.4f}"
            }
            for category, metrics in category_metrics.items()
        ]
        df = pd.DataFrame(category_details)
        category_table_html = df.to_html(classes="table table-striped", index=False)

    # 生成每个类别的 BLEURT Score 的小提琴图
    violin_plot_html = ""
    if category_metrics:
        fig_violin = px.violin(
            bleurt_data,
            x="category",
            y="score",
            template="plotly_white",
            color="category",
            box=True,
            points="all",
            title="BLEURT Score by Category",
            labels={"category": "Category", "score": "BLEURT Score"}
        )
        violin_plot_html = fig_violin.to_html(full_html=False)

    # 生成雷达图
    radar_chart_html = ""
    if category_metrics:
        radar_data = []
        for category, metrics in category_metrics.items():
            radar_data.append({
                "Category": category,
                "BLEURT Score": metrics["avg_score"]
            })
        radar_df = pd.DataFrame(radar_data)

        # 计算 BLEURT Score 的最小值和最大值
        min_score = radar_df["BLEURT Score"].min()
        max_score = radar_df["BLEURT Score"].max()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["BLEURT Score"],
            theta=radar_df["Category"],
            fill='toself',
            name="BLEURT Score"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min_score, max_score]  # 动态设置范围
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        radar_chart_html = fig_radar.to_html(full_html=False, include_plotlyjs='cdn')

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>BLEURT Results Report</title>
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
            <h1>BLEURT Results Report</h1>

            <h2>Key Metrics</h2>
            {overall_metrics_html}

            <h2>BLEURT Score Metrics</h2>
            {score_metrics_html}

            <h2>Distribution of BLEURT Score</h2>
            <div class="chart-container">
                {distplot_html}
            </div>

            {f"<h2>Metrics by Category</h2><div class='table-container'>{category_table_html}</div>" if category_metrics else ""}

            {f"<h2>Violin Plots by Category</h2><div class='chart-container'>{violin_plot_html}</div>" if category_metrics else ""}

            {f"<h2>Radar Chart: Average BLEURT Score by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_metrics else ""}
        </div>
    </body>
    </html>
    """

    return html_template