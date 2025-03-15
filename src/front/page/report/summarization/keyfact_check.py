import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

def keyfact_check_render(file_path):
    st.write("keyfact_check")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keyfact_check_data = pd.DataFrame(data)

    if keyfact_check_data.shape[0] == 0:
        st.error("No data to display.")
        return

    # 检查是否存在 category 字段
    if "category" not in keyfact_check_data.columns:
        st.warning("No 'category' field found in the data. Cannot display metrics by category.")
        return

    # 总体指标
    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", keyfact_check_data.shape[0])
    with success_num_col:
        st.metric("Success number", keyfact_check_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{keyfact_check_data['success'].mean():.2%}")

    # 按类别分组计算指标
    category_metrics = {}
    for category, group in keyfact_check_data.groupby("category"):
        success_data = group[group["success"]]
        all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]
        total_pred_labels = len(all_pred_labels)
        fact_correct_labels_count = sum(1 for label in all_pred_labels if label == 0)
        fact_correct_labels_ratio = fact_correct_labels_count / total_pred_labels if total_pred_labels > 0 else 0

        category_metrics[category] = {
            "total_success": success_data.shape[0],
            "total_pred_labels": total_pred_labels,
            "fact_correct_labels_count": fact_correct_labels_count,
            "fact_correct_labels_ratio": fact_correct_labels_ratio
        }

    # 显示各个类别的指标
    st.subheader("Metrics by Category")
    for category, metrics in category_metrics.items():
        st.markdown(f"**{category}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Success", metrics["total_success"])
        with col2:
            st.metric("Total pred_labels", metrics["total_pred_labels"])
        with col3:
            st.metric("Correct pred_labels", metrics["fact_correct_labels_count"])
        with col4:
            st.metric("Correct pred_labels ratio", f"{metrics['fact_correct_labels_ratio']:.2%}")

    # 堆叠条形图：显示各个类别的 Correct pred_labels 和 Incorrect pred_labels
    st.subheader("Stacked Bar Chart: Alignment by Category")
    stacked_bar_data = []
    for category, metrics in category_metrics.items():
        total_pred_labels = metrics["total_pred_labels"]
        correct_pred_labels = metrics["fact_correct_labels_count"]
        incorrect_pred_labels = total_pred_labels - correct_pred_labels
        stacked_bar_data.append({
            "Category": category,
            "Correct pred_labels": correct_pred_labels,
            "Incorrect pred_labels": incorrect_pred_labels
        })
    stacked_bar_df = pd.DataFrame(stacked_bar_data)
    fig_stacked_bar = px.bar(
        stacked_bar_df,
        x="Category",
        y=["Correct pred_labels", "Incorrect pred_labels"],
        title="Alignment by Category",
        labels={"value": "Count", "variable": "Status"},
        color_discrete_map={"Correct pred_labels": "lightgreen", "Incorrect pred_labels": "lightcoral"}
    )
    st.plotly_chart(fig_stacked_bar)

    # 雷达图：显示各个类别的 Correct pred_labels ratio
    st.subheader("Radar Chart: Alignment Ratio by Category")
    radar_data = []
    for category, metrics in category_metrics.items():
        radar_data.append({
            "Category": category,
            "Correct pred_labels ratio": metrics["fact_correct_labels_ratio"]
        })
    radar_df = pd.DataFrame(radar_data)
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["Correct pred_labels ratio"],
        theta=radar_df["Category"],
        fill='toself',
        name="Correct pred_labels ratio"
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
        label="Download Keyfact Check Report",
        data=generate_keyfact_check_html_report(file_path),
        file_name="keyfact_check_report.html",
        mime="text/html"
    ):
        st.success("Report downloaded successfully.")

def generate_keyfact_check_html_report(file_path):
    """生成关键事实检查数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keyfact_check_data = pd.DataFrame(data)

    if keyfact_check_data.shape[0] == 0:
        return "<h2>No data to display for keyfact check.</h2>"

    # 检查是否存在 category 字段
    if "category" not in keyfact_check_data.columns:
        return "<h2>No 'category' field found in the data. Cannot display metrics by category.</h2>"

    # 计算关键指标
    total_tests = keyfact_check_data.shape[0]
    successful_tests = keyfact_check_data["success"].sum()
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

    # 计算每个类别的指标
    category_metrics = {}
    for category, group in keyfact_check_data.groupby("category"):
        success_data = group[group["success"]]
        all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]
        total_pred_labels = len(all_pred_labels)
        fact_correct_labels_count = sum(1 for label in all_pred_labels if label == 0)
        fact_correct_labels_ratio = fact_correct_labels_count / total_pred_labels if total_pred_labels > 0 else 0

        category_metrics[category] = {
            "total_success": success_data.shape[0],
            "total_pred_labels": total_pred_labels,
            "fact_correct_labels_count": fact_correct_labels_count,
            "fact_correct_labels_ratio": fact_correct_labels_ratio
        }

    # 生成类别指标表格
    category_table_html = ""
    if category_metrics:
        category_details = [
            {
                "Category": category,
                "Total Success": metrics["total_success"],
                "Total pred_labels": metrics["total_pred_labels"],
                "Correct pred_labels": metrics["fact_correct_labels_count"],
                "Correct pred_labels ratio": f"{metrics['fact_correct_labels_ratio']:.2%}"
            }
            for category, metrics in category_metrics.items()
        ]
        df = pd.DataFrame(category_details)
        category_table_html = df.to_html(classes="table table-striped", index=False)

    # 生成堆叠条形图
    stacked_bar_html = ""
    if category_metrics:
        stacked_bar_data = []
        for category, metrics in category_metrics.items():
            total_pred_labels = metrics["total_pred_labels"]
            correct_pred_labels = metrics["fact_correct_labels_count"]
            incorrect_pred_labels = total_pred_labels - correct_pred_labels
            stacked_bar_data.append({
                "Category": category,
                "Correct pred_labels": correct_pred_labels,
                "Incorrect pred_labels": incorrect_pred_labels
            })
        stacked_bar_df = pd.DataFrame(stacked_bar_data)
        fig_stacked_bar = px.bar(
            stacked_bar_df,
            x="Category",
            y=["Correct pred_labels", "Incorrect pred_labels"],
            title="Alignment by Category",
            template="plotly_white",
            labels={"value": "Count", "variable": "Status"},
        )
        stacked_bar_html = fig_stacked_bar.to_html(full_html=False, include_plotlyjs="cdn")

    # 生成雷达图
    radar_chart_html = ""
    if category_metrics:
        radar_data = []
        for category, metrics in category_metrics.items():
            radar_data.append({
                "Category": category,
                "Correct pred_labels ratio": metrics["fact_correct_labels_ratio"]
            })
        radar_df = pd.DataFrame(radar_data)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Correct pred_labels ratio"],
            theta=radar_df["Category"],
            fill='toself',
            name="Correct pred_labels ratio"
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
        <title>Keyfact Check Report</title>
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
            <h1>Keyfact Check Report</h1>

            <h2>Key Metrics</h2>
            {overall_metrics_html}

            {f"<h2>Metrics by Category</h2><div class='table-container'>{category_table_html}</div>" if category_metrics else ""}

            {f"<h2>Alignment by Category</h2><div class='chart-container'>{stacked_bar_html}</div>" if category_metrics else ""}

            {f"<h2>Alignment Ratio by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_metrics else ""}
        </div>
    </body>
    </html>
    """

    return html_template