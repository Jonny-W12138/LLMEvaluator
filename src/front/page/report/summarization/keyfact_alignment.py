import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

def keyfact_alignment_render(file_path):
    st.write("keyfact_alignment")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keyfact_alignment_data = pd.DataFrame(data['results'])

    if keyfact_alignment_data.shape[0] == 0:
        st.error("No data to display.")
        return

    if "category" not in keyfact_alignment_data.columns:
        st.warning("No 'category' field found in the data. Cannot display metrics by category.")
        return

    total_num_col, success_num_col, success_ratio_col = st.columns(3)
    with total_num_col:
        st.metric("Total number", keyfact_alignment_data.shape[0])
    with success_num_col:
        st.metric("Success number", keyfact_alignment_data["success"].sum())
    with success_ratio_col:
        st.metric("Success ratio", f"{keyfact_alignment_data['success'].mean():.2%}")

    category_metrics = {}
    for category, group in keyfact_alignment_data.groupby("category"):
        success_data = group[group["success"]]
        all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]
        total_pred_labels = len(all_pred_labels)
        pred_labels_1_count = sum(1 for label in all_pred_labels if label == 1)
        pred_labels_1_ratio = pred_labels_1_count / total_pred_labels if total_pred_labels > 0 else 0

        category_metrics[category] = {
            "total_success": success_data.shape[0],
            "total_pred_labels": total_pred_labels,
            "pred_labels_1_count": pred_labels_1_count,
            "pred_labels_1_ratio": pred_labels_1_ratio
        }

    st.subheader("Metrics by Category")
    for category, metrics in category_metrics.items():
        st.markdown(f"**{category}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Success", metrics["total_success"])
        with col2:
            st.metric("Total pred_labels", metrics["total_pred_labels"])
        with col3:
            st.metric("Correct pred_labels", metrics["pred_labels_1_count"])
        with col4:
            st.metric("Correct pred_labels ratio", f"{metrics['pred_labels_1_ratio']:.2%}")

    stacked_bar_data = []
    for category, metrics in category_metrics.items():
        total_pred_labels = metrics["total_pred_labels"]
        correct_pred_labels = metrics["pred_labels_1_count"]
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

    radar_data = []
    for category, metrics in category_metrics.items():
        radar_data.append({
            "Category": category,
            "Alignment Ratio": metrics["pred_labels_1_ratio"]
        })
    radar_df = pd.DataFrame(radar_data)
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["Alignment Ratio"],
        theta=radar_df["Category"],
        fill='toself',
        name="Alignment Ratio"
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_radar)

    if st.button("Generate HTML Report", key="generate_keyfact_alignment_html_report"):
        generate_keyfact_alignment_html_report(file_path)

def generate_keyfact_alignment_html_report(file_path, output_path="keyfact_alignment_report.html"):
    """生成关键事实对齐数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keyfact_alignment_data = pd.DataFrame(data['results'])

    if keyfact_alignment_data.shape[0] == 0:
        return "<h2>No data to display for keyfact alignment.</h2>"

    # 检查是否存在 category 字段
    if "category" not in keyfact_alignment_data.columns:
        return "<h2>No 'category' field found in the data. Cannot display metrics by category.</h2>"

    # 计算关键指标
    total_tests = keyfact_alignment_data.shape[0]
    successful_tests = keyfact_alignment_data["success"].sum()
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
    for category, group in keyfact_alignment_data.groupby("category"):
        success_data = group[group["success"]]
        all_pred_labels = [item for sublist in success_data["pred_labels"] for item in sublist]
        total_pred_labels = len(all_pred_labels)
        pred_labels_1_count = sum(1 for label in all_pred_labels if label == 1)
        pred_labels_1_ratio = pred_labels_1_count / total_pred_labels if total_pred_labels > 0 else 0

        category_metrics[category] = {
            "total_success": success_data.shape[0],
            "total_pred_labels": total_pred_labels,
            "pred_labels_1_count": pred_labels_1_count,
            "pred_labels_1_ratio": pred_labels_1_ratio
        }

    # 生成类别指标表格
    category_table_html = ""
    if category_metrics:
        category_details = [
            {
                "Category": category,
                "Total Success": metrics["total_success"],
                "Total pred_labels": metrics["total_pred_labels"],
                "Correct pred_labels": metrics["pred_labels_1_count"],
                "Correct pred_labels ratio": f"{metrics['pred_labels_1_ratio']:.2%}"
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
            correct_pred_labels = metrics["pred_labels_1_count"]
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
        stacked_bar_html = fig_stacked_bar.to_html(full_html=False)

    # 生成雷达图
    radar_chart_html = ""
    if category_metrics:
        radar_data = []
        for category, metrics in category_metrics.items():
            radar_data.append({
                "Category": category,
                "Alignment Ratio": metrics["pred_labels_1_ratio"]
            })
        radar_df = pd.DataFrame(radar_data)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Alignment Ratio"],
            theta=radar_df["Category"],
            fill='toself',
            name="Alignment Ratio"
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
        radar_chart_html = fig_radar.to_html(full_html=False)

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Keyfact Alignment Report</title>
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
            <h1>Keyfact Alignment Report</h1>

            <h2>Key Metrics</h2>
            {overall_metrics_html}

            {f"<h2>Metrics by Category</h2><div class='table-container'>{category_table_html}</div>" if category_metrics else ""}

            {f"<h2>Alignment by Category</h2><div class='chart-container'>{stacked_bar_html}</div>" if category_metrics else ""}

            {f"<h2>Alignment Ratio by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_metrics else ""}
        </div>
    </body>
    </html>
    """

    # 保存为 HTML 文件
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_template)

    return output_path