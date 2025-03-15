import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def llm_score_render(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取总体分数和类别分数
    overall_scores = data['overall_scores']
    category_scores = data['category_scores']

    # 展示总体分数
    st.subheader("Overall Scores")
    faithfulness_col, completeness_col, conciseness_col = st.columns(3)
    with faithfulness_col:
        st.metric("Faithfulness", f"{overall_scores['faithfulness_score']:.4f}")
    with completeness_col:
        st.metric("Completeness", f"{overall_scores['completeness_score']:.4f}")
    with conciseness_col:
        st.metric("Conciseness", f"{overall_scores['conciseness_score']:.4f}")

    # 展示每个类别的分数
    if category_scores:
        st.subheader("Category Scores")
        for category, scores in category_scores.items():
            st.markdown(f"**{category}**")
            faithfulness_col, completeness_col, conciseness_col = st.columns(3)
            with faithfulness_col:
                st.metric("Faithfulness", f"{scores['faithfulness_score']:.4f}")
            with completeness_col:
                st.metric("Completeness", f"{scores['completeness_score']:.4f}")
            with conciseness_col:
                st.metric("Conciseness", f"{scores['conciseness_score']:.4f}")

    # 绘制雷达图
    if category_scores:
        st.subheader("Radar Chart of Scores by Category")
        categories = list(category_scores.keys())
        metrics = ["Faithfulness", "Completeness", "Conciseness"]

        # 准备雷达图数据
        radar_data = []
        for category in categories:
            scores = [
                category_scores[category]['faithfulness_score'],
                category_scores[category]['completeness_score'],
                category_scores[category]['conciseness_score']
            ]
            radar_data.append(go.Scatterpolar(
                r=scores,
                theta=metrics,
                fill='toself',
                name=category
            ))


        fig = go.Figure(data=radar_data)
        fig.update_layout(
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
        st.plotly_chart(fig)
    else:
        st.warning("No category scores available to display radar chart.")

    if st.download_button(
        label="Download HTML Report",
        data=generate_llm_score_html_report(file_path),
        file_name="llm_score_report.html",
        mime="text/html"
    ):
        st.success("HTML report downloaded successfully.")

def generate_llm_score_html_report(file_path):
    """生成 LLM 评分数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取总体分数和类别分数
    overall_scores = data['overall_scores']
    category_scores = data['category_scores']

    # 生成总体分数卡片
    overall_metrics_html = f"""
    <div class="metric">
        <div class="metric-row">
            <div class="metric-item">Faithfulness: {overall_scores['faithfulness_score']:.4f}</div>
            <div class="metric-item">Completeness: {overall_scores['completeness_score']:.4f}</div>
            <div class="metric-item">Conciseness: {overall_scores['conciseness_score']:.4f}</div>
        </div>
    </div>
    """

    # 生成类别分数表格
    category_table_html = ""
    if category_scores:
        category_details = [
            {
                "Category": category,
                "Faithfulness": scores['faithfulness_score'],
                "Completeness": scores['completeness_score'],
                "Conciseness": scores['conciseness_score']
            }
            for category, scores in category_scores.items()
        ]
        df = pd.DataFrame(category_details)
        category_table_html = df.to_html(classes="table table-striped", index=False)

    # 生成雷达图
    radar_chart_html = ""
    if category_scores:
        categories = list(category_scores.keys())
        metrics = ["Faithfulness", "Completeness", "Conciseness"]

        # 准备雷达图数据
        radar_data = []
        for category in categories:
            scores = [
                category_scores[category]['faithfulness_score'],
                category_scores[category]['completeness_score'],
                category_scores[category]['conciseness_score']
            ]
            radar_data.append(go.Scatterpolar(
                r=scores,
                theta=metrics,
                fill='toself',
                name=category
            ))

        # 创建雷达图
        fig = go.Figure(data=radar_data)
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template="plotly_white",
            showlegend=True
        )
        radar_chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>LLM Score Report</title>
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
            <h1>LLM Score Report</h1>

            <h2>Overall Scores</h2>
            {overall_metrics_html}

            {f"<h2>Category Scores</h2><div class='table-container'>{category_table_html}</div>" if category_scores else ""}

            {f"<h2>Radar Chart of Scores by Category</h2><div class='chart-container'>{radar_chart_html}</div>" if category_scores else ""}
        </div>
    </body>
    </html>
    """

    return html_template
