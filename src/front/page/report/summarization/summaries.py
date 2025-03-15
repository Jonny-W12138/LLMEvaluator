import json
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go


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

    if "category" in success_data.columns:  # 检查是否存在 category 字段
        # 计算每个类别的摘要长度
        success_data["summary_length"] = success_data["summary"].apply(len)

        # 创建小提琴图
        violin_fig = px.violin(
            success_data,
            x="category",  # x 轴为类别
            y="summary_length",  # y 轴为摘要长度
            box=True,  # 显示箱线图
            points="all",  # 显示所有数据点
            title="Summary Length Distribution by Category",  # 标题
            labels={"category": "Category", "summary_length": "Summary Length"},  # 标签
        )
        st.plotly_chart(violin_fig)
    else:
        st.warning("No 'category' field found in the data. Cannot create violin plot.")

    st.download_button(
        label="Download HTML Report",
        data=generate_summaries_html_report(file_path),
        file_name="summaries_report.html",
        mime="text/html"
    )

def generate_summaries_html_report(file_path):
    """生成摘要数据的 HTML 报告"""
    # 读取数据
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries_data = pd.DataFrame(data["results"])

    if summaries_data.shape[0] == 0:
        return "<h2>No data to display for summaries.</h2>"

    # 计算关键指标
    total_tests = summaries_data.shape[0]
    successful_tests = summaries_data["success"].sum()
    success_ratio = successful_tests / total_tests if total_tests > 0 else 0

    success_data = summaries_data[summaries_data["success"]]
    summary_lengths = success_data["summary"].apply(len)
    avg_summary_length = summary_lengths.mean()
    max_summary_length = summary_lengths.max()
    min_summary_length = summary_lengths.min()

    # 生成摘要长度分布图
    fig_hist = ff.create_distplot([summary_lengths], ["Summary Length"], show_hist=False)
    hist_chart_html = fig_hist.to_html(full_html=False, include_plotlyjs="cdn")

    # 生成类别摘要长度小提琴图（如果存在类别字段）
    violin_chart_html = ""
    if "category" in success_data.columns:
        success_data["summary_length"] = success_data["summary"].apply(len)
        fig_violin = px.violin(
            success_data,
            x="category",
            y="summary_length",
            color="category",
            template="plotly_white",
            box=True,
            points="all",
            title="Summary Length Distribution by Category",
            labels={"category": "Category", "summary_length": "Summary Length"},
        )
        violin_chart_html = fig_violin.to_html(full_html=False, include_plotlyjs="cdn")

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Summaries Report</title>
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
                padding: 20px;
                margin: 0 auto;
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
            <h1>Summaries Report</h1>

            <h2>Key Metrics</h2>
            <div class="metric">
                <div class="metric-row">
                    <div class="metric-item">Total Tests: {total_tests}</div>
                    <div class="metric-item">Successful Tests: {successful_tests}</div>
                    <div class="metric-item">Success Ratio: {success_ratio:.2%}</div>
                </div>
                <div class="metric-row">
                    <div class="metric-item">Avg Summary Length: {avg_summary_length:.2f}</div>
                    <div class="metric-item">Max Summary Length: {max_summary_length}</div>
                    <div class="metric-item">Min Summary Length: {min_summary_length}</div>
                </div>
            </div>

            <h2>Summary Length Distribution</h2>
            <div class="chart-container">
                {hist_chart_html}
            </div>

            {f"<h2>Summary Length by Category</h2><div class='chart-container'>{violin_chart_html}</div>" if violin_chart_html else ""}
        </div>
    </body>
    </html>
    """

    return html_template