import json
import pandas as pd
import streamlit as st
import plotly.express as px

def load_math_data(file_path):
    """
    加载计算能力测评结果数据
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def display_math_metrics(data):
    """
    展示计算能力测评的关键指标
    """
    total_tests = len(data["evaluation"])
    successful_tests = sum(1 for item in data["evaluation"] if item["success"])
    correct_tests = sum(1 for item in data["evaluation"] if item["evaluation"]["if_correct"] == "correct")
    correct_rate = (correct_tests / total_tests) * 100 if total_tests > 0 else 0

    # 显示关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Successful Tests", successful_tests)
    with col3:
        st.metric("Correct Tests", correct_tests)
    with col4:
        st.metric("Correct Rate", f"{correct_rate:.2f}%")


def display_math_charts(data):
    """
    展示计算能力测评的图表
    """
    # 准备数据
    chart_data = {
        "Total Tests": len(data["evaluation"]),
        "Successful Tests": sum(1 for item in data["evaluation"] if item["success"]),
        "Correct Tests": sum(1 for item in data["evaluation"] if item["evaluation"]["if_correct"] == "correct"),
    }
    df = pd.DataFrame(list(chart_data.items()), columns=["Metric", "Count"])

    # 显示柱状图
    st.write("### Test Results Distribution")
    st.bar_chart(df.set_index("Metric"))


def display_math_pie_chart(data):
    """
    展示通过情况和错误情况的占比饼图
    """
    # 准备数据
    correct_count = sum(1 for item in data["evaluation"] if item["evaluation"]["if_correct"] == "correct")
    incorrect_count = len(data["evaluation"]) - correct_count

    error_types = {}
    for item in data["evaluation"]:
        if item["evaluation"]["if_correct"] == "incorrect":
            for step in item["evaluation"]["eval_steps"].values():
                if step["categorie"] == "Negative":
                    sub_category = step["sub-categories"]
                    if sub_category in error_types:
                        error_types[sub_category] += 1
                    else:
                        error_types[sub_category] = 1

    # 创建饼图数据
    pie_data = {
        "Category": ["Correct", "Incorrect"],
        "Count": [correct_count, incorrect_count]
    }
    df_pie = pd.DataFrame(pie_data)

    # 创建饼图
    fig = px.pie(df_pie, values="Count", names="Category", title="Correct vs Incorrect Tests")
    st.plotly_chart(fig)

    # 如果有错误类型，显示错误类型的饼图
    if error_types:
        error_data = {
            "Error Type": list(error_types.keys()),
            "Count": list(error_types.values())
        }
        df_errors = pd.DataFrame(error_data)
        fig_errors = px.pie(df_errors, values="Count", names="Error Type", title="Error Types Distribution")
        st.plotly_chart(fig_errors)


def display_math_details(data):
    """
    展示计算能力测评的详细结果，并提供输入框查询 JSON 详情
    """
    st.write("### Detailed Test Results")

    # 准备表格数据
    details = []
    for index, item in enumerate(data["evaluation"]):
        eval_steps = item["evaluation"]["eval_steps"]
        positive_count = sum(1 for step in eval_steps.values() if step["categorie"] == "Positive")
        neutral_count = sum(1 for step in eval_steps.values() if step["categorie"] == "Neutral")
        negative_count = sum(1 for step in eval_steps.values() if step["categorie"] == "Negative")
        negative_types = {}
        for step in eval_steps.values():
            if step["categorie"] == "Negative":
                sub_category = step["sub-categories"]
                if sub_category in negative_types:
                    negative_types[sub_category] += 1
                else:
                    negative_types[sub_category] = 1
        negative_details = ", ".join([f"{k}: {v}" for k, v in negative_types.items()]) if negative_types else "None"

        details.append({
            "Test ID": index + 1,
            "Type": item["origin_response"]["metadata"]["type"],
            "Success": item["success"],
            "Correct": item["evaluation"]["if_correct"] == "correct",
            "Positive Steps": positive_count,
            "Neutral Steps": neutral_count,
            "Negative Steps": negative_count,
            "Negative Details": negative_details
        })
    df = pd.DataFrame(details)

    # 显示表格
    st.dataframe(df)

    # 初始化 session_state
    if "selected_test_id" not in st.session_state:
        st.session_state.selected_test_id = None

    # 输入框和按钮
    test_id = st.text_input("Enter Test ID to view JSON:", "")

    if st.button("View JSON"):
        try:
            test_index = int(test_id) - 1  # 转换为索引
            if 0 <= test_index < len(data["evaluation"]):
                st.session_state.selected_test_id = test_index  # 存储选择的 Test ID
            else:
                st.warning("Invalid Test ID. Please enter a valid ID.")
        except ValueError:
            st.warning("Please enter a valid numeric Test ID.")

    if st.button("Close Preview"):
        st.session_state.selected_test_id = None  # 重置状态，隐藏 JSON

    if st.session_state.selected_test_id is not None:
        st.write(f"### Test {st.session_state.selected_test_id + 1} JSON Data")
        st.json(data["evaluation"][st.session_state.selected_test_id])


def generate_math_html_report(data):
    """生成精美的 HTML 报告"""
    total_tests = len(data["evaluation"])
    successful_tests = sum(1 for item in data["evaluation"] if item["success"])
    correct_tests = sum(1 for item in data["evaluation"] if item["evaluation"]["if_correct"] == "correct")
    correct_rate = (correct_tests / total_tests) * 100 if total_tests > 0 else 0

    # 生成柱状图
    df_bar = pd.DataFrame({
        "Metric": ["Total Tests", "Successful Tests", "Correct Tests"],
        "Count": [total_tests, successful_tests, correct_tests]
    })
    fig_bar = px.bar(df_bar, x="Metric", y="Count", title="Test Results Distribution", text="Count",
                     template="plotly_white")
    bar_chart_html = fig_bar.to_html(full_html=False)

    # 生成通过情况饼图
    df_pie = pd.DataFrame({
        "Category": ["Correct", "Incorrect"],
        "Count": [correct_tests, total_tests - correct_tests]
    })
    fig_pie = px.pie(df_pie, values="Count", names="Category", title="Correct vs Incorrect Tests",
                     template="plotly_white")
    pie_chart_html = fig_pie.to_html(full_html=False)

    # 生成错误类型分布饼图
    error_types = {}
    for item in data["evaluation"]:
        if item["evaluation"]["if_correct"] == "incorrect":
            for step in item["evaluation"]["eval_steps"].values():
                if step["categorie"] == "Negative":
                    sub_category = step["sub-categories"]
                    error_types[sub_category] = error_types.get(sub_category, 0) + 1
    error_chart_html = ""
    if error_types:
        df_errors = pd.DataFrame({
            "Error Type": list(error_types.keys()),
            "Count": list(error_types.values())
        })
        fig_errors = px.pie(df_errors, values="Count", names="Error Type", title="Error Types Distribution",
                            template="plotly_white")
        error_chart_html = fig_errors.to_html(full_html=False)

    # 生成表格
    details = [
        {
            "Test ID": index + 1,
            "Type": item["origin_response"]["metadata"]["type"],
            "Success": item["success"],
            "Correct": item["evaluation"]["if_correct"] == "correct",
            "Positive Steps": sum(
                1 for step in item["evaluation"]["eval_steps"].values() if step["categorie"] == "Positive"
            ),
            "Neutral Steps": sum(
                1 for step in item["evaluation"]["eval_steps"].values() if step["categorie"] == "Neutral"
            ),
            "Negative Steps": sum(
                1 for step in item["evaluation"]["eval_steps"].values() if step["categorie"] == "Negative"
            ),
            "Negative Details": ", ".join(
                [
                    f"{k}: {v}"
                    for k, v in {
                    step["sub-categories"]: sum(
                        1
                        for s in item["evaluation"]["eval_steps"].values()
                        if s.get("sub-categories") == step["sub-categories"]
                    )
                    for step in item["evaluation"]["eval_steps"].values()
                    if step["categorie"] == "Negative" and "sub-categories" in step
                }.items()
                ]
            )
            if any(
                step["categorie"] == "Negative" and "sub-categories" in step
                for step in item["evaluation"]["eval_steps"].values()
            )
            else "None"
        }
        for index, item in enumerate(data["evaluation"])
    ]
    df = pd.DataFrame(details)
    table_html = df.to_html(classes="table table-striped", index=False)

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Math Test Report</title>
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
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }}
            h1, h2, h3 {{
                text-align: center;
                color: #333;
            }}
            .metric {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .metric div {{
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Math Test Report</h1>

            <h2>Key Metrics</h2>
            <div class="metric">
                <div>Total Tests: {total_tests}</div>
                <div>Successful Tests: {successful_tests}</div>
                <div>Correct Tests: {correct_tests}</div>
                <div>Correct Rate: {correct_rate:.2f}%</div>
            </div>

            <h2>Test Results Distribution</h2>
            {bar_chart_html}

            <h2>Correct vs Incorrect Tests</h2>
            {pie_chart_html}

            {f"<h2>Error Types Distribution</h2>{error_chart_html}" if error_types else ""}

            <h2>Detailed Test Results</h2>
            <div class="table-container">
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """

    # 保存为 HTML 文件
    with open("math_report.html", "w", encoding="utf-8") as file:
        file.write(html_template)

    return "math_report.html"


def render_math_tab(file_path):
    """
    渲染 Math 任务的选项卡内容
    """
    with st.container():
        st.write("### Math")
        data = load_math_data(file_path)

        # 展示关键指标
        display_math_metrics(data)

        # 展示图表
        display_math_charts(data)

        # 展示饼图
        display_math_pie_chart(data)

        # 展示详细结果
        display_math_details(data)

        if st.button("Generate Report"):
            report_path = generate_math_html_report(data)
            st.success(f"Report generated successfully: {report_path}")