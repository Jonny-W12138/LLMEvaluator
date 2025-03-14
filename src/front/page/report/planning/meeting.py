import json
import pandas as pd
import streamlit as st
import plotly.express as px


def load_meeting_data(file_path):
    """
    加载会面安排结果数据
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def display_meeting_metrics(data):
    """
    展示会面安排的关键指标
    """
    total_tests = len(data["results"])
    successful_tests = sum(1 for item in data["results"] if item["success"])
    passed_tests = sum(1 for item in data["results"] if item["evaluation"]["passed"])
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # 显示关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Successful Tests", successful_tests)
    with col3:
        st.metric("Passed Tests", passed_tests)
    with col4:
        st.metric("Pass Rate", f"{pass_rate:.2f}%")

def display_meeting_charts(data):
    """
    展示会面安排的图表
    """
    # 准备数据
    chart_data = {
        "Total Tests": len(data["results"]),
        "Successful Tests": sum(1 for item in data["results"] if item["success"]),
        "Passed Tests": sum(1 for item in data["results"] if item["evaluation"]["passed"]),
    }
    df = pd.DataFrame(list(chart_data.items()), columns=["Metric", "Count"])

    st.write("### Test Results Distribution")
    st.bar_chart(df.set_index("Metric"))

def display_meeting_pie_chart(data):

    passed_count = sum(1 for item in data["results"] if item["evaluation"]["passed"])
    failed_count = len(data["results"]) - passed_count

    error_types = {}
    for item in data["results"]:
        if not item["evaluation"]["passed"]:
            for error in item["evaluation"]["errors"]:
                error_type = error["type"]
                if error_type in error_types:
                    error_types[error_type] += 1
                else:
                    error_types[error_type] = 1

    pie_data = {
        "Category": ["Passed", "Failed"],
        "Count": [passed_count, failed_count]
    }
    df_pie = pd.DataFrame(pie_data)

    fig = px.pie(df_pie, values="Count", names="Category", title="Passed vs Failed Tests")
    st.plotly_chart(fig)

    if error_types:
        error_data = {
            "Error Type": list(error_types.keys()),
            "Count": list(error_types.values())
        }
        df_errors = pd.DataFrame(error_data)
        fig_errors = px.pie(df_errors, values="Count", names="Error Type", title="Error Types Distribution")
        st.plotly_chart(fig_errors)

def display_meeting_details(data):

    st.write("### Detailed Test Results")

    details = []
    for index, item in enumerate(data["results"]):
        details.append({
            "Test ID": index + 1,
            "City": item["metadata"]["city"],
            "Num Places": item["metadata"]["num_places"],
            "Num Conflicts": item["metadata"]["num_conflicts"],
            "Success": item["success"],
            "Passed": item["evaluation"]["passed"],
            "Errors": ", ".join([error["type"] for error in item["evaluation"]["errors"]]) if not item["evaluation"][
                "passed"] else "None"
        })
    df = pd.DataFrame(details)

    st.dataframe(df)

    if "selected_meeting_test_id" not in st.session_state:
        st.session_state.selected_meeting_test_id = None

    test_id = st.text_input("Enter Test ID to view JSON:", "", key="meeting_test_id")

    if st.button("View JSON", key="meeting_view_json"):
        try:
            test_index = int(test_id) - 1  # 转换为索引
            if 0 <= test_index < len(data["results"]):
                st.session_state.selected_meeting_test_id = test_index  # 存储选择的 Test ID
            else:
                st.warning("Invalid Test ID. Please enter a valid ID.")
        except ValueError:
            st.warning("Please enter a valid numeric Test ID.")

    if st.button("Close Preview", key="meeting_close_preview"):
        st.session_state.selected_meeting_test_id = None  # 重置状态，隐藏 JSON

    if st.session_state.selected_meeting_test_id is not None:
        st.write(f"### Test {st.session_state.selected_meeting_test_id + 1} JSON Data")
        st.json(data["results"][st.session_state.selected_meeting_test_id])

def display_meeting_difficulty_metrics(data):

    difficulty_groups = {}
    for item in data["results"]:
        key = (item["metadata"]["num_places"], item["metadata"]["num_conflicts"])
        if key not in difficulty_groups:
            difficulty_groups[key] = []
        difficulty_groups[key].append(item)

    difficulty_data = []
    for key, group in difficulty_groups.items():
        num_places, num_conflicts = key
        total_tests = len(group)
        successful_tests = sum(1 for item in group if item["success"])
        passed_tests = sum(1 for item in group if item["evaluation"]["passed"])
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        difficulty_data.append({
            "Difficulty": f"{num_places} Places, {num_conflicts} Conflicts",
            "Total Tests": total_tests,
            "Successful Tests": successful_tests,
            "Passed Tests": passed_tests,
            "Pass Rate": pass_rate
        })
    df = pd.DataFrame(difficulty_data)

    st.write("### Test Results by Difficulty")
    cols = st.columns(2)
    with cols[0]:

        fig_bar = px.bar(df, x="Difficulty", y=["Total Tests", "Successful Tests", "Passed Tests"],
                         title="Test Results by Difficulty", barmode="group", text_auto=True)
        st.plotly_chart(fig_bar)

    with cols[1]:
        fig_pie = px.pie(df, values="Pass Rate", names="Difficulty", title="Pass Rate by Difficulty")
        st.plotly_chart(fig_pie)

def generate_meeting_html_report(data):
    """生成精美的 HTML 报告"""
    total_tests = len(data["results"])
    successful_tests = sum(1 for item in data["results"] if item["success"])
    passed_tests = sum(1 for item in data["results"] if item["evaluation"]["passed"])
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # 生成柱状图
    df_bar = pd.DataFrame({
        "Metric": ["Total Tests", "Successful Tests", "Passed Tests"],
        "Count": [total_tests, successful_tests, passed_tests]
    })
    fig_bar = px.bar(df_bar, x="Metric", y="Count", title="Test Results Distribution", text="Count", template="plotly_white")
    bar_chart_html = fig_bar.to_html(full_html=False)

    # 生成通过情况饼图
    df_pie = pd.DataFrame({
        "Category": ["Passed", "Failed"],
        "Count": [passed_tests, total_tests - passed_tests]
    })
    fig_pie = px.pie(df_pie, values="Count", names="Category", title="Passed vs Failed Tests", template="plotly_white")
    pie_chart_html = fig_pie.to_html(full_html=False)

    # 生成错误类型分布饼图
    error_types = {}
    for item in data["results"]:
        if not item["evaluation"]["passed"]:
            for error in item["evaluation"]["errors"]:
                error_type = error["type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
    error_chart_html = ""
    if error_types:
        df_errors = pd.DataFrame({
            "Error Type": list(error_types.keys()),
            "Count": list(error_types.values())
        })
        fig_errors = px.pie(df_errors, values="Count", names="Error Type", title="Error Types Distribution", template="plotly_white")
        error_chart_html = fig_errors.to_html(full_html=False)

    # 生成不同难度下的表现数据
    difficulty_groups = {}
    for item in data["results"]:
        key = (item["metadata"]["num_places"], item["metadata"]["num_conflicts"])
        if key not in difficulty_groups:
            difficulty_groups[key] = []
        difficulty_groups[key].append(item)

    difficulty_data = []
    for key, group in difficulty_groups.items():
        num_places, num_conflicts = key
        total_tests_group = len(group)
        successful_tests_group = sum(1 for item in group if item["success"])
        passed_tests_group = sum(1 for item in group if item["evaluation"]["passed"])
        pass_rate_group = (passed_tests_group / total_tests_group) * 100 if total_tests_group > 0 else 0
        difficulty_data.append({
            "Difficulty": f"{num_places} Places, {num_conflicts} Conflicts",
            "Total Tests": total_tests_group,
            "Successful Tests": successful_tests_group,
            "Passed Tests": passed_tests_group,
            "Pass Rate": pass_rate_group
        })
    df_difficulty = pd.DataFrame(difficulty_data)

    # 生成不同难度下的柱状图
    fig_difficulty_bar = px.bar(df_difficulty, x="Difficulty", y=["Total Tests", "Successful Tests", "Passed Tests"],
                                title="Test Results by Difficulty", barmode="group", text_auto=True, template="plotly_white")
    difficulty_bar_chart_html = fig_difficulty_bar.to_html(full_html=False)

    # 生成不同难度下的饼图
    fig_difficulty_pie = px.pie(df_difficulty, values="Pass Rate", names="Difficulty", title="Pass Rate by Difficulty", template="plotly_white")
    difficulty_pie_chart_html = fig_difficulty_pie.to_html(full_html=False)

    # 生成表格
    details = [
        {
            "Test ID": index + 1,
            "City": item["metadata"]["city"],
            "Num Places": item["metadata"]["num_places"],
            "Num Conflicts": item["metadata"]["num_conflicts"],
            "Success": item["success"],
            "Passed": item["evaluation"]["passed"],
            "Errors": ", ".join([error["type"] for error in item["evaluation"]["errors"]]) if not item["evaluation"]["passed"] else "None"
        }
        for index, item in enumerate(data["results"])
    ]
    df = pd.DataFrame(details)
    table_html = df.to_html(classes="table table-striped", index=False)

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Meeting Test Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f4f4f4;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
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
            <h1>Meeting Test Report</h1>

            <h2>Key Metrics</h2>
            <div class="metric">
                <div>Total Tests: {total_tests}</div>
                <div>Successful Tests: {successful_tests}</div>
                <div>Passed Tests: {passed_tests}</div>
                <div>Pass Rate: {pass_rate:.2f}%</div>
            </div>

            <h2>Test Results Distribution</h2>
            {bar_chart_html}

            <h2>Passed vs Failed Tests</h2>
            {pie_chart_html}

            {f"<h2>Error Types Distribution</h2>{error_chart_html}" if error_types else ""}

            <h2>Performance by Difficulty</h2>
            {difficulty_bar_chart_html}
            {difficulty_pie_chart_html}

            <h2>Detailed Test Results</h2>
            <div class="table-container">
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """

    # 保存为 HTML 文件
    with open("meeting_report.html", "w", encoding="utf-8") as file:
        file.write(html_template)

    return "meeting_report.html"

def render_meeting_tab(file_path):
    with st.container():
        st.write("### Meeting")
        data = load_meeting_data(file_path)

        display_meeting_metrics(data)

        display_meeting_charts(data)

        display_meeting_pie_chart(data)

        display_meeting_difficulty_metrics(data)

        display_meeting_details(data)

        if st.button("Generate Report", key="meeting_generate_report"):
            report_path = generate_meeting_html_report(data)
            st.success(f"Report generated successfully: {report_path}")