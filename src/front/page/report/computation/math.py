import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

def display_math_category_metrics(data):
    """
    展示按类别的指标
    """
    # 按类别分组
    category_data = {}
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        if category not in category_data:
            category_data[category] = {"total": 0, "success": 0, "correct": 0}
        category_data[category]["total"] += 1
        if item["success"]:
            category_data[category]["success"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            category_data[category]["correct"] += 1

    # 显示类别指标
    st.write("### Metrics by Category")
    for category, metrics in category_data.items():
        st.write(f"**Category: {category}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", metrics["total"])
        with col2:
            st.metric("Success", metrics["success"])
        with col3:
            st.metric("Correct", metrics["correct"])

def display_math_difficulty_metrics(data):
    """
    展示按难度的指标
    """
    # 按难度分组
    difficulty_data = {}
    for item in data["evaluation"]:
        difficulty = item["origin_response"]["metadata"]["level"]
        if difficulty not in difficulty_data:
            difficulty_data[difficulty] = {"total": 0, "success": 0, "correct": 0}
        difficulty_data[difficulty]["total"] += 1
        if item["success"]:
            difficulty_data[difficulty]["success"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            difficulty_data[difficulty]["correct"] += 1

    # 显示难度指标
    st.write("### Metrics by Difficulty")
    for difficulty, metrics in difficulty_data.items():
        st.write(f"**Difficulty: {difficulty}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", metrics["total"])
        with col2:
            st.metric("Success", metrics["success"])
        with col3:
            st.metric("Correct", metrics["correct"])

def display_math_category_difficulty_chart(data):
    category_difficulty_data = []
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        difficulty = item["origin_response"]["metadata"]["level"]
        correct = 1 if item["evaluation"]["if_correct"] == "correct" else 0
        category_difficulty_data.append({
            "Category": category,
            "Difficulty": f"Level {difficulty}",
            "Correct": correct
        })

    # 创建数据框
    df = pd.DataFrame(category_difficulty_data)

    # 计算每个类别和难度的正确率
    df_correct_rate = df.groupby(["Category", "Difficulty"]).mean().reset_index()

    # 显示条形统计图
    st.write("### Correct Rate by Category and Difficulty")
    fig = px.bar(
        df_correct_rate,
        x="Category",
        y="Correct",
        color="Difficulty",
        template="plotly_dark",
        barmode="group",
        text_auto=True,
        labels={"Correct": "Correct Rate", "Category": "Category", "Difficulty": "Difficulty"}
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)

def display_math_radar_chart(data):
    """
    展示各个类别的正确率雷达图
    """
    # 按类别分组
    category_data = {}
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        if category not in category_data:
            category_data[category] = {"total": 0, "correct": 0}
        category_data[category]["total"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            category_data[category]["correct"] += 1

    # 计算正确率
    categories = list(category_data.keys())
    correct_rates = [(v["correct"] / v["total"]) if v["total"] > 0 else 0 for v in category_data.values()]

    # 创建雷达图
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=correct_rates,
        theta=categories,
        fill='toself',
        name="Correct Rate"
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

    # 显示雷达图
    st.write("### Correct Rate by Category (Radar Chart)")
    st.plotly_chart(fig_radar)

def display_math_details(data):
    """
    展示计算能力测评的详细结果，并提供输入框查询 JSON 详情
    """
    st.write("### Detailed Test Results")

    # 准备表格数据
    details = []
    for index, item in enumerate(data["evaluation"]):
        if item['success'] == False:
            continue
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
            "Difficulty": item["origin_response"]["metadata"]["level"],
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
    # 总体指标
    total_tests = len(data["evaluation"])
    successful_tests = sum(1 for item in data["evaluation"] if item["success"])
    correct_tests = sum(1 for item in data["evaluation"] if item["evaluation"]["if_correct"] == "correct")
    correct_rate = (correct_tests / total_tests) * 100 if total_tests > 0 else 0

    # 按类别分组
    category_data = {}
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        if category not in category_data:
            category_data[category] = {"total": 0, "success": 0, "correct": 0}
        category_data[category]["total"] += 1
        if item["success"]:
            category_data[category]["success"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            category_data[category]["correct"] += 1

    # 按难度分组
    difficulty_data = {}
    for item in data["evaluation"]:
        difficulty = item["origin_response"]["metadata"]["level"]
        if difficulty not in difficulty_data:
            difficulty_data[difficulty] = {"total": 0, "success": 0, "correct": 0}
        difficulty_data[difficulty]["total"] += 1
        if item["success"]:
            difficulty_data[difficulty]["success"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            difficulty_data[difficulty]["correct"] += 1

    # 按类别和难度分组
    category_difficulty_data = []
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        difficulty = item["origin_response"]["metadata"]["level"]
        correct = 1 if item["evaluation"]["if_correct"] == "correct" else 0
        category_difficulty_data.append({
            "Category": category,
            "Difficulty": f"Level {difficulty}",
            "Correct": correct
        })

    # 计算每个类别和难度的正确率
    df_category_difficulty = pd.DataFrame(category_difficulty_data)
    df_correct_rate = df_category_difficulty.groupby(["Category", "Difficulty"]).mean().reset_index()

    # 生成条形统计图
    fig_bar = px.bar(
        df_correct_rate,
        x="Category",
        y="Correct",
        color="Difficulty",
        barmode="group",
        text_auto=True,
        labels={"Correct": "Correct Rate", "Category": "Category", "Difficulty": "Difficulty"},
        title="Correct Rate by Category and Difficulty",
        template="plotly_white"
    )
    fig_bar.update_traces(textposition='outside')
    bar_chart_html = fig_bar.to_html(full_html=False, include_plotlyjs="cdn")

    # 生成雷达图
    category_rates = {}
    for item in data["evaluation"]:
        category = item["origin_response"]["metadata"]["type"]
        if category not in category_rates:
            category_rates[category] = {"total": 0, "correct": 0}
        category_rates[category]["total"] += 1
        if item["evaluation"]["if_correct"] == "correct":
            category_rates[category]["correct"] += 1

    categories = list(category_rates.keys())
    correct_rates = [(v["correct"] / v["total"]) if v["total"] > 0 else 0 for v in category_rates.values()]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=correct_rates,
        theta=categories,
        fill='toself',
        name="Correct Rate"
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
        showlegend=True,
        title="Correct Rate by Category (Radar Chart)"
    )
    radar_chart_html = fig_radar.to_html(full_html=False, include_plotlyjs="cdn")

    # 生成表格
    details = []
    for index, item in enumerate(data["evaluation"]):
        if item['success'] == False:
            continue
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
            "Difficulty": item["origin_response"]["metadata"]["level"],
            "Success": item["success"],
            "Correct": item["evaluation"]["if_correct"] == "correct",
            "Positive Steps": positive_count,
            "Neutral Steps": neutral_count,
            "Negative Steps": negative_count,
            "Negative Details": negative_details
        })
    df = pd.DataFrame(details)
    table_html = df.to_html(classes="table table-striped", index=False)

    # 生成按类别和难度的指标表格
    def generate_metrics_table(metrics, title):
        df = pd.DataFrame(metrics).T.reset_index()
        df.columns = ["Category/Difficulty", "Total Tests", "Success", "Correct"]
        html = f"""
        <h3>{title}</h3>
        {df.to_html(classes="table table-striped", index=False)}
        """
        return html

    category_metrics_html = generate_metrics_table(category_data, "Metrics by Category")
    difficulty_metrics_html = generate_metrics_table(difficulty_data, "Metrics by Difficulty")

    # HTML 模板
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Math Test Report</title>
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

            {category_metrics_html}
            {difficulty_metrics_html}

            <h2>Correct Rate by Category and Difficulty</h2>
            {bar_chart_html}

            <h2>Correct Rate by Category (Radar Chart)</h2>
            {radar_chart_html}

            <h2>Detailed Test Results</h2>
            <div class="table-container">
                {table_html}
            </div>
        </div>
    </body>
    </html>
    """

    return html_template


def render_math_tab(file_path):
    with st.container():
        st.write("### Math")
        data = load_math_data(file_path)

        # 展示关键指标
        display_math_metrics(data)

        # 展示按类别和难度的指标
        display_math_category_metrics(data)
        display_math_difficulty_metrics(data)

        # 展示各个类别的各个难度的正确率条形统计图
        display_math_category_difficulty_chart(data)

        # 展示雷达图
        display_math_radar_chart(data)

        # 展示详细结果
        display_math_details(data)

        if st.download_button(
            label="Download HTML Report",
            data=generate_math_html_report(data),
            file_name="math_report.html",
            mime="text/html"
        ):
            st.success("HTML report downloaded successfully!")