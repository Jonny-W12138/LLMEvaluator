import os
import streamlit as st
import json
import pandas as pd
import plotly.express as px

from src.front.page.report.planning.meeting import render_meeting_tab, generate_meeting_html_report
from src.front.page.report.planning.calender import render_calender_tab, generate_calender_html_report
from src.front.page.report.planning.trip import render_trip_tab, generate_trip_html_report


st.header("Planning Report")

with st.container(border=True):
    st.write("Task")
    task = st.selectbox(
        "Task name",
        [
            folder for folder in os.listdir(os.path.join(os.getcwd(), "tasks"))
            if os.path.isdir(os.path.join(os.getcwd(), "tasks", folder))
               and not folder.startswith('.')
        ]
    )

    if st.button("Select", key="select_task"):
        if task is not None or task != "":
            st.session_state["task"] = task

if "task" not in st.session_state:
    st.error("Please select a task first.")

else:
    with st.container(border=True):
        st.write("Result files")
        selected_files = {}

        if "task" in st.session_state:
            task_dir = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "planning")

            # 定义子文件夹名称
            sub_folders = ["calender", "meeting", "trip"]

            # 遍历每个子文件夹
            for sub_folder in sub_folders:
                sub_folder_path = os.path.join(task_dir, sub_folder, "evaluation")

                # 检查子文件夹是否存在
                if os.path.exists(sub_folder_path):
                    # 获取子文件夹中的文件
                    files = [f for f in os.listdir(sub_folder_path) if not f.startswith('.')]

                    if files:
                        # 显示子文件夹名称的复选框
                        is_selected = st.checkbox(sub_folder, value=True, key=f"checkbox_{sub_folder}")

                        if is_selected:
                            # 显示文件选择的下拉框
                            selected_file = st.selectbox(
                                f"Select a file from {sub_folder}",
                                files,
                                key=f"select_{sub_folder}",
                                label_visibility="visible"
                            )
                            selected_files[sub_folder] = os.path.join(sub_folder_path, selected_file)

            # 显示用户选择的文件
            if selected_files:
                st.session_state["selected_files"] = selected_files
                st.write("Selected files:")
                for folder, file_path in selected_files.items():
                    st.write(f"{folder}: {file_path}")

    with st.container(border=True):
        st.write("Result preview")

        if selected_files == {}:
            st.error("Please select a file first.")
            st.stop()

        # 创建选项卡
        sub_task_tabs = st.tabs(list(selected_files.keys()))

        # 将选项卡元组转换为字典
        sub_task_tabs_dict = dict(zip(selected_files.keys(), sub_task_tabs))

        # 根据选择的文件调用对应的渲染函数
        for sub_folder, file_path in selected_files.items():
            with sub_task_tabs_dict[sub_folder]:
                if sub_folder == "calender":
                    render_calender_tab(file_path)
                elif sub_folder == "meeting":
                    render_meeting_tab(file_path)
                elif sub_folder == "trip":
                    render_trip_tab(file_path)

    with st.container(border=True):
        def load_data(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
            return data

        def generate_combined_html_report(selected_files):
            """
            根据用户勾选的子任务生成整合的 HTML 报告
            """
            reports = {}
            report_paths = {}

            # 根据用户勾选的子任务生成对应的报告
            if "calender" in selected_files:
                calendar_data = load_data(selected_files["calender"])
                report_paths["calender"] = generate_calender_html_report(calendar_data)
            if "meeting" in selected_files:
                meeting_data = load_data(selected_files["meeting"])
                report_paths["meeting"] = generate_meeting_html_report(meeting_data)
            if "trip" in selected_files:
                trip_data = load_data(selected_files["trip"])
                report_paths["trip"] = generate_trip_html_report(trip_data)

            # 读取生成的报告内容
            for sub_folder, report_path in report_paths.items():
                with open(report_path, "r", encoding="utf-8") as file:
                    reports[sub_folder] = file.read()

            # 创建整合的 HTML 模板
            combined_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Combined Test Reports</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        background-color: #f4f4f4;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }
                    h1, h2, h3 {
                        text-align: center;
                        color: #333;
                    }
                    .report-section {
                        margin-bottom: 40px;
                    }
                    .report-section h2 {
                        background-color: #007BFF;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                    }
                </style>
            </head>
            <body>
                <div>
                    <h1>Planning Reports</h1>
            """

            # 动态添加用户勾选的子任务报告
            if "calender" in reports:
                combined_html += f"""
                    <div class="report-section">
                        {reports["calender"]}
                    </div>
                """
            if "meeting" in reports:
                combined_html += f"""
                    <div class="report-section">
                        {reports["meeting"]}
                    </div>
                """
            if "trip" in reports:
                combined_html += f"""
                    <div class="report-section">
                        {reports["trip"]}
                    </div>
                """

            combined_html += """
                </div>
            </body>
            </html>
            """

            # 保存整合的 HTML 文件
            combined_report_path = "combined_report.html"
            with open(combined_report_path, "w", encoding="utf-8") as file:
                file.write(combined_html)

            return combined_report_path

        if st.button("Generate Combined Test Report"):
            combined_report_html = generate_combined_html_report(selected_files)
            st.markdown(combined_report_html, unsafe_allow_html=True)