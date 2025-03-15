import math
import streamlit as st
import os
import json
import sys
from htmlmin import minify

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
from src.report.summarization.report_generate \
    import init_template, generate_bertscore_report, generate_bleurt_report, generate_keyfact_alignment_report, generate_keyfact_check_report, generate_rouge_report, generate_summarization_report, generate_llm_score_report


from src.front.page.report.summarization.summaries import summaries_render, generate_summaries_html_report
from src.front.page.report.summarization.llm_score import llm_score_render, generate_llm_score_html_report
from src.front.page.report.summarization.keyfact_alignment import keyfact_alignment_render, generate_keyfact_alignment_html_report
from src.front.page.report.summarization.keyfact_check import keyfact_check_render, generate_keyfact_check_html_report
from src.front.page.report.summarization.rouge_results import rouge_results_render, generate_rouge_results_html_report
from src.front.page.report.summarization.bertscore_results import bertscore_results_render, generate_bertscore_results_html_report
from src.front.page.report.summarization.bleurt_results import bleurt_results_render, generate_bleurt_results_html_report

def generate_combined_report(selected_files):
    reports = {}

    # 评估角度与对应的报告生成函数
    report_generators = {
        "summaries": generate_summaries_html_report,
        "llm_score": generate_llm_score_html_report,
        "keyfact_alignment": generate_keyfact_alignment_html_report,
        "keyfact_check": generate_keyfact_check_html_report,
        "rouge_results": generate_rouge_results_html_report,
        "bertscore_results": generate_bertscore_results_html_report,
        "bleurt_results": generate_bleurt_results_html_report,
    }

    # 生成各个评估角度的报告
    for category, generator in report_generators.items():
        if category in selected_files:
            reports[category] = generator(selected_files[category])

    # 生成 HTML 页面
    combined_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Evaluation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                }
                .sidebar {
                    width: 250px;
                    background: #027bff;
                    color: white;
                    height: 100vh;
                    padding-top: 20px;
                    position: fixed;
                    overflow-y: auto;
                }
                .sidebar h2 {
                    text-align: center;
                    margin-bottom: 20px;
                }
                .sidebar ul {
                    list-style: none;
                    padding: 0;
                }
                .sidebar ul li {
                    padding: 10px;
                    text-align: center;
                }
                .sidebar ul li a {
                    color: white;
                    text-decoration: none;
                    display: block;
                    padding: 10px;
                    transition: 0.3s;
                }
                .sidebar ul li a:hover {
                    background: #34495e;
                }
                .content {
                    margin-left: 270px;
                    padding: 20px;
                    width: calc(100% - 270px);
                }
                .report-section {
                    display: none;
                }
                .active {
                    display: block;
                }
            </style>
            <script>
                function showReport(reportId) {
                    var sections = document.getElementsByClassName("report-section");
                    for (var i = 0; i < sections.length; i++) {
                        sections[i].classList.remove("active");
                    }
                    document.getElementById(reportId).classList.add("active");
                }
            </script>
        </head>
        <body>
            <div class="sidebar">
                <h2 style="color:white;">Summarization Reports</h2>
                <ul>
        """

    # 生成侧边栏目录
    for category in reports.keys():
        display_name = category.replace("_", " ").title()  # 先进行常规格式化
        display_name = display_name.replace("Llm", "LLM")  # 额外处理 LLM
        combined_html += f'<li><a href="#" onclick="showReport(\'{category}\')">{display_name}</a></li>'

    combined_html += """
                </ul>
            </div>
            <div class="content">
                <h1>Summarization Report</h1>
        """

    # 生成各个评估角度的内容，并默认显示第一个
    first = True
    for category, content in reports.items():
        active_class = "active" if first else ""
        combined_html += f'<div id="{category}" class="report-section {active_class}">{content}</div>'
        first = False

    combined_html += """
            </div>
        </body>
        </html>
        """

    return minify(combined_html)

render_functions = {
    "bertscore_results": bertscore_results_render,
    "bleurt_results": bleurt_results_render,
    "keyfact_alignment": keyfact_alignment_render,
    "keyfact_check": keyfact_check_render,
    "llm_score": llm_score_render,
    "rouge_results": rouge_results_render,
    "summaries": summaries_render,
}

st.header("Summarization Report")

with (st.container(border=True)):
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

if "task" in st.session_state:
    task_dir = os.path.join(os.getcwd(), "tasks", st.session_state["task"])

    # 定义不同前缀对应的目录
    prefix_dirs = {
        "summaries": os.path.join(task_dir, "summarization", "response"),
        "llm_score": os.path.join(task_dir, "summarization", "evaluation", "llm_judge", "llm_scores"),
        "keyfact_alignment": os.path.join(task_dir, "summarization", "evaluation", "llm_judge", "keyfact_alignment"),
        "keyfact_check": os.path.join(task_dir, "summarization", "evaluation", "llm_judge", "keyfact_check"),
        "bertscore_results": os.path.join(task_dir, "summarization", "evaluation", "bertscore_results"),
        "bleurt_results": os.path.join(task_dir, "summarization", "evaluation", "bleurt_results"),
        "rouge_results": os.path.join(task_dir, "summarization", "evaluation", "rouge_results"),
    }

    # 获取所有文件并分组
    file_groups = {prefix: [] for prefix in prefix_dirs}
    for prefix, dir_path in prefix_dirs.items():
        if os.path.exists(dir_path):
            all_files = os.listdir(dir_path)
            for file in all_files:
                if file.startswith(prefix):
                    file_groups[prefix].append(file)

    with st.container(border=True):
        st.write("Result files")
        selected_files = {}

        for prefix, files in file_groups.items():
            if files:
                is_selected = st.checkbox(prefix, value=True, key=f"checkbox_{prefix}")
                if is_selected:
                    selected_file = st.selectbox(
                        "",
                        files,
                        key=f"select_{prefix}",
                        label_visibility="collapsed"
                    )
                    selected_files[prefix] = os.path.join(prefix_dirs[prefix], selected_file)

        # 显示用户选择的文件
        if selected_files:
            st.session_state["selected_files"] = selected_files
            st.write(st.session_state["selected_files"])

    with st.container(border=True):
        st.write("Result preview")

        if "selected_files" in st.session_state and st.session_state["selected_files"]:
            tabs = st.tabs(list(st.session_state["selected_files"].keys()))  # 创建标签页

            for i, (prefix, file) in enumerate(st.session_state["selected_files"].items()):
                file_path = os.path.join(task_dir, file)

                with tabs[i]:  # 依次填充每个 tab
                    st.write(f"### {prefix}")

                    render_func = render_functions.get(prefix)
                    render_func(file_path)

                    # 记录分页信息
                    if f"page_{prefix}" not in st.session_state:
                        st.session_state[f"page_{prefix}"] = 1  # 当前页码

                    # 读取数据并存入 session_state
                    if prefix!='llm_score':
                        if st.button("Load", key=f"load_{prefix}"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            if "results" in data:
                                results = data["results"]
                            else:
                                results = data
                            st.session_state[f"results_{prefix}"] = results  # 存入 session_state

                    # 只有当数据已加载时才进行显示
                    if f"results_{prefix}" in st.session_state:
                        results = st.session_state[f"results_{prefix}"]

                        if isinstance(results, list):  # 只有列表数据才分页
                            total_pages = max(1, math.ceil(len(results) / 10))

                            if total_pages > 1:  # 仅在 total_pages > 1 时显示 slider
                                page = st.number_input(
                                    "Page number", min_value=1, max_value=total_pages,
                                    value=st.session_state[f"page_{prefix}"],
                                    help=f"Total pages: {total_pages}",
                                    key=f"page_input_{prefix}"
                                )
                                st.session_state[f"page_{prefix}"] = page  # 确保 slider 交互后数据不丢失
                            else:
                                page = 1  # 只有 1 页时，默认显示第一页
                                st.session_state[f"page_{prefix}"] = page  # 更新页码

                            # 计算起始和结束索引
                            start_idx = (st.session_state[f"page_{prefix}"] - 1) * 10
                            end_idx = start_idx + 10
                            paginated_results = results[start_idx:end_idx]

                            st.dataframe(paginated_results)
                        else:
                            st.dataframe(results)  # 非列表数据直接显示

    with st.container(border=True):
        st.write("Report generation")
        if st.download_button(
            label="Download combined report",
            data=generate_combined_report(st.session_state["selected_files"]),
            file_name="summarization_combined_report.html",
            mime="text/html"
        ):
            st.success("HTML report downloaded successfully.")




