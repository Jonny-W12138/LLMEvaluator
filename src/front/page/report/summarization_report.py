import math
import streamlit as st
import os
import json
import sys

from nltk.metrics.aline import similarity_matrix

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
from src.report.summarization.report_generate \
    import init_template, generate_bertscore_report, generate_bleurt_report, generate_keyfact_alignment_report, generate_keyfact_check_report, generate_rouge_report, generate_summarization_report, generate_llm_score_report
from src.report.summarization.results_render import bertscore_results_render, bleurt_results_render, keyfact_alignment_render, keyfact_check_render, llm_score_render, rouge_results_render, summaries_render

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
    task_dir = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization")
    all_files = os.listdir(task_dir)


    prefixes = [
        "summaries",
        "llm_score",
        "keyfact_alignment",
        "keyfact_check",
        "bertscore_results",
        "bleurt_results",
        "rouge_results",
    ]

    file_groups = {prefix: [] for prefix in prefixes}
    for file in all_files:
        for prefix in prefixes:
            if file.startswith(prefix):
                file_groups[prefix].append(file)
                break

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
                        selected_files[prefix] = selected_file

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
        if st.button("Generate report"):
            report_template = init_template(st.session_state["task"])
            similarity_judged = False
            for prefix, file in st.session_state["selected_files"].items():
                if prefix == "bertscore_results":
                    if not similarity_judged:
                        report_template +='<h2>Similarity Based Judgement</h2>'
                        similarity_judged = True
                    report_template = generate_bertscore_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

                elif prefix == "bleurt_results":
                    if not similarity_judged:
                        report_template +='<h2>Similarity Based Judgement</h2>'
                        similarity_judged = True
                    report_template = generate_bleurt_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

                elif prefix == "keyfact_alignment":
                    report_template = generate_keyfact_alignment_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

                elif prefix == "keyfact_check":
                    report_template = generate_keyfact_check_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

                elif prefix == "rouge_results":
                    if not similarity_judged:
                        report_template +='<h2>Similarity Based Judgement</h2>'
                        similarity_judged = True
                    report_template = generate_rouge_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

                elif prefix == "summaries":
                    report_template = generate_summarization_report(
                        os.path.join(task_dir, file),
                        report_template
                    )
                elif prefix == "llm_score":
                    report_template = generate_llm_score_report(
                        os.path.join(task_dir, file),
                        report_template
                    )

            report_template += '''
</div>
</body>
</html>
            '''

            st.components.v1.html(report_template, height=800, scrolling=True)
            st.download_button(
                label="Download Report",
                data=report_template,
                file_name="report_template.html",
                mime="text/html"
            )