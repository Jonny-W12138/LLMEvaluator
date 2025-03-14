import streamlit as st
import os
from src.front.page.report.computation.math import render_math_tab
st.header("Computation Report")

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
            task_dir = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "computation", "math", "evaluation")

            files = [f for f in os.listdir(task_dir) if not f.startswith('.')]

            if files:
                selected_file = st.selectbox(
                    "Select a file from math",
                    files,
                    key=f"select_math",
                    label_visibility="visible"
                )
                selected_file_path = os.path.join(task_dir, selected_file)
                st.session_state["selected_math_eval_files"] = selected_file_path

    with st.container(border=True):
        render_math_tab(st.session_state['selected_math_eval_files'])





