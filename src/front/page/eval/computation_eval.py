import streamlit as st
import sys
import os
import configparser

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import json
import pandas as pd

st.header("Computation Evaluation")

with st.container(border=True):
    st.write("Task")
    task_type = st.segmented_control("task type", ["New task", "Existing task"], default="New task", label_visibility="collapsed")
    task = None

    if task_type == "New task":
        input_task = st.text_input("Task name")
        if st.button("Create"):
            if input_task is None or input_task == "":
                st.error("Task name cannot be empty.")
            else:
                if os.path.exists(os.path.join(os.getcwd(), "tasks", input_task)):
                    st.error(f"Task {input_task} already exists. See {os.path.join(os.getcwd(), 'tasks', input_task)}")
                else:
                    os.mkdir(os.path.join(os.getcwd(), "tasks", input_task))
                    st.success(f"Task {input_task} created.")
                    task = input_task
                    if task is not None or task != "":
                        st.session_state["task"] = task

    else:
        task = st.selectbox("Task name", os.listdir(os.path.join(os.getcwd(), "tasks")))
        if st.button("Select", key="select_task"):
            if task is not None or task != "":
                st.session_state["task"] = task

with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface/local", "API"])

    if model_source == "huggingface/local":
        with model_path_col:
            st.write("Model path")
            model_path = st.text_input(
                "Path to pretrained model or model identifier from Hugging Face"
            )
        original_model_adapter_path = None
        original_model_use_adapter = st.toggle("Use adapter", value=False)
        if original_model_use_adapter:
            original_model_adapter_path = st.text_input("Adapter path", value="",
                                                        key="original_model_adapter_path")

    if model_source == "API":
        api_url = st.text_input("API URL")
        api_key = st.text_input("API key")
        model_engine = st.text_input("Model engine")

with st.container(border=True):
    st.write("GSM8K benchmark")

    if 'task' not in st.session_state:
        st.error("Please select a task first.")

    elif not os.path.exists(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json")):
        st.error("Please make sure the data_config.json is in the dataset/planning folder.")

    else:

        st.write("Generate eval data")
        if_generate_new_gsm8k = st.toggle("Generate new GSM8K eval data", value=False)
        if if_generate_new_gsm8k:
            types = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
                     "Number Theory", "Prealgebra", "Precalculus"]

            # 选择类型
            selected_types = st.pills(
                "Select types",
                types,
                default=types,
                selection_mode="multi"
            )

            # 生成数据集
            generate_config = pd.DataFrame(
                [
                    {"type": t, "level": str(i), "num_problems": 5}
                    for t in types for i in range(1, 6)
                ]
            )

            # 过滤已选择的 types
            filtered_config = generate_config[generate_config["type"].isin(selected_types)]

            # 按 type 分组
            grouped = filtered_config.groupby("type")

            # 计算列数
            num_columns = 3
            grouped_types = list(grouped.groups.keys())
            num_rows = (len(grouped_types) + num_columns - 1) // num_columns  # 计算行数（向上取整）

            # 按行列布局显示
            for i in range(num_rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    index = i * num_columns + j
                    if index < len(grouped_types):
                        type_name = grouped_types[index]
                        with cols[j]:
                            st.write(f"### {type_name}")
                            st.data_editor(grouped.get_group(type_name), key=f"editor_{type_name}")

            if st.button("Generate"):
                pass
