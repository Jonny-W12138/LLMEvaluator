import streamlit as st
import sys
import os
import json
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
from src.eval.computation.computation import generate_random_problem, generate_math_response, evaluate_math_response
import pandas as pd
import configparser
config = configparser.ConfigParser()
config.read('src/front/page/eval/eval_config.ini')

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
    st.write("MATH benchmark")

    if 'task' not in st.session_state:
        st.error("Please select a task first.")

    elif not os.path.exists(os.path.join(os.getcwd(), "dataset", "computation", "data_config.json")):
        st.error("Please make sure the data_config.json is in the dataset/planning folder.")

    else:

        st.write("Generate eval data")
        if_generate_new_math = st.toggle("Generate new math eval data", value=False)
        if if_generate_new_math:
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

            updated_data = {}

            # 按行列布局显示
            for i in range(num_rows):
                cols = st.columns(num_columns)
                for j in range(num_columns):
                    index = i * num_columns + j
                    if index < len(grouped_types):
                        type_name = grouped_types[index]
                        with cols[j]:
                            st.write(f"### {type_name}")
                            edited_df = st.data_editor(grouped.get_group(type_name), key=f"editor_{type_name}")
                            updated_data[type_name] = edited_df  # 存储修改后的 DataFrame

            # 合并所有编辑后的数据
            if updated_data:
                updated_config = pd.concat(updated_data.values(), ignore_index=True)

            if st.button("Generate"):
                generate_random_problem(updated_config.to_dict(orient="records"))

        st.divider()

        st.write("Response Generate")

        math_prompt_template = config.get("computation", "math_prompt_template")
        prompt = st.text_area(
            "Prompt",
            math_prompt_template.strip().replace("\\n", "\n"),
            height=400,
        )

        with open(os.path.join(os.getcwd(), "dataset", "computation", "data_config.json"), 'r', encoding='utf-8') as f:
            data_config = json.load(f)
        data_config = data_config['math']
        datafolder_path = data_config['data_folder']

        selected_dataset = st.selectbox("Dataset", [f for f in os.listdir(datafolder_path) if
                                                    not f.startswith('.')])
        dataset_path = os.path.join(datafolder_path, selected_dataset)

        if st.button("Preview dataset", key="preview_math_dataset"):
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
                metadata = dataset["metadata"]
            st.dataframe(pd.DataFrame(metadata))

        response_generate_args_col = st.columns(3)
        with response_generate_args_col[0]:
            max_tokens = st.number_input("Max tokens", value=1200, min_value=1, key="math_max_tokens")
        with response_generate_args_col[1]:
            temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1, key="math_temperature")
        with response_generate_args_col[2]:
            top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1, key="math_top_p")
        if st.button("Generate", key="generate_math_response"):
            if 'task' not in st.session_state or st.session_state['task'] is None:
                st.error("Please select a task first.")
            else:
                if model_source == "API":
                    generate_math_response(
                        dataset_path=dataset_path,
                        call_method=model_source,
                        api_key=api_key,
                        api_url=api_url,
                        model_engine=model_engine,
                        task_name=st.session_state['task'],
                        prompt_template=prompt,
                        max_new_token=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                else:
                    generate_math_response(
                        dataset_path=dataset_path,
                        call_method=model_source,
                        model_name=model_path,
                        model_adapter=original_model_adapter_path if original_model_use_adapter else None,
                        task_name=st.session_state['task'],
                        prompt_template=prompt,
                        max_new_token=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

        st.divider()

        st.write("Response Evaluation")

        eval_prompt_template = config.get("computation", "math_eval_prompt_template")
        eval_prompt = st.text_area("Prompt", eval_prompt_template.strip().replace("\\n", "\n"), height=400)

        if not os.path.exists(os.path.join(os.getcwd(), "tasks", st.session_state['task'], "computation", "math", "response")):
            st.error("Please make sure the response data is generated.")
            st.stop()

        response_path = st.selectbox("Select response file", [f for f in os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state['task'], "computation", "math", "response")) if f.endswith(".json")])
        if st.button("Preview response", key="preview_math_response"):
            with open(os.path.join(os.getcwd(), "tasks", st.session_state['task'], "computation", "math", "response", response_path), "r") as f:
                response = json.load(f)
            st.json(response['metadata'])

        st.write("Judge model")
        model_source_col, model_path_col = st.columns(2)
        with model_source_col:
            st.write("Model source")
            model_source = st.selectbox("Choose the source of the model",
                                        ["huggingface/local", "API"], key="math_eval_model_source")

        if model_source == "huggingface/local":
            with model_path_col:
                st.write("Model path")
                model_path = st.text_input(
                    "Path to pretrained model or model identifier from Hugging Face",
                    key="math_eval_model_path"
                )
            original_model_adapter_path = None
            original_model_use_adapter = st.toggle("Use adapter", value=False, key="math_eval_use_adapter")
            if original_model_use_adapter:
                original_model_adapter_path = st.text_input("Adapter path", value="",
                                                            key="original_model_adapter_path")

        if model_source == "API":
            api_url = st.text_input("API URL", key="math_eval_api_url")
            api_key = st.text_input("API key", key="math_eval_api_key")
            model_engine = st.text_input("Model engine", key="math_eval_model_engine")


        eval_generate_col = st.columns(3)
        with eval_generate_col[0]:
            max_tokens = st.number_input("Max tokens", value=2000, min_value=1, key="math_eval_max_tokens")
        with eval_generate_col[1]:
            temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1, key="math_eval_temperature")
        with eval_generate_col[2]:
            top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1, key="math_eval_top_p")

        if st.button("Evaluate", key="evaluate_math_response"):
            if model_source == "API":
                evaluate_math_response(
                    response_path=os.path.join(os.getcwd(), "tasks", st.session_state['task'], "computation", "math", "response", response_path),
                    task_name=st.session_state['task'],
                    call_method=model_source,
                    api_key=api_key,
                    api_url=api_url,
                    model_engine=model_engine,
                    prompt_template=eval_prompt,
                    max_new_token=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                evaluate_math_response(
                    response_path=os.path.join(os.getcwd(), "tasks", st.session_state['task'], "computation", "math", "response", response_path),
                    task_name=st.session_state['task'],
                    call_method=model_source,
                    model_name=model_path,
                    model_adapter=original_model_adapter_path if original_model_use_adapter else None,
                    prompt_template=eval_prompt,
                    max_new_token=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
