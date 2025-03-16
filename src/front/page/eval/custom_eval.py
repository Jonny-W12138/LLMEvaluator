import importlib
import sys

import streamlit as st
import os
import pandas as pd
from src.eval.custom.custom import generate_custom_response

st.header("Custom Evaluation")

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

    if "task" not in st.session_state:
        st.warning("Choose a task to continue.")
        st.stop()

    if "task" in st.session_state:
        st.markdown(f"- Selected task: `{st.session_state['task']}`")


with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface/local", "API", "custom"])

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

    elif model_source == "API":
        api_url = st.text_input("API URL")
        api_key = st.text_input("API key")
        model_engine = st.text_input("Model engine")

    else:
        st.write("Please set up custom model response in `Response generate` below.")

if model_source != 'custom':
    with st.container(border=True):
        st.write("Ability")

        ability_type = st.segmented_control("ability type", ["New ability", "Existing ability"], default="New ability", label_visibility="collapsed")

        if ability_type == "New ability":
            input_ability = st.text_input("Ability name")
            if st.button("Create", key="create_ability"):
                if input_ability is None or input_ability == "":
                    st.error("Ability name cannot be empty.")
                else:
                    if os.path.exists(os.path.join(os.getcwd(), "src", "eval", "custom", input_ability)):
                        st.error(f"Ability {input_ability} already exists. See {os.path.join(os.getcwd(), "src", "eval", "custom", input_ability)}")
                    else:
                        os.mkdir(os.path.join(os.getcwd(), "src", "eval", "custom", input_ability))
                        st.success(f"Ability {input_ability} created.")
                        ability = input_ability
                        if ability is not None or ability != "":
                            st.session_state["ability"] = ability

        else:
            ability = st.selectbox("Ability name", [
                f for f in os.listdir(os.path.join(os.getcwd(), "src", "eval", "custom"))
                    if os.path.isdir(os.path.join(os.getcwd(), "src", "eval", "custom", f))
            ])

            if st.button("Select", key="select_ability"):
                if ability is not None or ability != "":
                    st.session_state["ability"] = ability

        if "ability" in st.session_state:
            st.markdown(f"- Selected ability: `{st.session_state["ability"]}`")
        else:
            st.markdown("- No ability selected.")

if "ability" not in st.session_state:
    st.warning("Choose an ability to continue.")
    st.stop()

if model_source != 'custom':
    with st.container(border=True):
        st.write("Dataset")

        with st.expander("Dataset format", expanded=False):
            st.markdown("- Dataset file should be in JSON format.\n"
                        "- `metadata` field is optional.\n"
                        "- `problems` field is required.\n"
                        "  - `category` field is suggested in problem for fine-grained evaluation.\n")
            st.json("\n"
                    "{\n"
                    "  \"metadata\": {\n"
                    "    \"field_1\": \"...\",\n"
                    "    \"field_2\": \"...\"\n"
                    "  },\n"
                    "  \"problems\": [\n"
                    "    {\n"
                    "      \"field_1_of_problem_1\": \"...\",\n"
                    "      \"field_2_of_problem_1\": \"...\"\n"
                    "    },\n"
                    "    {\n"
                    "      \"field_1_of_problem_2\": \"...\",\n"
                    "      \"field_2_of_problem_2\": \"...\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "")

        dataset_source = st.segmented_control("dataset type", ["Upload dataset", "Existing dataset"], default="Upload dataset", label_visibility="collapsed")
        if dataset_source == "Upload dataset":
            uploaded_file = st.file_uploader("Choose a file", type="json")

            if uploaded_file is not None:
                file_name = uploaded_file.name
                os.makedirs(os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"]), exist_ok=True)
                save_path = os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"], file_name)

                with open(save_path, mode="wb") as f:
                    f.write(uploaded_file.getvalue())

                st.success(f"Dataset uploaded. See {save_path}.")

        elif dataset_source == "Existing dataset":
            if not os.path.exists(os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"])):
                st.error(f"Dataset path {os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"])} does not exist.")
                st.stop()

            selected_dataset = st.selectbox("Select dataset", os.listdir(os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"])))
            if st.button("Select", key="select_dataset"):
                if selected_dataset is not None or selected_dataset != "":
                    st.session_state["custom_dataset_path"] = os.path.join(os.getcwd(), "dataset", "custom", st.session_state["ability"], selected_dataset)

with st.container(border=True):
    st.write("Response generate")
    if "custom_dataset_path" not in st.session_state:
        st.warning("Choose a dataset to continue.")
        st.stop()

    if model_source == "custom":

        with st.expander("Custom model", expanded=False):
            st.markdown(
                "- Custom model should be a Python module with a function that takes a prompt and returns a response.\n"
                f"- The response should be in JSON format.\n"
                f"- The response file should be saved in the folder "
                f"`{os.path.join(os.getcwd(), 'tesks', st.session_state['task'], 'custom', st.session_state['ability'])}`\n"
                f"- The response file should contain `response` field.")
            st.json("{\n"
                    "  \"response\": [\n"
                    "    {\n"
                    "      \"response_field_1\": \"...\",\n"
                    "      \"response_field_2\": \"...\"\n"
                    "    }\n"
                    "  ]\n"
                    "}")

        script_path = st.text_input("Path to module(abs path)")
        function_name = st.text_input("Function name")

        if st.button("Generate", key="generate_custom_response"):
            script_path = os.path.abspath(script_path)

            # 获取模块名称（去掉.py）
            module_name = os.path.splitext(os.path.basename(script_path))[0]

            # 加载模块
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None:
                raise ImportError(f"Could not load module from {script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 获取函数
            if hasattr(module, function_name):
                function = getattr(module, function_name)
            else:
                raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'")

            with st.spinner("Executing ..."):
                function()
            st.success("Successfully generated.")

    else:

        dataset_field_mapping_df = pd.DataFrame(
            [
                {"dataset_field": "dataset_field_1", "prompt_field": "prompt_field_1"},
            ]
        )
        dataset_field_mapping = st.data_editor(
            dataset_field_mapping_df, num_rows="dynamic"
        )

        for i, row in dataset_field_mapping.iterrows():
            for col in row.index:
                if pd.isna(row[col]):
                    st.error(f"Column '{col}' cannot be empty.")
                    break

        prompt_template = st.text_area("Prompt template",
                                       value="This is a prompt template. \n"
                                             "{{prompt_field_1}} is a placeholder for the value of dataset_field_1.\n"
                                             "Please instruct the LLM to reply in JSON format:\n"
                                             "{\n"
                                             "  \"response_field_1\": \"...\",\n"
                                             "  \"response_field_2\": \"...\"\n"
                                             "}",
                                       height=200)

        generate_args_col = st.columns(3)
        response_generate_args_col = st.columns(3)
        with response_generate_args_col[0]:
            max_tokens = st.number_input("Max tokens", value=600, min_value=1)
        with response_generate_args_col[1]:
            temperature = st.number_input("Temperature", value=0.7, min_value=0.0, step=0.1)
        with response_generate_args_col[2]:
            top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1)

        if st.button("Generate", key="generate_trip_response"):
            if model_source == "API":
                generate_custom_response(
                    dataset_path=st.session_state["custom_dataset_path"],
                    call_method=model_source,
                    field_mapping=dataset_field_mapping,
                    api_key=api_key,
                    api_url=api_url,
                    model_engine=model_engine,
                    task_name=st.session_state['task'],
                    prompt_template=prompt_template,
                    max_new_token=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            elif model_source == "huggingface/local":
                generate_custom_response(
                    dataset_path=st.session_state["custom_dataset_path"],
                    call_method=model_source,
                    field_mapping=dataset_field_mapping,
                    model_name=model_path,
                    model_adapter=original_model_adapter_path if original_model_use_adapter else None,
                    task_name=st.session_state['task'],
                    prompt_template=prompt_template,
                    max_new_token=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

with st.container(border=True):
    st.write("Evaluation")

    evaluate_field_mapping_df = pd.DataFrame(
        [
            {"response_field": "response_field_1", "prompt_field": "prompt_field_1"},
        ]
    )

    evaluate_field_mapping = st.data_editor(
        evaluate_field_mapping_df, num_rows="dynamic"
    )

    for i, row in evaluate_field_mapping.iterrows():
        for col in row.index:
            if pd.isna(row[col]):
                st.error(f"Column '{col}' cannot be empty.")
                break


    tabs = st.tabs(["LLM", "custom"])


    with tabs[0]:
        st.write("LLM evaluation")

        judge_prompt_template = st.text_area(
            "Judge prompt",
            "This is a judge prompt template. \n"
            "Please instruct the LLM to reply in JSON format:\n"
            "{\n"
            "  \"response_field_1\": \"...\",\n"
            "  \"response_field_2\": \"...\"\n"
            "}",
            height=200,
        )

        st.write("Judge model")
        model_source_col, model_path_col = st.columns(2)
        with model_source_col:
            st.write("Model source")
            model_source = st.selectbox("Choose the source of the model",
                                        ["huggingface/local", "API"], key="custom_eval_model_source")

        if model_source == "huggingface/local":
            with model_path_col:
                st.write("Model path")
                model_path = st.text_input(
                    "Path to pretrained model or model identifier from Hugging Face",
                    key="custom_eval_model_path"
                )
            original_model_adapter_path = None
            original_model_use_adapter = st.toggle("Use adapter", value=False, key="custom_eval_use_adapter")
            if original_model_use_adapter:
                original_model_adapter_path = st.text_input("Adapter path", value="",
                                                            key="original_model_adapter_path")

        if model_source == "API":
            api_url = st.text_input("API URL", key="custom_eval_api_url")
            api_key = st.text_input("API key", key="custom_eval_api_key")
            model_engine = st.text_input("Model engine", key="custom_eval_model_engine")

        eval_generate_col = st.columns(3)
        with eval_generate_col[0]:
            max_tokens = st.number_input("Max tokens", value=2000, min_value=1, key="custom_eval_max_tokens")
        with eval_generate_col[1]:
            temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1,
                                          key="custom_eval_temperature")
        with eval_generate_col[2]:
            top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1, key="custom_eval_top_p")

    with tabs[1]:
        st.write("Custom evaluation")
        st.write("Coming soon...")
