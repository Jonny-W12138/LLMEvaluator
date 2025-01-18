import streamlit as st
from streamlit import columns
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import config as conf
import json
import pandas as pd

st.header("Summarization Evaluate")

with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface", "API", "local"])

    with model_path_col:
        if model_source != "API":
            st.write("Model path")
            model_path = st.text_input(
                "Path to pretrained model or model identifier from Hugging Face"
            )

    if model_source == "API":
        api_url = st.text_input("API URL")
        api_key = st.text_input("API key")

with st.container(border=True):
    data_path = conf.get("summarization", "data_path").strip('"')
    data_config_path = os.path.join(data_path, "data_config.json")
    if not os.path.isfile(os.path.join(data_path, "data_config.json")):
        st.error(f"""Please set data_config.json in "{data_path}".""")
    else:
            st.write("Dataset")
            with open(data_config_path, 'r') as dataset_file:
                datasets = json.load(dataset_file)

            names_list = list(datasets.keys())

            # 显示下拉菜单
            selected_dataset = st.selectbox("Choose the dataset to evaluate", names_list)
            if st.button("Preview dataset"):
                # 加载选中的数据集文件
                with open(os.path.join(os.getcwd(), datasets[selected_dataset]['path']), 'r') as dataset_file:
                    dataset_head = json.load(dataset_file)[:5]

                df = pd.DataFrame(dataset_head)
                st.dataframe(df)

with st.container(border=True):
    st.write("Response generation")
    response_generate_prompt = st.text_area("Instruction", height=200, help="")
    st.markdown("- Use `{original text}` for the text to summarize. \n\n"
                "- Use`{ref_summary_01}` `{ref_summary_02}`... for reference summaries.\n\n"
                "- Output will be saved at runs folder.")
    with st.expander("Instruction examples"):
        st.write("Please summarize the given text:\n\n"
                 "{original text}")
    response_generate_args_col = st.columns(3)
    with response_generate_args_col[0]:
        max_tokens = st.number_input("Max tokens", value=400, min_value=1)
    with response_generate_args_col[1]:
        temperature = st.number_input("Temperature", value=0.0, min_value=0.0, max_value=1.0, step=0.1)
    with response_generate_args_col[2]:
        top_p = st.number_input("Top p", value=1.0, min_value=0.0, max_value=1.0, step=0.1)

    if st.button("Generate"):
        if response_generate_prompt is None or response_generate_prompt == "":
            st.error("Instruction cannot be empty.")
        else:
            pass

with st.container(border=True):
    st.write("LLM Evaluation")

    use_llm_judge = st.toggle("Use LLM model", value=True)
    if use_llm_judge:
        llm_settings_col1, llm_settings_col2 = st.columns(2)
        with llm_settings_col1:
            st.write("LLM model")
            judge_model_source = st.selectbox("Choose the source of the model",
                                     ["huggingface", "API", "local"], key="llm_model_source")

        with llm_settings_col2:
            if judge_model_source != "API":
                st.write("LLM model path")
                llm_model_path = st.text_input(
                    "Path to pretrained LLM model or model identifier from Hugging Face"
                )
        if judge_model_source == "API":
            api_url = st.text_input("API URL", key="judge_model_api_url")
            api_key = st.text_input("API key", key="judge_model_api_key")

        else:
            llm_judge_use_adapter = st.toggle("Use adapter", value=False)
            if llm_judge_use_adapter:
                adapter_path = st.text_input("Adapter path", value="")
