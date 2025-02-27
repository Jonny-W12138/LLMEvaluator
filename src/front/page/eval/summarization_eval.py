import subprocess

import streamlit as st
from streamlit import columns, session_state
import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import config as conf
from src.eval.summarization.response_generate import generate_summaries_model, generate_summaries_api
from src.eval.summarization.generate_keyfact import generate_keyfact_model, generate_keyfact_api
from src.eval.summarization.llm_judge import llm_fact_checking_judge_model, llm_fact_checking_judge_api, \
llm_fact_alignment_judge_model, llm_fact_alignment_judge_api, parse_score
from src.eval.summarization.bert import bert_judge
from src.eval.summarization.bleurt_judge import bleurt_judge
from src.eval.summarization.rouge_judge import rouge_judge
import json
import pandas as pd
import time

st.header("Summarization Evaluate")

# 任务选择
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

# 模型选择
with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface/local", "API"])

    with model_path_col:
        if model_source != "API":
            st.write("Model path")
            model_path = st.text_input(
                "Path to pretrained model or model identifier from Hugging Face"
            )

    if model_source == "API":
        api_url = st.text_input("API URL")
        api_key = st.text_input("API key")
        model_engine = st.text_input("Model engine")

    original_model_adapter_path = None
    if model_source != "API":
        original_model_use_adapter = st.toggle("Use adapter", value=False)
        if original_model_use_adapter:
            original_model_adapter_path = st.text_input("Adapter path", value="", key="original_model_adapter_path")

# 数据集选择
with (st.container()):
    data_path = conf.get("base_config", "summarization", "data_path").strip('"')
    data_config_path = os.path.join(data_path, "data_config.json")
    if not os.path.isfile(data_config_path):
        st.error(f"""Please set data_config.json in "{data_path}".""")
    else:
        st.write("Dataset")
        try:
            with open(data_config_path, 'r') as dataset_file:
                datasets = json.load(dataset_file)
        except Exception as e:
            st.error(f"Failed to load dataset config: {e}. \nPlease check {data_config_path}")

        names_list = list(datasets.keys())
        selected_dataset = st.selectbox("Choose the dataset to evaluate", names_list)

        if st.button("Select & Preview"):
            try:
                dataset_path = os.path.join(os.getcwd(), datasets[selected_dataset]['data_path'])
                with open(dataset_path, 'r') as dataset_file:
                    dataset_head = json.load(dataset_file)[:5]

                # 将预览数据存储到 session_state 中
                st.session_state["dataset_preview"] = dataset_head
                st.session_state["selected_dataset"] = selected_dataset
            except Exception as e:
                st.error(f"Failed to preview dataset: {e}")

        # 如果有已加载的预览数据，显示它
        if "dataset_preview" in st.session_state:
            st.write(f"Preview of dataset: {st.session_state['selected_dataset']}")
            df = pd.DataFrame(st.session_state["dataset_preview"])
            st.dataframe(df)

        st.write("Field mapping")

        # 在 Streamlit 界面中显示带下拉框的编辑器
        field_mapping = st.data_editor(
            pd.DataFrame(
                {
                    "Dataset Field": [None],
                    "Instruction Placeholder": [None],
                    "Field Type": [None]  # 新增列，默认值为空
                }
            ),
            column_config={
                "Field Type": st.column_config.SelectboxColumn(
                    options=["", "Ref Summary", "Transcript"],
                    help="Select the type of the field."
                )
            },
            num_rows="dynamic",
            use_container_width=True,
        )

        st.session_state["field_mapping"] = field_mapping

        with st.expander("Field mapping examples"):
            st.markdown("- Dataset:\n\n"
                        "```json\n"
                        "[\n"
                        "    {\n"
                        "        \"text\": \"This is the text to summarize.\",\n"
                        "        \"summary\": \"This is the summary of the text.\"\n"
                        "    }\n"
                        "]\n"
                        "```"
                        "\n\n"
                        "- Instruction:\n\n"
                        "```plain\n"
                        "Please summarize the given text:\n"
                        "{{original_text}}\n"
                        "```\n\n"
                        "You should fill the table above like this:")


            st.markdown("Then the instruction will be like this:\n\n"
                        "```plain\n"
                        "Please summarize the given text:\n"
                        "This is the text to summarize.\n"
                        "```\n\n")

# 生成摘要
with st.container(border=True):
    st.write("Response generation")

    prompt_template_default = """\
Text:{{input_text}}
Instruction: Summarize the Text.
Provide your answer in JSON format. The answer should be a dictionary with the key "summary" containing a generated summary as a string:
{"summary": "your summary"}
JSON Output:"""
    prompt_template = st.text_area("Prompt template", value=prompt_template_default, height=200)

    st.markdown("- Output should be in `JSON format`\n\n - Output will be saved at `tasks` folder")

    response_generate_args_col = st.columns(3, vertical_alignment="bottom")
    with response_generate_args_col[0]:
        max_tokens = st.number_input("Max tokens", value=400, min_value=1)
    with response_generate_args_col[1]:
        temperature = st.number_input("Temperature",
                                      value=0.0, min_value=0.0, max_value=1.0, step=0.1)
    with response_generate_args_col[2]:
        top_p = st.number_input("Top p", value=1.0, min_value=0.0, max_value=1.0, step=0.1)


    if st.button("Generate"):
        if "field_mapping" not in st.session_state:
            st.error("Please set field mapping.")
        else:
            if 'selected_dataset' not in st.session_state:
                st.error("Please select a dataset first.")
            else:
                if model_source == "API":
                    generate_summaries_api(
                        selected_data_path=os.path.join(os.getcwd(), datasets[selected_dataset]['data_path']),
                        api_url=api_url,
                        api_key=api_key,
                        model_engine=model_engine,
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        task_name=st.session_state["task"],
                        prompt_template=prompt_template
                    )
                else:
                    generate_summaries_model(
                        selected_data_path=os.path.join(os.getcwd(), datasets[selected_dataset]['data_path']),
                        model_path=model_path,
                        adapter_path=original_model_adapter_path,
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        task_name=st.session_state["task"],
                        prompt_template=prompt_template
                    )

# 评估
with st.container(border=True):
    st.write("LLM Evaluation")

    llm_tab, bert_tab, bleurt_tab, rouge_tab = st.tabs(["LLM", "BertScore", "BLEURT", "ROUGE"])


    with llm_tab:
        st.write("Keyfact")
        if_generate_keyfact = st.toggle("Generate keyfact", value=False,
                                        help="Evaluation contains keyfact alignment.\n\n"
                                             "If the dataset does not contains keyfact, please enable this option.\n\n"
                                             "After generating, please add the keyfact datapath in data_config.json")
        if if_generate_keyfact:
            st.write("Choose the LLM model for keyfact generation")

            model_source_col, model_path_col = st.columns(2, vertical_alignment="bottom")
            with model_source_col:
                st.write("LLM model")
                keyfact_generate_llm_source = st.selectbox("Choose the source of the model",
                                        ["huggingface/local", "API"], key="keyfact_generate_llm_source")

            with model_path_col:
                if keyfact_generate_llm_source != "API":
                    st.write("Model path")
                    keyfact_generate_llm_model_path = st.text_input(
                        "Path to pretrained model or model identifier from Hugging Face", key="keyfact_generate_llm_model_path"
                    )

            generate_keyfact_use_adapter = st.toggle("Use adapter", value=False, key="generate_keyfact_use_adapter")
            if generate_keyfact_use_adapter:
                adapter_path = st.text_input("Adapter path", value="", key="generate_keyfact_adapter_path")

            if keyfact_generate_llm_source == "API":
                keyfact_generate_api_url = st.text_input("API URL", key="keyfact_generate_api_url")
                keyfact_generate_api_key = st.text_input("API key", key="keyfact_generate_api_key")
                keyfact_generate_model_engine = st.text_input("Model engine", key="keyfact_generate_model_engine")

            keyfact_generate_prompt_template = st.text_area(
                "Prompt template",
                value= \
"""You will be provided with a summary. Your task is to decompose \
the summary into a set of "key facts". A "key fact" is a single \
fact written as briefly and clearly as possible, encompassing at \
most 2-3 entities. 
Here are nine examples of key facts to illustrate the desired \
level of granularity:
* Kevin Carr set off on his journey from Haytor.
* Kevin Carr set off on his journey from Dartmoor.
* Kevin Carr set off on his journey in July 2013.
* Kevin Carr is less than 24 hours away from completing his trip.
* Kevin Carr ran around the world unsupported.
* Kevin Carr ran with his tent.
* Kevin Carr is set to break the previous record.
* Kevin Carr is set to break the record by 24 hours.
* The previous record was held by an Australian.
Instruction:
First, read the summary carefully. \
Second, decompose the summary into (at most 16) key facts. \
Provide your answer in JSON format. The answer should be a \
dictionary with the key"key facts" containing the key facts as a \
list:
{"key facts": ["first key fact", "second key fact", "third key fact"]}
Summary:
{{summary}}
""",
                height=200,
                key="keyfact_generate_prompt_template"
            )

            max_token_col, temperature_col, top_p_col = st.columns(3)
            with max_token_col:
                keyfact_generate_max_tokens = st.number_input("Max tokens", value=400, min_value=1, key="keyfact_generate_max_tokens")
            with temperature_col:
                keyfact_generate_temperature = st.number_input("Temperature", value=0.0, min_value=0.0, max_value=1.0, step=0.1, key="keyfact_generate_temperature")
            with top_p_col:
                keyfact_generate_top_p = st.number_input("Top p", value=1.0, min_value=0.0,
                                                         max_value=1.0, step=0.1, key="keyfact_generate_top_p")

            if st.button("Generate keyfact"):
                if keyfact_generate_llm_source != "API":
                    generate_keyfact_model_args = {
                        "model_path": keyfact_generate_llm_model_path,
                        "selected_dataset": st.session_state["selected_dataset"],
                        "task_name": st.session_state["task"],
                        "prompt_template": keyfact_generate_prompt_template,
                        "field_mapping": pd.DataFrame(st.session_state["field_mapping"]),
                        "max_tokens": keyfact_generate_max_tokens,
                        "temperature": keyfact_generate_temperature,
                        "top_p": keyfact_generate_top_p
                    }

                    if generate_keyfact_use_adapter and adapter_path.strip():
                        generate_keyfact_model_args["adapter_path"] = adapter_path

                    generate_keyfact_model(**generate_keyfact_model_args)
                else:
                    generate_keyfact_api(
                        task_name=st.session_state["task"],
                        selected_dataset=st.session_state["selected_dataset"],
                        prompt_template=keyfact_generate_prompt_template,
                        api_url=keyfact_generate_api_url,
                        api_key=keyfact_generate_api_key,
                        model_engine=keyfact_generate_model_engine,
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        max_tokens=keyfact_generate_max_tokens,
                        temperature=keyfact_generate_temperature,
                        top_p=keyfact_generate_top_p
                    )


        st.divider()

        st.write("Fact Checking")

        llm_settings_col1, llm_settings_col2 = st.columns(2)
        with llm_settings_col1:
            st.write("Judge model")
            judge_model_source = st.selectbox("Choose the source of the model",
                                     ["huggingface/local", "API"], key="llm_model_source")

        with llm_settings_col2:
            if judge_model_source != "API":
                st.write("model path")
                llm_model_path = st.text_input(
                    "Path to pretrained LLM model or model identifier from Hugging Face"
                )
        if judge_model_source == "API":
            api_url = st.text_input("API URL", key="judge_model_api_url")
            api_key = st.text_input("API key", key="judge_model_api_key")
            model_engine = st.text_input("Model engine", key="judge_model_engine")

        else:
            llm_judge_use_adapter = st.toggle("Use adapter", value=False, key="llm_judge_use_adapter")
            if llm_judge_use_adapter:
                adapter_path = st.text_input("Adapter path", value="")

        judge_template_default = """\
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.
Instruction:
First, compare each summary sentence with the transcript.
Second, find the exact quote which can confirm the factual consistency of the sentence and insert them. If you cannot, write "not mentioned".
Third, classify each sentence into one of the nine categories and then provide the classified category.
Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "quote", and "category":
[{"sentence": "first sentence", "quote": "identified quote", "category": "no error"}, {"sentence": "second sentence", "quote": "not mentioned", "category": "out-of-context error"}, {"sentence": "third sentence", "quote": "identified quote", "category": "entity error"}]
Transcript:
{{input_text}}
Summary:
{{summary_to_judge}}
"""
        judge_template = st.text_area("Judge template", value=judge_template_default, height=200)
        st.markdown("Please use: \n\n"
                    "- `{{summary_to_judge}}` to represent the summary to judge.\n\n"
                    "- `{{input_text}}` to represent the transcript.")

        if 'task' not in st.session_state or st.session_state["task"] is None:
            summary_files = []
            st.info("Please select a task first.")
        else:
            summary_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))
            summary_files = [file for file in summary_files if file.startswith("summaries_")]
            selected_summary_file = st.selectbox("Summary file", summary_files)

            selected_summary_file_path = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_summary_file)

            max_token_col, temperature_col, top_p_col = st.columns(3)
            with max_token_col:
                max_tokens = st.number_input("Max tokens", value=1200, min_value=1,key="llm_judge_max_tokens")
            with temperature_col:
                temperature = st.number_input("Temperature",
                                              value=0.0, min_value=0.0, max_value=1.0, step=0.1, key="llm_judge_temperature")
            with top_p_col:
                top_p = st.number_input("Top p", value=1.0, min_value=0.0, max_value=1.0, step=0.1, key="llm_judge_top_p")

            if st.button("Judge", key="llm_judge_fact_check_button"):
                if judge_model_source != "API":
                    kwargs = {
                        "model_path": llm_model_path,
                        "selected_dataset": st.session_state["selected_dataset"],
                        "selected_summary_file_path": selected_summary_file_path,
                        "task_name": st.session_state["task"],
                        "prompt_template": judge_template,
                        "field_mapping": pd.DataFrame(st.session_state["field_mapping"]),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }

                    if llm_judge_use_adapter and adapter_path.strip():
                        kwargs["adapter_path"] = adapter_path
                    llm_fact_checking_judge_model(**kwargs)
                else:
                    llm_fact_checking_judge_api(
                        task_name=st.session_state["task"],
                        selected_dataset=st.session_state["selected_dataset"],
                        prompt_template=judge_template,
                        selected_summary_file_path=selected_summary_file_path,
                        api_url=api_url,
                        api_key=api_key,
                        model_engine=model_engine,
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )


        st.divider()

        st.write("Fact Alignment")

        llm_settings_col1, llm_settings_col2 = st.columns(2)
        with llm_settings_col1:
            st.write("Judge model")
            judge_model_source = st.selectbox("Choose the source of the model",
                                     ["huggingface/local", "API"], key="llm_model_source_fact_alignment")

        with llm_settings_col2:
            if judge_model_source != "API":
                st.write("model path")
                llm_model_path = st.text_input(
                    "Path to pretrained LLM model or model identifier from Hugging Face",
                    key="llm_model_path_fact_alignment"
                )

        if judge_model_source == "API":
            api_url = st.text_input("API URL", key="judge_model_api_url_fact_alignment")
            api_key = st.text_input("API key", key="judge_model_api_key_fact_alignment")
            model_engine = st.text_input("Model engine", key="judge_model_engine_fact_alignment")

        else:
            llm_judge_use_adapter = st.toggle("Use adapter", value=False, key="llm_judge_use_adapter_fact_alignment")
            if llm_judge_use_adapter:
                adapter_path = st.text_input("Adapter path", value="", key="llm_judge_adapter_path_fact_alignment")

        judge_template_default = """\
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact. 

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}, {"key fact": "third key fact", "response": "Yes", "line number": [1, 2, 3]}]

Summary:
{{summary_to_judge}}

key facts:
{{ref_keyfacts}}
"""

        judge_template = st.text_area("Judge template", value=judge_template_default, height=200, key="llm_judge_template_fact_alignment")
        st.markdown("Please use: \n\n"
                    "- `{{summary_to_judge}}` to represent the summary to judge.\n\n"
                    "- `{{ref_keyfacts}}` to represent the reference key facts.\n\n"
                    "- Please ensure that the reference event is in the dataset folder.")

        if 'task' not in st.session_state or st.session_state["task"] is None:
            summary_files = []
            st.info("Please select a task first.")
        else:
            summary_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))

            summary_files = [file for file in summary_files if file.startswith("summaries_")]

            selected_summary_file = st.selectbox("Summary file", summary_files, key="llm_judge_selected_summary_file_fact_alignment")

            selected_summary_file_path = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_summary_file)

            max_token_col, temperature_col, top_p_col = st.columns(3)
            with max_token_col:
                max_tokens = st.number_input("Max tokens", value=400, min_value=1, key="llm_judge_max_tokens_fact_alignment")
            with temperature_col:
                temperature = st.number_input("Temperature",
                                              value=0.0, min_value=0.0, max_value=1.0, step=0.1, key="llm_judge_temperature_fact_alignment")
            with top_p_col:
                top_p = st.number_input("Top p", value=1.0, min_value=0.0, max_value=1.0, step=0.1, key="llm_judge_top_p_fact_alignment")

            if st.button("Judge", key="llm_judge_fact_alignment_button"):
                if judge_model_source != "API":
                    kwargs = {
                        "model_path": llm_model_path,
                        "selected_dataset": st.session_state["selected_dataset"],
                        "task_name": st.session_state["task"],
                        "prompt_template": judge_template,
                        "selected_summary_file_path": selected_summary_file_path,
                        "field_mapping": pd.DataFrame(st.session_state["field_mapping"]),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }

                    if llm_judge_use_adapter and adapter_path.strip():
                        kwargs["adapter_path"] = adapter_path
                    llm_fact_alignment_judge_model(**kwargs)
                else:
                    llm_fact_alignment_judge_api(
                        task_name=st.session_state["task"],
                        selected_dataset=st.session_state["selected_dataset"],
                        prompt_template=judge_template,
                        selected_summary_file_path=selected_summary_file_path,
                        api_url=api_url,
                        api_key=api_key,
                        model_engine=model_engine,
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

        st.divider()
        st.write("Parse and Score")
        if "task" not in st.session_state or st.session_state["task"] is None:
            st.info("Please select a task first.")

        else:
            keyfact_check_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))
            keyfact_check_files = [file for file in keyfact_check_files if file.startswith("keyfact_check")]
            selected_keyfact_check_file = st.selectbox("Keyfact check file", keyfact_check_files)

            keyfact_alignment_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))
            keyfact_alignment_files = [file for file in keyfact_alignment_files if file.startswith("keyfact_alignment")]
            selected_keyfact_alignment_file = st.selectbox("Keyfact alignment file", keyfact_alignment_files)

            if st.button("Parse and Score"):
                parse_score(
                    st.session_state["task"],
                    os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_keyfact_check_file),
                    os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_keyfact_alignment_file)
                )
    with bert_tab:
        use_custom_cli = st.toggle("Custom cli", value=False)
        if not use_custom_cli:
            model_col, language_col, num_layer_col = st.columns(3)
            with model_col:
                st.write("BertScore model")
                judge_model_source = st.text_input("Input the source of the model", key="bertscore_judge_model_source",
                                                   value="microsoft/deberta-v3-base")
                st.markdown(
                    "- See [here](https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI) for supported models")
            with language_col:
                st.write("Summary language")
                language = st.text_input("Choose the language of the summary", key="bertscore_language", value="en")
                st.markdown(
                    "- See [here](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)\
                    for supported languages")

            with num_layer_col:
                st.write("Number of layers[optional]")
                num_layers = st.text_input("Input the number of layers", key="bertscore_num_layers")


            if 'task' not in st.session_state or st.session_state["task"] is None:
                summary_files = []
                st.info("Please select a task first.")

            else:
                summary_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))

                summary_files = [file for file in summary_files if file.startswith("summaries_")]
                selected_summary_file = st.selectbox("Summary file", summary_files, key="selected_summary_file_bert")

                selected_summary_file_path = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_summary_file)

                if st.button("Evaluate"):
                    bert_judge(
                        bert_model=judge_model_source,
                        lang=language,
                        num_layers=num_layers,
                        summary_file_path=selected_summary_file_path,
                        task=st.session_state["task"],
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        selected_data_path=os.path.join(os.getcwd(), datasets[st.session_state["selected_dataset"]]['data_path'])
                    )

        else:
            custom_cli = st.text_area("Command", height=200, key="custom_cli")
            st.markdown("- See [here](https://pypi.org/project/bert-score) for more information on BertScore CLI.\n\n"
                        "- The result should be saved in the `tasks` folder.")
            if st.button("Run"):
                subprocess.run(custom_cli.split(" "))

    with bleurt_tab:
        use_custom_cli = st.toggle("Custom cli", value=False, key="use_custom_cli_bleu")
        if not use_custom_cli:
            if 'task' not in st.session_state or st.session_state["task"] is None:
                summary_files = []
                st.info("Please select a task first.")
            else:
                model_col, summary_files_col = st.columns(2)
                with model_col:
                    judge_model_source = st.text_input("BLEURT Checkpoint", key="bleurt_judge_model_source",
                                                       value="BLEURT-20")
                    st.markdown(
                        "- See [here](https://github.com/google-research/bleurt/blob/master/checkpoints.md) for supported checkpoints")

                with summary_files_col:
                    summary_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))

                    summary_files = [file for file in summary_files if file.startswith("summaries_")]
                    selected_summary_file = st.selectbox("Summary file", summary_files, key="selected_summary_file_bleu")

                    selected_summary_file_path = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_summary_file)

                if st.button("Evaluate", key="bleurt_judge_button"):
                    bleurt_judge(
                        bleurt_model=judge_model_source,
                        summary_file_path=selected_summary_file_path,
                        task=st.session_state["task"],
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        selected_data_path=os.path.join(os.getcwd(), datasets[st.session_state["selected_dataset"]]['data_path'])
                    )

    with rouge_tab:
        use_custom_cli = st.toggle("Custom cli", value=False, key="use_custom_cli_rouge")
        if not use_custom_cli:
            if 'task' not in st.session_state or st.session_state["task"] is None:
                summary_files = []
                st.info("Please select a task first.")
            else:
                summary_files = os.listdir(os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization"))

                summary_files = [file for file in summary_files if file.startswith("summaries_")]
                selected_summary_file = st.selectbox("Summary file", summary_files, key="selected_summary_file_rouge")

                selected_summary_file_path = os.path.join(os.getcwd(), "tasks", st.session_state["task"], "summarization", selected_summary_file)

                if st.button("Evaluate", key="rouge_judge_button"):
                    rouge_judge(
                        summary_file_path=selected_summary_file_path,
                        task=st.session_state["task"],
                        field_mapping=pd.DataFrame(st.session_state["field_mapping"]),
                        selected_data_path=os.path.join(os.getcwd(), datasets[st.session_state["selected_dataset"]]['data_path'])
                    )
