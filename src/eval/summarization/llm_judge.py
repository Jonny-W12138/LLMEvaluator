import streamlit as st
import pandas as pd
import torch
import peft
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json
import os
import time
import datetime
import sys
import ast
from pyexpat.errors import messages

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import config as conf

ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']

def llm_fact_checking_judge_model(model_path, selected_dataset, task_name, prompt_template,
                    field_mapping, max_tokens=400, temperature=0.0, top_p=1, adapter_path=None):
    """
    Generate key facts using LLM model.
    """
    results = []

    data_path = conf.get("base_config", "summarization", "data_path").strip('"')
    data_config_path = os.path.join(data_path, "data_config.json")
    with open(data_config_path, 'r') as dataset_file:
        datasets = json.load(dataset_file)
    summary_dataset = pd.read_json(os.path.join(os.getcwd(), datasets[selected_dataset]['data_path']))
    total_data_num = summary_dataset.shape[0]

    my_bar = st.progress(0)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if adapter_path:
        peft_config = peft.PeftConfig.from_pretrained(adapter_path)
        model = peft.PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    success_num = 0

    for i in range(total_data_num):
        try:
            current_row = summary_dataset.iloc[i]
            prompt = prompt_template

            for _, row in field_mapping.iterrows():
                dataset_field = row["Dataset Field"]
                placeholder = row["Instruction Placeholder"]
                prompt = prompt.replace(f"{placeholder}", str(current_row[dataset_field]))

            messages = [
                {"role": "system", "content": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""},
                {"role": "user", "content": prompt}
            ]

            kwargs = {
                "messages": messages,
                "max_new_tokens": max_tokens,
                "top_p": top_p
            }

            if temperature == 0.0:
                kwargs["do_sample"] = False
            else:
                kwargs["temperature"] = temperature

            output = pipe(**kwargs)

            start_idx = output.find('[')

            if start_idx != -1:
                end_idx = output.find(']')
                output = output[start_idx:end_idx + 1]
                output = output.replace('\n', '')
                output = ast.literal_eval(output)

                pred_labels, pred_types = [], []
                for out in output:
                    category = out["category"]
                    category = category.replace('\n', '').replace('[', '').replace(']', '')
                    if category.lower() == "no error":
                        pred_labels.append(0)
                    else:
                        pred_labels.append(1)
                    pred_types.append(category)
                return pred_labels, pred_types

            else:
                start_idx = output.find('{')
                end_idx = output.find('}')
                output = output[start_idx:end_idx + 1]
                output = output.replace('\n', '')
                output = ast.literal_eval(output)

                pred_labels, pred_types = [], []
                category = output["category"]
                category = category.replace('\n', '').replace('[', '').replace(']', '')
                if category.lower() == "no error":
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
                pred_types.append(category)
                return pred_labels, pred_types

        except Exception as e:

            try:
                subseqs = output.split("category")

                def error_detection(subseq):
                    detected = False
                    for error_type in ERROR_TYPES:
                        if error_type in subseq:
                            detected = True
                            detected_type = error_type
                    if detected:
                        return 1, error_type
                    else:
                        return 0, "no error"

                pred_labels, pred_types = [], []
                for subseq in subseqs:
                    error_label, error_type = error_detection(subseq)
                    pred_labels.append(error_label)
                    pred_types.append(error_type)

                return pred_labels, pred_types

            except Exception as e:
                print('parsing error:', e)
                return [], []


def llm_fact_checking_judge_api(task_name, selected_dataset, prompt_template, api_url, api_key, model_engine,
                       field_mapping, max_tokens, temperature, top_p):
    """
    Generate key facts using LLM model.
    """
    results = []

    data_path = conf.get("base_config", "summarization", "data_path").strip('"')
    data_config_path = os.path.join(data_path, "data_config.json")
    with open(data_config_path, 'r') as dataset_file:
        datasets = json.load(dataset_file)
    dataset = pd.read_json(os.path.join(os.getcwd(), datasets[selected_dataset]['data_path']))

    total_data_num = dataset.shape[0]

    my_bar = st.progress(0)

    client = OpenAI(api_key=api_key, base_url=api_url)

    success_num = 0

    for i in range(total_data_num):
        current_row = dataset.iloc[i]
        prompt = prompt_template
        # 替换提示模板中的占位符
        for _, row in field_mapping.iterrows():
            dataset_field = row["Dataset Field"]
            placeholder = row["Instruction Placeholder"]
            prompt = prompt.replace(f"{placeholder}", str(current_row[dataset_field]))

        # 调用生成模型进行推理
        response = client.chat.completions.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
            stop=None,
        )

        generated_text = response.choices[0].message.content.strip().replace("\n", "")
        print(f"index {i}: ", generated_text)
