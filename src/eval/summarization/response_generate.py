import datetime
import os
import time
import json
import pandas as pd
import torch
from openai import OpenAI
from streamlit import success
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import streamlit as st
import openai
import re


# 模拟子进程逻辑（实际情况下替换为大模型调用逻辑）
def generate_summaries_model(task_name, selected_data_path, model_path, prompt_template,
                       field_mapping, max_tokens, temperature, top_p,adapter_path=None):
    """
    Generate summaries using a model from local path or huggingface.
    """

    print("[Generating summaries]", "data_path:", selected_data_path, "prompt:", prompt_template,
          "max_tokens:", max_tokens, "temperature:", temperature, "top_p:", top_p, "adapter_path:", adapter_path)

    if (field_mapping["Field Type"] == "Ref Summary").sum() != 1 \
        or (field_mapping['Field Type'] == 'Transcript').sum() != 1:
        st.error("Field mapping must contain one 'Ref Summary' and one 'Transcript' field.")
        return

    # 读取数据集
    dataset = pd.read_json(selected_data_path)
    results = []

    st.session_state["log"] = []
    total_data_num = dataset.shape[0]

    # 滚动区域
    log_area = st.empty()
    my_bar = st.progress(0)

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if adapter_path:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
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
            current_row = dataset.iloc[i]
            prompt = prompt_template
            # 替换提示模板中的占位符
            for _, row in field_mapping.iterrows():
                dataset_field = row["Dataset Field"]
                placeholder = row["Instruction Placeholder"]
                prompt = prompt.replace(f"{placeholder}", str(current_row[dataset_field]))

            messages = [
                {"role": "system", "content": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""},
                {"role": "user", "content": prompt}
            ]

            # 调用生成模型进行推理
            kwargs = {
                "messages": messages,
                "max_new_tokens": max_tokens,
                "top_p": top_p
            }

            if temperature == 0.0:
                kwargs["do_sample"] = False
            else:
                kwargs["temperature"] = temperature

            outputs = pipe(**kwargs)

            match = re.search(r'{"summary":\s*".*?"}', outputs["generated_text"][-1]["content"])
            if not match:
                raise ValueError("No match found for the JSON pattern.")

            if match:
                json_data = match.group()
                # 转换为字典并提取summary
                parsed_data = json.loads(json_data)
                summary = parsed_data.get("summary")
                summary = summary.replace(". ", ".\n ")

            if not summary:
                raise ValueError("No 'summary' key found in the JSON data.")

            # 构造成功的结果数据
            result_data = {
                "record_index": i,
                "success": True,
                "summary": summary
            }

            success_num += 1

        except Exception as e:
            print(f"Error generating summary for record {i}: {e}")
            result_data = {
                "record_index": i,
                "success": False,
                "summary": ""
            }

        results.append(result_data)

        # 更新进度条
        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")
        time.sleep(1)  # 模拟耗时


    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(os.getcwd(), "tasks", task_name, "summarization")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_file = os.path.join(save_dir, f"summaries_{current_time}.json")

    metadata = {
        "model_source": "Model",
        "total_data_num": total_data_num,
        "success_num": success_num,
        "failed_num": total_data_num - success_num,
        "selected_data_path": selected_data_path,
        "prompt_template": prompt_template,
        "field_mapping": field_mapping.to_dict(),  # 将field_mapping转换为字典
        "adapter_path": adapter_path,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    output_data = {
        "metadata": metadata,
        "results": results
    }


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    st.success(f"Summaries generated successfully! Output saved to {output_file}")

def generate_summaries_api(task_name, selected_data_path, prompt_template, api_url, api_key, model_engine,
                       field_mapping, max_tokens, temperature, top_p):
    """
    Generate summaries using the OpenAI API.
    """
    print("[Generating summaries]", "data_path:", selected_data_path, "prompt:", prompt_template,
            "max_tokens:", max_tokens, "temperature:", temperature, "top_p:", top_p)

    if (field_mapping["Field Type"] == "Ref Summary").sum() != 1 \
        or (field_mapping['Field Type'] == 'Transcript').sum() != 1:
        st.error("Field mapping must contain one 'Ref Summary' and one 'Transcript' field.")
        return

    client = OpenAI(api_key=api_key, base_url=api_url)

    # 读取数据集
    dataset = pd.read_json(selected_data_path)
    results = []

    st.session_state["log"] = []
    total_data_num = dataset.shape[0]

    # 滚动区域
    log_area = st.empty()
    my_bar = st.progress(0)

    success_num = 0

    for i in range(total_data_num):
        try:
            current_row = dataset.iloc[i]
            prompt = prompt_template
            # 替换提示模板中的占位符
            for _, row in field_mapping.iterrows():
                dataset_field = row["Dataset Field"]
                placeholder = row["Instruction Placeholder"]
                prompt = prompt.replace(f"{placeholder}", str(current_row[dataset_field]))

            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "system", "content": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                                    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
                stop=None,
            )

            # 提取生成的文本
            generated_text = response.choices[0].message.content.strip().replace("\n", "")

            match = re.search(r'{"summary":\s*".*?"}', generated_text)
            if not match:
                raise ValueError("No match found for the JSON pattern.")

            if match:
                json_data = match.group()
                # 转换为字典并提取summary
                parsed_data = json.loads(json_data)
                summary = parsed_data.get("summary")
                summary = summary.replace(". ", ".\n ")

            if not summary:
                raise ValueError("No 'summary' key found in the JSON data.")

            # 构造成功的结果数据
            result_data = {
                "record_index": i,
                "success": True,
                "summary": summary
            }

            success_num += 1

        except Exception as e:
            print(f"Error generating summary for record {i}: {e}")
            result_data = {
                "record_index": i,
                "success": False,
                "summary": ""
            }

        results.append(result_data)

        # 更新进度条
        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")
        time.sleep(1)  # 模拟耗时

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(os.getcwd(), "tasks", task_name, "summarization")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_file = os.path.join(save_dir, f"summaries_{current_time}.json")

    metadata = {
        "model_source": "API",
        "model_engine": model_engine,
        "api_url": api_url,
        "total_data_num": total_data_num,
        "success_num": success_num,
        "failed_num": total_data_num - success_num,
        "selected_data_path": selected_data_path,
        "prompt_template": prompt_template,
        "field_mapping": field_mapping.to_dict(),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    output_data = {
        "metadata": metadata,
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    st.success(f"Summaries generated successfully! Output saved to {output_file}")