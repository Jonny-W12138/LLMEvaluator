import streamlit as st
import pandas as pd
import torch
import peft
from openai import OpenAI
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json
import os
import time
import datetime
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import config as conf

def generate_keyfact_model(model_path, selected_dataset, task_name, prompt_template,
                         field_mapping, max_tokens=400, temperature=0.0, top_p=1, adapter_path=None):
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

    # Load model and tokenizer
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

    # Use pipeline
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
                {"role": "user", "content": prompt}
            ]

            # 调用生成模型进行推理
            kwargs = {
                "text_inputs": messages,
                "max_new_tokens": max_tokens,
                "top_p": top_p
            }

            if temperature == 0.0:
                kwargs["do_sample"] = False
            else:
                kwargs["temperature"] = temperature

            outputs = pipe(**kwargs)

            generated_text = outputs[0]["generated_text"][-1]["content"].strip().replace("\n", "")

            model_res = generated_text

            model_res = model_res.replace('```', '')
            start = model_res.find('{')
            end = model_res.rfind('}')

            if start != -1 and end != -1:
                key_fact = model_res[start:end + 1]
                key_facts_content = json.loads(key_fact)
                key_facts = key_facts_content['key facts']


            else:
                key_facts = []

            # 保存结果，并附加上对应的数据集记录序号
            result_data = {
                "record_index": i,
                "success": True,
                "model_output": generated_text,
                "key_facts": key_facts
            }

            success_num += 1



        except Exception as e:
            print(f"Error generating keyfacts for record {i}: {e}")
            result_data = {
                "record_index": i,
                "success": False,
                "model_output": generated_text,
                "key_facts": []
            }

        results.append(result_data)

        # 更新进度条
        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")
        time.sleep(1)  # 模拟耗时


    save_dir = os.path.join(datasets[selected_dataset]['root_path'], f"{selected_dataset}_keyfacts.json")

    metadata = {
        "model_source": "Model",
        "total_data_num": total_data_num,
        "success_num": success_num,
        "failed_num": total_data_num - success_num,
        "selected_dataset": selected_dataset,
        "prompt_template": prompt_template,
        "field_mapping": field_mapping.to_dict(),
        "adapter_path": adapter_path,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    output_data = {
        "metadata": metadata,
        "results": results
    }

    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    datasets[selected_dataset]['keyfacts_path'] = save_dir
    with open(data_config_path, 'w') as dataset_file:
        json.dump(datasets, dataset_file, indent=4)

    st.success(f"Keyfacts generated successfully! Output saved to {save_dir}")

def generate_keyfact_api(task_name, selected_dataset, prompt_template, api_url, api_key, model_engine,
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
        try:
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
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
                stop=None,
            )

            generated_text = response.choices[0].message.content.strip().replace("\n", "")

            model_res = generated_text
            model_res = model_res.replace('```', '')
            start = model_res.find('{')
            end = model_res.rfind('}')

            if start != -1 and end != -1:
                key_fact = model_res[start:end+1]
                key_facts_content = json.loads(key_fact)
                key_facts = key_facts_content['key facts']

            else:
                key_facts = []
                raise Exception("Key facts not found in generated text")

            # 保存结果，并附加上对应的数据集记录序号
            result_data = {
                "record_index": i,
                "success": True,
                "model_output": generated_text,
                "key_facts": key_facts
            }

            success_num += 1

        except Exception as e:
            print(f"Error generating keyfacts for record {i}: {e}")
            result_data = {
                "record_index": i,
                "success": False,
                "model_output": generated_text,
                "key_facts": []
            }

        results.append(result_data)

        # 更新进度条
        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")
        time.sleep(1)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(datasets[selected_dataset]['root_path'], f"{selected_dataset}_keyfacts.json")

    metadata = {
        "model_source": "API",
        "model_engine": model_engine,
        "api_url": api_url,
        "total_data_num": total_data_num,
        "success_num": success_num,
        "failed_num": total_data_num - success_num,
        "selected_dataset": selected_dataset,
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

    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    datasets[selected_dataset]['keyfacts_path'] = save_dir
    with open(data_config_path, 'w') as dataset_file:
        json.dump(datasets, dataset_file, indent=4)

    st.success(f"Keyfacts generated successfully! Output saved to {save_dir}")