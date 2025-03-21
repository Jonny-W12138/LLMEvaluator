import streamlit as st
import os
import json
from openai import OpenAI
from datetime import datetime
from src.eval.utils import init_model_pipe, get_model_response, get_api_response

def generate_custom_response(dataset_path, task_name, ability_name, prompt_template, field_mapping, max_new_token, temperature, top_p,
                              call_method=None, model_name=None, model_adapter=None,
                              api_key=None, api_url=None, model_engine=None):
    if call_method is None:
        raise ValueError("call_method must be provided.")


    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data is not None, "Data is empty."
    assert "problems" in data, "Data must contain 'problem' field."

    problems = data["problems"]

    total_data_num = len(problems)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "custom", ability_name, "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"{ability_name}_response_{current_time}.json")

    if call_method == "huggingface/local":
        if not model_name:
            raise ValueError("model_name must be provided for 'huggingface/local' call method.")
        pipe = init_model_pipe(model_name, model_adapter)

        data_metadata = {
            "dataset": dataset_path,
            "model_name": model_name,
            "model_adapter": model_adapter,
            "total_data_num": total_data_num,
            "prompt_template": prompt_template,
            "max_new_token": max_new_token,
            "temperature": temperature,
            "top_p": top_p
        }

    elif call_method == "API":
        if not (api_key and api_url and model_engine):
            raise ValueError("api_key, api_url, and model_engine must be provided for 'API' call method.")
        client = OpenAI(api_key=api_key, base_url=api_url)

        data_metadata = {
            "dataset": dataset_path,
            "api_url": api_url,
            "model_engine": model_engine,
            "total_data_num": total_data_num,
            "prompt_template": prompt_template,
            "max_new_token": max_new_token,
            "temperature": temperature,
            "top_p": top_p
        }

    else:
        raise ValueError(f"Invalid call_method: {call_method}. Must be 'huggingface/local' or 'API'.")

    mybar = st.progress(0)

    responses = []

    for i, problem in enumerate(problems):

        prompt = prompt_template
        for _, row in field_mapping.iterrows():
            dataset_field = row['dataset_field']
            prompt_field = row['prompt_field']
            prompt = prompt.replace(prompt_field, problem[dataset_field])

        llm_output = ""
        origin_output = ""

        try:
            if call_method == "huggingface/local":
                llm_output = get_model_response(pipe, prompt, max_new_token, temperature, top_p)
            elif call_method == "API":
                llm_output = get_api_response(client, prompt, model_engine, max_new_token, temperature, top_p)

            origin_output = llm_output
            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("Invalid JSON format")
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output.replace("\\", "\\\\"))

            response = {
                "success": True,
                "problem": problem,
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }

        except Exception as e:
            response = {
                "success": False,
                "problem": problem,
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": llm_output,
            }
            print(e)

        responses.append(response)
        mybar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

        with open(save_file, "w", encoding="utf-8") as file:
            json.dump({
                "metadata": data_metadata,
                "response": responses
            }, file, indent=4, ensure_ascii=False)

    st.success(f"{ability_name} response generated successfully! See {save_file}.")

def evaluate_custom_response(dataset_path, response_path, task_name, ability_name, prompt_template, field_mapping,
                             dataset_field_mapping, max_new_token, temperature, top_p,
                              call_method=None, model_name=None, model_adapter=None,
                              api_key=None, api_url=None, model_engine=None):
    if call_method is None:
        raise ValueError("call_method must be provided.")

    with open(response_path, "r", encoding="utf-8") as file:
        response_data = json.load(file)

    responses = response_data["response"]

    if dataset_path is not None:
        with open(dataset_path, "r", encoding="utf-8") as file:
            dataset_data = json.load(file)

        dataset = dataset_data["problems"]

    total_data_num = len(responses)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "custom", ability_name, "evaluation")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"{ability_name}_evaluation_{current_time}.json")

    if call_method == "huggingface/local":
        if not model_name:
            raise ValueError("model_name must be provided for 'huggingface/local' call method.")
        pipe = init_model_pipe(model_name, model_adapter)

        data_metadata = {
            "dataset": dataset_path if dataset_path is not None else "None",
            "response": response_path,
            "model_name": model_name,
            "model_adapter": model_adapter,
            "total_data_num": total_data_num,
            "prompt_template": prompt_template,
            "max_new_token": max_new_token,
            "temperature": temperature,
            "top_p": top_p
        }

    elif call_method == "API":
        if not (api_key and api_url and model_engine):
            raise ValueError("api_key, api_url, and model_engine must be provided for 'API' call method.")
        client = OpenAI(api_key=api_key, base_url=api_url)

        data_metadata = {
            "dataset": dataset_path if dataset_path is not None else "None",
            "response": response_path,
            "api_url": api_url,
            "model_engine": model_engine,
            "total_data_num": total_data_num,
            "prompt_template": prompt_template,
            "max_new_token": max_new_token,
            "temperature": temperature,
            "top_p": top_p
        }

    else:
        raise ValueError(f"Invalid call_method: {call_method}. Must be 'huggingface/local' or 'API'.")

    mybar = st.progress(0)

    evaluations = []

    for i, response in enumerate(responses):

        if not response['success']:
            evaluation = {
                "origin_response": response,
                "success": False,
                "evaluation": "null",
            }

            evaluations.append(evaluation)
            continue

        prompt = prompt_template
        # for field in field_mapping['dataset_field']:
        #     prompt = prompt.replace(f"{{{{field}}}}", response['problem'][field_mapping[field]])

        llm_output = ""
        origin_output = ""

        try:
            for _, row in field_mapping.iterrows():
                response_field = row['response_field']
                prompt_field = row['prompt_field']

                fields = response_field.split('/')
                value = response
                # 逐层访问字段
                for field in fields:
                    if isinstance(value, dict) and field in value:
                        value = value[field]
                    else:
                        value = None
                        break

                if value is not None:
                    prompt = prompt.replace(prompt_field, value)

            # 如果 dataset_path 不是 None，并且 dataset_field_mapping 存在
            if dataset_path is not None and dataset_field_mapping is not None:
                # 使用 response 的索引 i 从 dataset 中获取对应的数据
                dataset_entry = dataset[i]
                for _, row in dataset_field_mapping.iterrows():
                    dataset_field = row['dataset_field']
                    prompt_field = row['prompt_field']
                    # 使用 dataset_entry 中的字段替换 prompt
                    prompt = prompt.replace(prompt_field, dataset_entry[dataset_field])

            if call_method == "huggingface/local":
                llm_output = get_model_response(pipe, prompt, max_new_token, temperature, top_p)
            elif call_method == "API":
                llm_output = get_api_response(client, prompt, model_engine, max_new_token, temperature, top_p)

            origin_output = llm_output
            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("Invalid JSON format")
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output.replace("\\", "\\\\"))

            evaluation = {
                "origin_response": response,
                "success": True,
                "evaluation": {
                    "judge_llm_output": origin_output,
                    "parsed_llm_output": parsed_output,
                }
            }

        except Exception as e:
            evaluation = {
                "origin_response": response,
                "success": False,
                "evaluation": {
                    "judge_llm_output": origin_output,
                    "parsed_llm_output": "null"
                }
            }
            print(e)

        evaluations.append(evaluation)
        mybar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

        with open(save_file, "w", encoding="utf-8") as file:
            json.dump({
                "metadata": data_metadata,
                "evaluation": evaluations
            }, file, indent=4, ensure_ascii=False)