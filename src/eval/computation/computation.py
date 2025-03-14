import json
import random
import os
from datetime import datetime
import streamlit as st
from src.eval.utils import init_model_pipe, get_model_response, get_api_response
from openai import OpenAI

def generate_random_problem(selected_config):
    with open(os.path.join(os.getcwd(), "dataset", "computation", "data_config.json"), 'r', encoding='utf-8') as f:
        data_config = json.load(f)
    data_config = data_config['math']
    dataset_path = data_config['data_path']

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categorized_data = {}
    for item in data:
        key = (item["type"], item["level"])
        if key not in categorized_data:
            categorized_data[key] = []
        categorized_data[key].append(item)

    selected_problems = []

    for config in selected_config:
        t, level, num_problems = config["type"], config["level"], config["num_problems"]
        key = (t, level)

        if key not in categorized_data or len(categorized_data[key]) < num_problems:
            raise ValueError(
                f"Insufficient problems: {t} (Level {level}) has only {len(categorized_data.get(key, []))} problems available, but {num_problems} are required.")

        selected_problems.extend(random.sample(categorized_data[key], num_problems))

    output_json = {
        "metadata": selected_config,
        "problems": selected_problems
    }

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(data_config['data_folder'], f"math_data_{current_time}.json")

    os.makedirs(data_config['data_folder'], exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    st.success(f"Generated successfully. See {output_file}!")

def generate_math_response(dataset_path, task_name, prompt_template, max_new_token, temperature, top_p,
                              call_method=None, model_name=None, model_adapter=None,
                              api_key=None, api_url=None, model_engine=None):
    if call_method is None:
        raise ValueError("call_method must be provided.")

    required_fields = ["{{math_problem}}"]
    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    problems = data["problems"]

    total_data_num = len(problems)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "computation", "math", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"math_response_{current_time}.json")

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
        math_problem = problem['problem']
        metadata = {
            "level": problem['level'],
             "type": problem['type'],
             "solution": problem['solution'],
             "answer": problem['answer']
        }

        prompt = prompt_template.replace("{{math_problem}}", math_problem)

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
                "metadata": metadata,
                "success": True,
                "problem": problem['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }

        except Exception as e:
            response = {
                "metadata": metadata,
                "success": False,
                "problem": problem['problem'],
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

    st.success(f"Math response generated successfully! See {save_file}.")

def evaluate_math_response(response_path, task_name, prompt_template, max_new_token, temperature, top_p,
                              call_method=None, model_name=None, model_adapter=None,
                              api_key=None, api_url=None, model_engine=None):
    if call_method is None:
        raise ValueError("call_method must be provided.")

    required_fields = ["{{math_problem}}", "{{solution_to_evaluate}}", "{{reference_solution}}", "{{answer_to_evaluate}}", "{{reference_answer}}"]
    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(response_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    responses = data["response"]

    total_data_num = len(responses)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "computation", "math", "evaluation")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"math_evaluation_{current_time}.json")

    if call_method == "huggingface/local":
        if not model_name:
            raise ValueError("model_name must be provided for 'huggingface/local' call method.")
        pipe = init_model_pipe(model_name, model_adapter)

        data_metadata = {
            "dataset": data['metadata']['dataset'],
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
            "dataset": data['metadata']['dataset'],
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
                "evaluation": {
                    "judge_llm_output" : "",
                    "eval_steps": "",
                    "if_correct": "",
                }
            }

            evaluations.append(evaluation)
            continue

        metadata = response['metadata']
        math_problem = response['problem']
        solution = metadata['solution']
        answer = metadata['answer']

        solution_steps = ""
        for index, (step_key, step_description) in enumerate(response['llm_output']['steps'].items(), start=1):
            solution_steps += f"Step {index}. {step_description}\n"


        prompt = prompt_template.replace("{{math_problem}}", math_problem) \
                .replace("{{solution_to_evaluate}}", solution_steps) \
                .replace("{{answer_to_evaluate}}", response['llm_output']['answer']) \
                .replace("{{reference_solution}}", solution) \
                .replace("{{reference_answer}}", answer)

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

            evaluation = {
                "origin_response": response,
                "success": True,
                "evaluation": {
                    "judge_llm_output" : origin_output,
                    "eval_steps": parsed_output['steps'],
                    "if_correct": parsed_output['answer'],
                }
            }

        except Exception as e:
            evaluation = {
                "origin_response": response,
                "success": False,
                "evaluation": {
                    "judge_llm_output" : origin_output,
                    "eval_steps": "",
                    "if_correct": "",
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

    st.success(f"Evaluation generated successfully! See {save_file}.")