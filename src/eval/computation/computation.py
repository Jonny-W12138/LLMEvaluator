import json
import random
import os
from datetime import datetime
import streamlit as st
from src.eval.utils import init_model_pipe, get_model_response, get_api_response
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                           api_key=None, api_url=None, model_engine=None,
                           if_parallel=False, parallel_num=4):

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
        raise ValueError(f"Invalid call_method: {call_method}.")

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": data_metadata, "response": []}, f, indent=4)

    mybar = st.progress(0)
    progress = 0

    def process_one(index):
        problem = problems[index]
        prompt = prompt_template.replace("{{math_problem}}", problem['problem'])
        metadata = {
            "level": problem["level"],
            "type": problem["type"],
            "solution": problem["solution"],
            "answer": problem["answer"]
        }
        origin_output, parsed_output = "", ""

        try:
            if call_method == "huggingface/local":
                origin_output = get_model_response(pipe, prompt, max_new_token, temperature, top_p)
            elif call_method == "API":
                origin_output = get_api_response(client, prompt, model_engine, max_new_token, temperature, top_p)

            json_start = origin_output.find("{")
            json_end = origin_output.rfind("}")
            if json_start == -1 or json_end == -1:
                raise ValueError("No valid JSON found.")
            parsed_output = json.loads(origin_output[json_start:json_end + 1].replace("\\", "\\\\"))

            return index, {
                "metadata": metadata,
                "success": True,
                "problem": problem["problem"],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }

        except Exception as e:
            print(f"[ERROR {index}] {e}")
            return index, {
                "metadata": metadata,
                "success": False,
                "problem": problem["problem"],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }

    if if_parallel:
        with ThreadPoolExecutor(max_workers=parallel_num) as executor:
            for start in range(0, total_data_num, parallel_num):
                indices = list(range(start, min(start + parallel_num, total_data_num)))
                futures = {executor.submit(process_one, i): i for i in indices}
                results = {}

                for future in as_completed(futures):
                    i, res = future.result()
                    results[i] = res

                # 读取已有保存文件
                with open(save_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)

                for i in indices:
                    saved["response"].append(results[i])
                    progress += 1
                    mybar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=4, ensure_ascii=False)
    else:
        for i in range(total_data_num):
            _, res = process_one(i)

            with open(save_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved["response"].append(res)
            progress += 1
            mybar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(saved, f, indent=4, ensure_ascii=False)

    st.success(f"Math response generation completed. Results saved to {save_file}")

def evaluate_math_response(response_path, task_name, prompt_template, max_new_token, temperature, top_p,
                           call_method=None, model_name=None, model_adapter=None,
                           api_key=None, api_url=None, model_engine=None,
                           if_parallel=False, parallel_num=4):
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
        pipe = init_model_pipe(model_name, model_adapter)
    elif call_method == "API":
        client = OpenAI(api_key=api_key, base_url=api_url)
    else:
        raise ValueError(f"Invalid call_method: {call_method}")

    metadata = {
        "dataset": data["metadata"]["dataset"],
        "response": response_path,
        "model_name": model_name if call_method == "huggingface/local" else None,
        "model_adapter": model_adapter if call_method == "huggingface/local" else None,
        "api_url": api_url if call_method == "API" else None,
        "model_engine": model_engine if call_method == "API" else None,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    def evaluate_single(i):
        response = responses[i]
        if not response['success']:
            return i, {
                "origin_response": response,
                "success": False,
                "evaluation": {
                    "judge_llm_output": "",
                    "eval_steps": "",
                    "if_correct": "",
                }
            }

        origin_output = ""
        try:
            metadata = response['metadata']
            math_problem = response['problem']
            solution = metadata['solution']
            answer = metadata['answer']

            steps = response['llm_output']['steps']
            solution_steps = "\n".join([f"Step {idx + 1}. {desc}" for idx, (_, desc) in enumerate(steps.items())])

            prompt = prompt_template.replace("{{math_problem}}", math_problem) \
                                    .replace("{{solution_to_evaluate}}", solution_steps) \
                                    .replace("{{answer_to_evaluate}}", response['llm_output']['answer']) \
                                    .replace("{{reference_solution}}", solution) \
                                    .replace("{{reference_answer}}", answer)

            if call_method == "huggingface/local":
                llm_output = get_model_response(pipe, prompt, max_new_token, temperature, top_p)
            else:
                llm_output = get_api_response(client, prompt, model_engine, max_new_token, temperature, top_p)

            origin_output = llm_output
            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1
            parsed_output = json.loads(llm_output[json_start:json_end].replace("\\", "\\\\"))

            result = {
                "origin_response": response,
                "success": True,
                "evaluation": {
                    "judge_llm_output": origin_output,
                    "eval_steps": parsed_output["steps"],
                    "if_correct": parsed_output["answer"]
                }
            }
        except Exception as e:
            print(f"[ERROR index {i}]:", e)
            result = {
                "origin_response": response,
                "success": False,
                "evaluation": {
                    "judge_llm_output": origin_output,
                    "eval_steps": "",
                    "if_correct": "",
                }
            }
        return i, result

    mybar = st.progress(0)
    progress = 0
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "evaluation": []}, f, indent=4, ensure_ascii=False)

    if if_parallel:
        with ThreadPoolExecutor(max_workers=parallel_num) as executor:
            for start in range(0, total_data_num, parallel_num):
                indices = list(range(start, min(start + parallel_num, total_data_num)))
                futures = {executor.submit(evaluate_single, i): i for i in indices}
                results = {}

                for future in as_completed(futures):
                    idx, res = future.result()
                    results[idx] = res

                with open(save_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)

                for i in indices:
                    saved["evaluation"].append(results[i])
                    progress += 1
                    mybar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=4, ensure_ascii=False)
    else:
        evaluations = []
        for i in range(total_data_num):
            _, eval_result = evaluate_single(i)

            with open(save_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

            saved["evaluation"].append(eval_result)
            progress += 1
            mybar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(saved, f, indent=4, ensure_ascii=False)

    st.success(f"Evaluation completed! Results saved to {save_file}")