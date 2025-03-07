import json
import os
import random
from datetime import datetime, timedelta
import streamlit as st
from matplotlib.style.core import available
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
from src.eval.utils import init_model_pipe, get_model_response, get_api_response


# Utility functions
def time_to_mins(time_str):
    """Convert time string to minutes."""
    t = datetime.strptime(time_str, "%H:%M")
    return t.hour * 60 + t.minute

def mins_to_time(mins):
    """Convert minutes to time string."""
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}"

def days_to_indices(days):
    """Convert consecutive days to indices (Monday=0, Sunday=6)."""
    if days < 1 or days > 7:
        raise ValueError("Days must be between 1 and 7.")
    return list(range(days))

def index_to_day(index):
    """Convert index to day of the week."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days[index]

def generate_time_slot(available_days, duration_mins):
    """Generate a valid time slot."""
    day_indices = days_to_indices(available_days)
    day_idx = random.choice(day_indices)

    # Generate start time (in 30-minute increments)
    max_start = 17 * 60 - duration_mins
    start = random.randrange(9 * 60, max_start + 1, 30)
    end = start + duration_mins
    return index_to_day(day_idx), start, end

def generate_busy_blocks(available_days, num_blocks, forbidden_day=None, forbidden_start=None, forbidden_end=None):
    """Generate busy blocks (avoiding a specified time slot)."""
    blocks = []
    day_indices = days_to_indices(available_days)

    for _ in range(num_blocks):
        while True:
            # Randomly select a day
            day_idx = random.choice(day_indices)
            day = index_to_day(day_idx)

            # Randomly generate a time slot
            start = random.randrange(9 * 60, 16 * 60 + 1, 30)  # Latest start at 16:30
            duration = random.choice([30, 60])
            end = start + duration
            if end > 17 * 60:
                end = 17 * 60

            # Check if the slot needs to avoid a specific time
            if day == forbidden_day:
                if not (end <= forbidden_start or start >= forbidden_end):
                    continue  # Conflict exists, regenerate

            blocks.append({
                "day": day,
                "start": mins_to_time(start),
                "end": mins_to_time(end)
            })
            break
    return blocks

def generate_test_case(num_people, duration_mins, available_days, busy_blocks_per_person):
    """Generate a test case."""
    # Generate a guaranteed valid time slot
    golden_day, golden_start, golden_end = generate_time_slot(available_days, duration_mins)

    participants = []
    for i in range(num_people):
        # Generate busy blocks (avoiding the golden time slot)
        busy_blocks = generate_busy_blocks(
            available_days=available_days,
            num_blocks=busy_blocks_per_person,
            forbidden_day=golden_day,
            forbidden_start=golden_start,
            forbidden_end=golden_end
        )
        participants.append({
            "name": f"Participant {i + 1}",
            "busy_blocks": busy_blocks
        })

    return {
        "parameters": {
            "num_people": num_people,
            "duration_mins": duration_mins,
            "available_days": [index_to_day(i) for i in days_to_indices(available_days)],
            "busy_blocks_per_person": busy_blocks_per_person,
            "work_hours": "09:00-17:00"
        },
        "participants": participants,
        "golden_example": {
            "day": golden_day,
            "start": mins_to_time(golden_start),
            "end": mins_to_time(golden_end)
        }
    }

def generate_calender_problem(calender_data_config, config_df):
    with st.status("Generating calender problems", expanded=True) as status:
        problems = []

        st.write("Checking config...")
        for index, row in config_df.iterrows():
            num_people = row['num_people']
            duration_mins = row['duration_mins']
            total_days = row['total_days']
            busy_blocks = row['busy_blocks']
            num_data = row['num_data']

            if not 2 <= num_people <= 10:
                raise ValueError("Number of people must be between 2 and 10.")
            if not 0 <= duration_mins:
                raise ValueError("Duration must be a positive integer.")
            if not 1 <= total_days <= 7:
                raise ValueError("Total days must be between 1 and 7.")
            if not 0 <= busy_blocks:
                raise ValueError("Busy blocks must be a non-negative integer.")
            if not 1 <= num_data:
                raise ValueError("Number of data must be a positive integer.")

        st.write("Generating problems...")

        for index, row in config_df.iterrows():
            num_people = int(row['num_people'])
            duration_mins = int(row['duration_mins'])
            total_days = int(row['total_days'])
            busy_blocks = int(row['busy_blocks'])
            num_data = int(row['num_data'])

            for i in range(num_data):
                test_case = generate_test_case(
                    num_people=num_people,
                    duration_mins=duration_mins,
                    available_days=total_days,
                    busy_blocks_per_person=busy_blocks
                )
                problems.append(
                    {
                        "metadata": {
                            "num_people": num_people,
                            "duration_mins": duration_mins,
                            "total_days": total_days,
                            "busy_blocks": busy_blocks,
                        },
                        "problem": test_case
                    }
                )

        st.write("Saving problems...")

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(calender_data_config['calender_data_folder'], f"calender_data_{current_time}.json")

        os.makedirs(calender_data_config['calender_data_folder'], exist_ok=True)

        output_data = {
            "metadata": config_df.to_dict(orient="records"),
            "problems": problems
        }
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4, ensure_ascii=False)

        status.update(
            label=f"Generated successfully! See {output_file}.",
            state="complete",
            expanded=False
        )

def generate_calender_response_model(dataset_path, model_name, model_adapter, task_name,
                                     prompt_template, max_new_token, temperature, top_p):
    required_fields = ["{{num_people}}", "{{duration_mins}}", "{{available_days}}", "{{busy_blocks}}"]
    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    problems = data["problems"]

    total_data_num = len(problems)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "calender", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"calender_response_{current_time}.json")

    metadata = {
        "dataset": dataset_path,
        "model_name": model_name,
        "model_adapter": model_adapter,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_adapter:
        peft_config = PeftConfig.from_pretrained(model_adapter)
        model = PeftModel.from_pretrained(model, model_adapter)
        model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    mybar = st.progress(0)

    responses = []

    for i, problem in enumerate(problems):
        available_days = ", ".join(problem["problem"]["parameters"]["available_days"])
        busy_blocks = ""

        for participant in problem["problem"]["participants"]:
            busy_blocks += f"{participant['name']} is busy when:\n"
            for block in participant["busy_blocks"]:
                busy_blocks += f"- {block['day']}, {block['start']} - {block['end']}\n"

        prompt = prompt_template.replace("{{num_people}}", str(problem["metadata"]["num_people"])) \
            .replace("{{duration_mins}}", str(problem["metadata"]["duration_mins"])) \
            .replace("{{available_days}}", available_days) \
            .replace("{{busy_blocks}}", busy_blocks)

        messages = [
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "text_inputs": messages,
            "max_new_tokens": max_new_token,
            "top_p": top_p
        }

        if temperature == 0.0:
            kwargs["do_sample"] = False
        else:
            kwargs["temperature"] = temperature

        llm_output = ""
        origin_output = ""

        try:
            llm_output = pipe(**kwargs)[0]["generated_text"][-1]["content"].strip().replace("\n", "")

            origin_output = llm_output
            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1

            if json_start == -1 or json_end == -1:
                raise ValueError("Invalid JSON format")
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output)

            response = {
                "metadata": problem["metadata"],
                "success": True,
                "problem": problem['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }
        except Exception as e:
            response = {
                "metadata": problem["metadata"],
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
            json.dump(responses, file, indent=4, ensure_ascii=False)

    st.success(f"Calender response generated successfully! See {save_file}.")

def generate_calender_response_api(dataset_path, api_key, api_url, model_engine, task_name,
                                  prompt_template, max_new_token, temperature, top_p):
    required_fields = ["{{num_people}}", "{{duration_mins}}", "{{available_days}}", "{{busy_blocks}}"]
    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    problems = data["problems"]

    total_data_num = len(problems)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "calender", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"calender_response_{current_time}.json")

    metadata = {
        "dataset": dataset_path,
        "api_url": api_url,
        "model_engine": model_engine,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    client = OpenAI(api_key=api_key, base_url=api_url)

    mybar = st.progress(0)

    responses = []

    for i, problem in enumerate(problems):
        available_days = ", ".join(problem["problem"]["parameters"]["available_days"])
        busy_blocks = ""

        for participant in problem["problem"]["participants"]:
            busy_blocks += f"{participant['name']} is busy when:\n"
            for block in participant["busy_blocks"]:
                busy_blocks += f"- {block['day']}, {block['start']} - {block['end']}\n"

        prompt = prompt_template.replace("{{num_people}}", str(problem["metadata"]["num_people"])) \
            .replace("{{duration_mins}}", str(problem["metadata"]["duration_mins"])) \
            .replace("{{available_days}}", available_days) \
            .replace("{{busy_blocks}}", busy_blocks)

        messages = [
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "text_inputs": messages,
            "max_new_tokens": max_new_token,
            "top_p": top_p
        }

        if temperature == 0.0:
            kwargs["do_sample"] = False
        else:
            kwargs["temperature"] = temperature

        llm_output = ""
        origin_output = ""

        try:
            llm_output = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_token,
                temperature=temperature,
                top_p=top_p,
            )

            llm_output = llm_output.choices[0].message.content.strip().replace("\n", "")

            origin_output = llm_output
            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output)

            response = {
                "metadata": problem["metadata"],
                "success": True,
                "problem": problem['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }
        except Exception as e:
            response = {
                "metadata": problem["metadata"],
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
                "metadata": metadata,
                "response": responses
            }, file, indent=4, ensure_ascii=False)

    st.success(f"Calender response generated successfully! See {save_file}.")

def generate_calender_response(dataset_path, task_name, prompt_template, max_new_token, temperature, top_p,
                              call_method=None, model_name=None, model_adapter=None,
                              api_key=None, api_url=None, model_engine=None):
    if call_method is None:
        raise ValueError("call_method must be provided.")

    required_fields = ["{{num_people}}", "{{duration_mins}}", "{{available_days}}", "{{busy_blocks}}"]
    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    problems = data["problems"]

    total_data_num = len(problems)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "calender", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"calender_response_{current_time}.json")

    if call_method == "huggingface/local":
        if not model_name:
            raise ValueError("model_name must be provided for 'huggingface/local' call method.")
        pipe = init_model_pipe(model_name, model_adapter)

        metadata = {
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

        metadata = {
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
        available_days = ", ".join(problem["problem"]["parameters"]["available_days"])
        busy_blocks = ""

        for participant in problem["problem"]["participants"]:
            busy_blocks += f"{participant['name']} is busy when:\n"
            for block in participant["busy_blocks"]:
                busy_blocks += f"- {block['day']}, {block['start']} - {block['end']}\n"

        prompt = prompt_template.replace("{{num_people}}", str(problem["metadata"]["num_people"])) \
            .replace("{{duration_mins}}", str(problem["metadata"]["duration_mins"])) \
            .replace("{{available_days}}", available_days) \
            .replace("{{busy_blocks}}", busy_blocks)

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
            parsed_output = json.loads(llm_output)

            response = {
                "metadata": problem["metadata"],
                "success": True,
                "problem": problem['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            }
        except Exception as e:
            response = {
                "metadata": problem["metadata"],
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
                "metadata": metadata,
                "response": responses
            }, file, indent=4, ensure_ascii=False)

    st.success(f"Calender response generated successfully! See {save_file}.")

def validate_solution(proposed_time, test_case):
    """Validate the proposed solution."""
    try:
        # Parse the proposed time
        proposed_day = proposed_time["day"]

        # Convert to minutes
        start = time_to_mins(proposed_time["start"])
        end = time_to_mins(proposed_time["end"])

        if end < start:
            return {
                "passed": False,
                "errors": [
                    {
                        "type": "Invalid Time Range",
                        "reason": "End time is earlier than start time."
                    }
                ]
            }
    except:
        return {
            "passed": False,
            "errors": [
                {
                    "type": "Invalid Format",
                    "reason": f"Invalid time format: {proposed_time}"
                }
            ]
        }

    params = test_case["parameters"]

    # Check day validity
    if proposed_day not in params["available_days"]:
        return {
            "passed": False,
            "errors": [
                {
                    "type": "Invalid Day",
                    "reason": f"Day is not within the allowed range({params["available_days"]}): {proposed_day}"
                }
            ]
        }

    # Check working hours
    if start < 9 * 60 or end > 17 * 60:
        return {
            "passed": False,
            "errors": [
                {
                    "type": "Outside Working Hours",
                    "reason": f"Time is outside working hours: {proposed_time}"
                }
            ]
        }
    # Check duration
    if end - start != params["duration_mins"]:
        return {
            "passed": False,
            "errors": [
                {
                    "type": "Invalid Duration",
                    "reason": f"Meeting duration does not meet requirements: {proposed_time}"
                }
            ]
        }
    # Check conflicts with all participants
    for participant in test_case["participants"]:
        for block in participant["busy_blocks"]:
            if block["day"] != proposed_day:
                continue

            block_start = time_to_mins(block["start"])
            block_end = time_to_mins(block["end"])

            if not (end <= block_start or start >= block_end):
                return {
                    "passed": False,
                    "errors": [
                        {
                            "type": "Schedule Conflict",
                            "reason": f"Conflict with {participant['name']}'s schedule: {proposed_time}"
                        }
                    ]
                }

    return {
        "passed": True,
        "errors": []
    }

def evaluate_calender_response(response_path, task):
    with open(response_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    responses = data["response"]

    total_data_num = len(responses)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task, "planning", "calender", "evaluation")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"calender_evaluation_{current_time}.json")

    mybar = st.progress(0)

    evaluations = []

    for i, response in enumerate(responses):
        proposed_time = response["llm_output"]
        test_case = response["problem"]

        evaluation = validate_solution(proposed_time, test_case)

        evaluations.append({
            "metadata": response["metadata"],
            "success": response["success"],
            "problem": response["problem"],
            "prompt": response["prompt"],
            "origin_output": response["origin_output"],
            "llm_output": response["llm_output"],
            "evaluation": evaluation
        })
        mybar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(evaluations, file, indent=4, ensure_ascii=False)

    st.success(f"Calender evaluation generated successfully! See {save_file}.")