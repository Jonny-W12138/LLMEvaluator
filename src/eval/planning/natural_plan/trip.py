import json
import random
from collections import defaultdict
from itertools import permutations
from openai import OpenAI
import os
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import streamlit as st

def generate_trip_problem(trip_data_config, config_df):
    """
    根据前端配置生成旅行规划问题。

    Args:
        trip_data_config (dict): 包含城市数据路径和输出文件夹路径的配置。
        config_df (pd.DataFrame): 包含生成配置的 DataFrame。

    Returns:
        dict: 包含 metadata 和 data 的 JSON 格式数据。
    """

    with st.status("Generating trip dataset...", expanded=True) as status:
        st.write("Loading cities data...")
        with open(trip_data_config['cities_path'], 'r', encoding='utf-8') as f:
            cities = json.load(f)

        st.write("Checking config...")
        for index, row in config_df.iterrows():
            num_cities = int(row["num_cities"])
            min_stay = int(row["min_stay"])
            max_stay = int(row["max_stay"])
            constraint_days = int(row["constraint_days"])
            direct_flight_rate = float(row["direct_flight_rate"])
            trips_amount = int(row["trips_amount"])

            if not (3 <= num_cities <= 50):
                raise ValueError("num_cities must between 3 and 50")
            if not (2 <= min_stay <= 7):
                raise ValueError("min_stay must between 2 and 7")
            if not (2 <= max_stay <= 7):
                raise ValueError("max_stay must between 2 and 7")
            if not (1 <= constraint_days <= num_cities):
                raise ValueError("constraint_daysmust between 1 and num_cities")
            if not (0 <= direct_flight_rate <= 1):
                raise ValueError("direct_flight_rate must between 0 and 1")
            if trips_amount < 1:
                raise ValueError("trips_amount above 0")

        result = {
            "metadata": {
                "config": config_df.to_dict(orient="records")
            },
            "data": []
        }

        st.write("Generating trip data...")
        for index, row in config_df.iterrows():
            num_cities = int(row["num_cities"])
            min_stay = int(row["min_stay"])
            max_stay = int(row["max_stay"])
            constraint_days = int(row["constraint_days"])
            direct_flight_rate = float(row["direct_flight_rate"])
            trips_amount = int(row["trips_amount"])

            for _ in range(trips_amount):

                selected = random.sample(cities, k=num_cities)
                stays = {city: random.randint(min_stay, max_stay) for city in selected}
                path = selected.copy()

                current_day = 1
                constraint_candidates = []
                for city in path:
                    stay_days = stays[city]
                    end_day = current_day + stay_days - 1
                    constraint_candidates.append((city, (current_day, end_day)))
                    current_day += stay_days

                constraint = random.choice(constraint_candidates)
                constraint_city, (start, end) = constraint
                constraint_days_range = (start, min(end, start + constraint_days - 1))


                flights = defaultdict(dict)

                for i in range(len(path) - 1):
                    flights[path[i]][path[i + 1]] = True

                for a, b in permutations(selected, 2):
                    if b not in flights[a]:
                        flights[a][b] = random.choices([True, False], weights=[direct_flight_rate, 1 - direct_flight_rate])[0]

                keys = list(flights.keys())
                random.shuffle(keys)
                shuffled_flights = {key: flights[key] for key in keys}
                shuffled_cities = selected.copy()
                random.shuffle(shuffled_cities)

                stays_items = list(stays.items())
                random.shuffle(stays_items)
                shuffled_stays = dict(stays_items)

                feasible_solution = {}
                current_day = 1
                for city in path:
                    stay_days = stays[city]
                    end_day = current_day + stay_days - 1
                    feasible_solution[city] = {
                        "start": current_day,
                        "end": end_day
                    }
                    current_day += stay_days

                problem = {
                    'cities': shuffled_cities,
                    'stays': shuffled_stays,
                    'flights': shuffled_flights,
                    'constraint': {'city': constraint_city, 'days': constraint_days_range}
                }

                result["data"].append({
                    "config": {
                        "num_cities": num_cities,
                        "min_stay": min_stay,
                        "max_stay": max_stay,
                        "constraint_days": constraint_days,
                        "direct_flight_rate": direct_flight_rate,
                        "trips_amount": trips_amount
                    },
                    "problem": problem,
                    "feasible_solution": feasible_solution
                })

        st.write("Saving trip data...")
        if not os.path.exists(trip_data_config['trip_data_folder']):
            os.makedirs(trip_data_config['trip_data_folder'])
        trip_data_folder = trip_data_config['trip_data_folder']

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(trip_data_folder, f"trip_data_{current_time}.json")

        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

        status.update(
            label=f"Generated successfully! See the {output_file}.",
            state="complete",
            expanded=False
        )

def generate_trip_response_model(dataset_path, model_name, model_adapter, task_name,
                                 prompt_template, max_new_token, temperature, top_p):

    required_fields = ["{{city_nums}}", "{{total_day}}", "{{stay_requirements}}", "{{flight_requirements}}"]
    for field in required_fields:
        if field not in prompt_template:
            st.error(f"Error: Prompt template is missing required field: {field}")
            return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    data = data["data"]
    total_data_num = len(data)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "trip", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"trip_response_{current_time}.json")

    metadata = {
        "dataset_path": dataset_path,
        "model_name": model_name,
        "model_adapter": model_adapter,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    responses = []


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

    for item in data:
        item_config = item["config"]
        item_problem = item["problem"]
        cities = item_problem["cities"]
        stays = item_problem["stays"]
        flights = item_problem["flights"]
        constraint = item_problem["constraint"]

        city_nums = len(cities)
        total_day = sum(stays.values())
        stay_requirements = []
        for city, days in stays.items():
            stay_req = f"You would like to visit {city} for {days} days."
            if city == constraint["city"]:
                start_day, end_day = constraint["days"]
                if start_day == end_day:
                    stay_req += f" You must be in {city} in day {start_day}."
                else:
                    stay_req += f" You must be in {city} between day {start_day} and day {end_day}."
            stay_requirements.append(stay_req)
        stay_requirements = "\n".join(stay_requirements)

        flight_requirements = []
        for city_a, connections in flights.items():
            for city_b, is_direct in connections.items():
                if is_direct:
                    flight_requirements.append(f"{city_a} and {city_b}")
        flight_requirements = ", ".join(flight_requirements) + "."

        prompt = prompt_template.replace("{{city_nums}}", str(city_nums)) \
                               .replace("{{total_day}}", str(total_day)) \
                               .replace("{{stay_requirements}}", stay_requirements) \
                               .replace("{{flight_requirements}}", flight_requirements)

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
            # 生成 LLM 输出
            llm_output = pipe(**kwargs)[0]["generated_text"][-1]["content"].strip().replace("\n", "")

            origin_output = llm_output
            # 解析 LLM 输出的 JSON 内容

            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output)

            response = {
                "config": item_config,
                "success": True,
                "problem": item_problem,
                "prompt": prompt,
                "feasible_solution": item.get("feasible_solution", []),
                "llm_output": origin_output,
                "parsed_output": parsed_output
            }

        except Exception as e:
            response = {
                "config": item_config,
                "success": False,
                "problem": item_problem,
                "prompt": prompt,
                "feasible_solution": item.get("feasible_solution", []),
                "llm_output": origin_output,
                "parsed_output": ""
            }
            print(f"Error at index {data.index(item)}: {e}")

        responses.append(response)
        mybar.progress((data.index(item) + 1) / total_data_num, text=f"Processing {data.index(item) + 1}/{total_data_num}")

        # 将当前结果保存到 JSON 文件
        with open(save_file, "w") as f:
            json.dump({"metadata": metadata, "responses": responses}, f, indent=4)

    st.success(f"Trip response generation completed. Results saved to {save_file}")


def generate_trip_response_api(dataset_path, api_key, api_url, model_engine, task_name,
                               prompt_template, max_new_token, temperature, top_p):
    # 检查 prompt_template 是否包含所有需要的字段
    required_fields = ["{{city_nums}}", "{{total_day}}", "{{stay_requirements}}", "{{flight_requirements}}"]
    for field in required_fields:
        if field not in prompt_template:
            st.error(f"Error: Prompt template is missing required field: {field}")
            return

    with open(dataset_path, "r") as f:
        data = json.load(f)
    data = data["data"]

    total_data_num = len(data)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "trip", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"trip_response_{current_time}.json")

    metadata = {
        "dataset_path": dataset_path,
        "api_url": api_url,
        "model_engine": model_engine,
        "total_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    responses = []

    client = OpenAI(api_key=api_key, base_url=api_url)

    my_bar = st.progress(0)

    for item in data:
        item_config = item["config"]
        item_problem = item["problem"]
        cities = item_problem["cities"]
        stays = item_problem["stays"]
        flights = item_problem["flights"]
        constraint = item_problem["constraint"]

        # 替换 prompt_template 中的字段
        city_nums = len(cities)
        total_day = sum(stays.values())
        stay_requirements = []
        for city, days in stays.items():
            stay_req = f"You would like to visit {city} for {days} days."
            if city == constraint["city"]:
                start_day, end_day = constraint["days"]
                if start_day == end_day:
                    stay_req += f" You want to meet a friend in {city} in day {start_day}."
                else:
                    stay_req += f" You want to meet a friend in {city} between day {start_day} and day {end_day}."
            stay_requirements.append(stay_req)
        stay_requirements = "\n".join(stay_requirements)

        flight_requirements = []
        for city_a, connections in flights.items():
            for city_b, is_direct in connections.items():
                if is_direct:
                    flight_requirements.append(f"{city_a} and {city_b}")
        flight_requirements = ", ".join(flight_requirements) + "."

        prompt = prompt_template.replace("{{city_nums}}", str(city_nums)) \
            .replace("{{total_day}}", str(total_day)) \
            .replace("{{stay_requirements}}", stay_requirements) \
            .replace("{{flight_requirements}}", flight_requirements)

        llm_output = ""
        try:
            response = client.chat.completions.create(
                model=model_engine,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_token,
                temperature=temperature,
                top_p=top_p,
                # n=1,
                # stop=None,
            )

            llm_output = response.choices[0].message.content.strip().replace("\n", "")

            # 去除首尾的空白字符并替换换行符
            llm_output = llm_output.strip().replace("\n", "")
            llm_output_original = llm_output

            json_start = llm_output.index("{")
            json_end = llm_output.rindex("}") + 1
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output)

            response = {
                "config": item_config,
                "success": True,
                "problem": item_problem,
                "prompt": prompt,
                "feasible_solution": item.get("feasible_solution", []),
                "llm_output": llm_output_original,
                "parsed_output": parsed_output
            }

        except Exception as e:
            response = {
                "config": item_config,
                "success": False,
                "problem": item_problem,
                "prompt": prompt,
                "feasible_solution": item.get("feasible_solution", []),
                "llm_output": llm_output_original,
                "parsed_output": ""
            }

            print(f"Error at index {data.index(item)}: {e}")

        responses.append(response)
        my_bar.progress((data.index(item) + 1) / total_data_num, text=f"Processing {data.index(item) + 1}/{total_data_num}")
        # time.sleep(25)

        # 将当前结果保存到 JSON 文件
        with open(save_file, "w") as f:
            json.dump({"metadata": metadata, "responses": responses}, f, indent=4)

    st.success(f"Trip response generation completed. Results saved to {save_file}")


def evaluate_trip_response(response_path, task_name):
    with open(response_path, "r") as f:
        data = json.load(f)

    metadata = data['metadata']
    responses = data['responses']

    total_data_num = len(responses)

    # Prepare the folder and filename for saving results
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "trip", "evaluation")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"trip_evaluate_{current_time}.json")

    # Initialize the data structure for saving results
    evaluated_responses = []

    my_bar = st.progress(0)

    for response in responses:
        evaluated_response = response.copy()  # Copy the original data
        evaluated_response['evaluation'] = {
            "passed": True,
            "errors": []
        }

        if response['success']:
            # If success is True, perform evaluation
            is_valid, errors = validate(response['parsed_output'], response['problem'])
            evaluated_response['evaluation']['passed'] = is_valid
            if not is_valid:
                evaluated_response['evaluation']['errors'] = errors

        evaluated_responses.append(evaluated_response)
        my_bar.progress((responses.index(response) + 1) / total_data_num, text=f"Processing {responses.index(response) + 1}/{total_data_num}")

    # Save the evaluation results
    with open(save_file, "w") as f:
        json.dump({
            'metadata': metadata,
            'responses': evaluated_responses
        }, f, indent=4)


def validate(solution, problem):
    """Validate whether the given solution satisfies the problem constraints."""
    errors = []

    # Convert solution to itinerary and stays
    itinerary = [city for city in solution.keys()]
    stays = {city: solution[city]['end'] - solution[city]['start'] + 1 for city in solution}

    # Validate that each city is visited only once
    if len(itinerary) != len(set(itinerary)):
        duplicates = [city for city in set(itinerary) if itinerary.count(city) > 1]
        errors.append({
            "type": "Repeated city visits",
            "reason": f"City(s) {', '.join(duplicates)} are visited multiple times."
        })

    # Validate the stay durations
    if stays != problem['stays']:
        mismatched_cities = [city for city in stays if stays[city] != problem['stays'].get(city)]
        errors.append({
            "type": "Stay durations do not match",
            "reason": f"Stay durations for {', '.join(mismatched_cities)} do not match the problem constraints."
        })

    # Validate flight continuity
    for i in range(len(itinerary) - 1):
        if not problem['flights'][itinerary[i]].get(itinerary[i + 1], False):
            errors.append({
                "type": "No direct flight",
                "reason": f"No direct flight from {itinerary[i]} to {itinerary[i + 1]}."
            })

    # Sort itinerary by start day to ensure correct order
    sorted_itinerary = sorted(itinerary, key=lambda x: solution[x]['start'])

    # Validate continuity of days (no gaps or overlaps)
    for i in range(len(sorted_itinerary) - 1):
        current_city = sorted_itinerary[i]
        next_city = sorted_itinerary[i + 1]
        current_end = solution[current_city]['end']
        next_start = solution[next_city]['start']

        # Check if days are continuous
        if next_start != current_end + 1:
            errors.append({
                "type": "Days are not continuous",
                "reason": f"{current_city} ends on day {current_end}, but {next_city} starts on day {next_start}."
            })

        # Check for overlapping days
        current_range = range(solution[current_city]['start'], solution[current_city]['end'] + 1)
        next_range = range(solution[next_city]['start'], solution[next_city]['end'] + 1)
        if set(current_range).intersection(next_range):
            errors.append({
                "type": "Overlapping days",
                "reason": f"Days overlap between {current_city} and {next_city}."
            })

    # Validate the time constraints
    current_day = 1
    target_days = set(range(problem['constraint']['days'][0], problem['constraint']['days'][1] + 1))
    visited_days = set()

    for city in sorted_itinerary:
        stay = stays[city]
        city_days = range(current_day, current_day + stay)
        if city == problem['constraint']['city']:
            visited_days.update(city_days)
        current_day += stay

    if not target_days.issubset(visited_days):
        missing_days = target_days - visited_days
        errors.append({
            "type": "Time constraints not satisfied",
            "reason": f"Time constraints for {problem['constraint']['city']} are not satisfied. Missing days: {', '.join(map(str, missing_days))}."
        })

    if errors:
        return False, errors
    else:
        return True, []