import json
import os
import random
import datetime

import streamlit as st
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


# 计算两地之间的通勤时间
def get_travel_time(cities_data, city, start, end):
    """Get travel time between two locations"""
    for route in cities_data.get(city, []):
        if route["start"] == start and route["end"] == end:
            return route["duration"]
    return None

def generate_random_problem(city_data, city, num_places, conflict_pairs):
    unique_places = list(
        set([route["start"] for route in city_data[city]] + [route["end"] for route in city_data[city]]))

    if len(unique_places) < num_places:
        raise ValueError(f"城市 {city} 可选地点不足 {num_places} 个，当前仅有 {len(unique_places)} 个。")

    visit_info = {}
    available_time_slots = []  # 存放时间段

    base_places = random.sample(unique_places, num_places)  # 先选出所有地点
    non_conflict_places = num_places - 2 * conflict_pairs  # 计算无冲突的地点数量

    if non_conflict_places < 0:
        raise ValueError(f"冲突对数 {conflict_pairs} 过大，最多可以有 {num_places // 2} 对冲突。")

    standalone_places = base_places[:non_conflict_places]  # 无冲突地点
    conflict_place_pairs = [base_places[i:i + 2] for i in range(non_conflict_places, num_places, 2)]  # 形成冲突对

    # **生成 (num_places - conflict_pairs) 个时间段**
    start_time = datetime.datetime.strptime("09:00", "%H:%M")
    start_time = start_time + datetime.timedelta(minutes=random.randint(0, 30))  # 随机起始时间

    for _ in range(num_places - conflict_pairs):
        stay_duration = random.randint(10, 30)  # 每个地点停留 10~30 分钟
        end_time = (start_time + datetime.timedelta(minutes=stay_duration)
                    +  datetime.timedelta(minutes=random.randint(5, 50)))

        available_time_slots.append((start_time, end_time))  # 记录时间段
        start_time = end_time + datetime.timedelta(minutes=random.randint(20, 50))  # 留出间隔

    # **分配无冲突地点**
    for place, (start, end) in zip(standalone_places, available_time_slots[:non_conflict_places]):
        visit_info[place] = {
            "stay_duration": (end - start).seconds // 60,
            "available_time": [start.strftime("%H:%M"), end.strftime("%H:%M")]
        }

    # **分配冲突地点**
    conflict_time_slots = available_time_slots[non_conflict_places:]  # 剩余时间段用于冲突对
    for (place1, place2), (base_start, base_end) in zip(conflict_place_pairs, conflict_time_slots):
        visit_info[place1] = {
            "stay_duration": (base_end - base_start).seconds // 60,
            "available_time": [base_start.strftime("%H:%M"), base_end.strftime("%H:%M")]
        }
        visit_info[place2] = {
            "stay_duration": (base_end - base_start).seconds // 60,
            "available_time": [base_start.strftime("%H:%M"), base_end.strftime("%H:%M")]
        }

    # 计算最优路线
    origin_place = random.choice(base_places)
    best_route, travel_times = find_best_route(city_data, city, visit_info, origin_place)

    shuffled_keys = list(visit_info.keys())
    random.shuffle(shuffled_keys)

    shuffled_visit_info = {key: visit_info[key] for key in shuffled_keys}

    problem_data = {
        "city": city,
        "places": base_places,
        "max_visits": len(best_route),
        "origin_place": origin_place,
        "visit_info": shuffled_visit_info,
        "best_route": best_route,
        "travel_times": travel_times
    }

    return problem_data


def find_best_route(city_data, city, visit_info, start_place):
    sorted_places = list(visit_info.keys())

    best_route = []
    best_travel_times = []
    max_visits = 0

    def dfs(current_time, visited, route, travel_times, index):
        nonlocal best_route, max_visits, best_travel_times

        if len(visited) > max_visits:
            max_visits = len(visited)
            best_route = route[:]
            best_travel_times = travel_times[:]

        if index > len(sorted_places):
            return

        for i in range(index, len(sorted_places)):
            place = sorted_places[i]
            if place in visited:
                continue

            start_time_str, end_time_str = visit_info[place]["available_time"]
            available_start = datetime.datetime.strptime(start_time_str, "%H:%M")
            available_end = datetime.datetime.strptime(end_time_str, "%H:%M")
            stay_duration = datetime.timedelta(minutes=visit_info[place]["stay_duration"])

            if not route:  # 第一次访问
                if place == start_place:
                    arrival_time = datetime.datetime.strptime("09:00", "%H:%M")  # 固定起点时间
                    travel_duration = datetime.timedelta(minutes=0)
                else:
                    travel_duration = datetime.timedelta(minutes=get_travel_time(city_data, city, start_place, place))
                    arrival_time = datetime.datetime.strptime("09:00", "%H:%M") + travel_duration
            else:
                last_place = route[-1][0]
                travel_duration = datetime.timedelta(minutes=get_travel_time(city_data, city, last_place, place))
                arrival_time = current_time + travel_duration

            if arrival_time < available_start:
                arrival_time = available_start

            if arrival_time + stay_duration <= available_end:
                new_time = arrival_time + stay_duration
                dfs(new_time, visited | {place},
                    route + [(place, arrival_time.strftime("%H:%M"), new_time.strftime("%H:%M"))],
                    travel_times + [travel_duration.seconds // 60],
                    i + 1)

        # 继续搜索可能的其他路径
        dfs(current_time, visited, route, travel_times, index + 1)

    dfs(datetime.datetime.strptime("09:00", "%H:%M"), set(), [], [], 0)

    return best_route, best_travel_times


def generate_meeting_problem(meeting_data_config, config_df):
    with st.status("Generating meeting problems", expanded=True) as status:

        st.write("Loading data...")
        with open(meeting_data_config['cities_path'], "r", encoding="utf-8") as file:
            all_city_data = json.load(file)

        all_cities = list(all_city_data.keys())

        problems = []

        metadata = {
            "config": config_df.to_dict(orient="records"),
            "cities_data_path": meeting_data_config['cities_path']
        }


        st.write("Checking config...")
        for index, row in config_df.iterrows():
            num_places = int(row['num_places'])
            num_meetings = int(row['num_meetings'])
            num_conflicts = int(row['num_conflicts'])

            if not (1 <= num_places <= 10):
                raise ValueError(f"num_places must be between 1 and 10, but got {num_places}")
            if not (1 <= num_meetings):
                raise ValueError(f"num_meetings must above 1, but got {num_meetings}")
            if not (0 <= num_conflicts <= num_places // 2):
                raise ValueError(f"num_conflicts must be between 0 and num_places // 2, but got {num_conflicts}")

        st.write("Generating problems...")
        for index, row in config_df.iterrows():
            num_places = int(row['num_places'])
            num_meetings = int(row['num_meetings'])
            num_conflicts = int(row['num_conflicts'])

            st.write(f"Generating problem {index + 1} of {len(config_df)}, num_places: {num_places}")



            for i in range(num_meetings):
                city = random.choice(all_cities)

                city_data = {
                    city: all_city_data[city]
                }

                problem_data = {
                    "metadata" :
                        {
                            "city": city,
                            "num_places": num_places,
                            "num_conflicts": num_conflicts
                        },
                    'problem': generate_random_problem(city_data, city, num_places, num_conflicts)
                }

                problems.append(problem_data)

        st.write("Saving problems...")
        output_data = {
            "metadata": metadata,
            "problems": problems
        }

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(meeting_data_config['meeting_data_folder'], f"meeting_data_{current_time}.json")

        os.makedirs(meeting_data_config['meeting_data_folder'], exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4, ensure_ascii=False)

        status.update(
            label=f"Generated successfully! See {output_file}.",
            state="complete",
            expanded=False
        )

def get_commuting_time(com_data, start, end):
    for data in com_data:
        if data['start'] == start and data['end'] == end:
            return data['duration']

    raise ValueError(f"Cannot find commuting time between {start} and {end}.")

def generate_meeting_response_model(dataset_path, model_name, model_adapter, task_name,
                                    prompt_template, max_new_token, temperature, top_p):
    required_fields = ["{{original_position}}", "{{available_times}}", "{{commuting_time}}"]

    if not all(field in prompt_template for field in required_fields):
        raise ValueError(f"Prompt template must contain all required fields: {required_fields}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    metadata_data = data['metadata']
    cities_data_path = metadata_data['cities_data_path']

    data = data['problems']
    total_data_num = len(data)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "meeting", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"meeting_response_{current_time}.json")

    metadata = {
        "dataset_path": dataset_path,
        "cities_data_path": cities_data_path,
        "model_name": model_name,
        "model_adapter": model_adapter,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    with open(cities_data_path, "r", encoding="utf-8") as file:
        city_data = json.load(file)

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

        places = item['problem']['places']
        available_times = ""
        commuting_time = ""

        for place, info in item['problem']['visit_info'].items():
            available_times += f"You'd like to meet at {place} for {info['stay_duration']} minutes. Your friend will be here from {info['available_time'][0]} to {info['available_time'][1]}\n"

        for i in range(len(places)):
            for j in range(len(places)):
                if i == j:
                    continue  # 跳过同一个地点的组合
                start = places[i]
                end = places[j]
                commuting_time += f"It takes {get_commuting_time(city_data[item['problem']['city']], start, end)} minutes to travel from {start} to {end}. "

        prompt = prompt_template.replace("{{original_position}}", item['problem']['origin_place']) \
        .replace("{{available_times}}", available_times) \
        .replace("{{commuting_time}}", commuting_time)
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
            llm_output = llm_output[json_start:json_end]
            parsed_output = json.loads(llm_output)

            response = ({
                "metadata": item['metadata'],
                "success": True,
                "problem": item['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": parsed_output,
            })
        except Exception as e:
            response = {
                "metadata": item['metadata'],
                "success": False,
                "problem": item['problem'],
                "prompt": prompt,
                "origin_output": origin_output,
                "llm_output": llm_output,
            }
            print(e)

        responses.append(response)
        mybar.progress((data.index(item) + 1) / len(data), text=f"Processing {data.index(item) + 1} / {len(data)}")

        with open(save_file, "w", encoding="utf-8") as file:
            json.dump({"metadata": metadata, "responses": responses}, file, indent=4, ensure_ascii=False)

    st.success(f"Meeting response generation completed. See {save_file}.")

def generate_meeting_response_api(dataset_path, api_key, api_url, model_engine, task_name,
                                  prompt_template, max_new_token, temperature, top_p,
                                  if_parallel=False, parallel_num=4):

    required_fields = ["{{original_position}}", "{{available_times}}", "{{commuting_time}}"]
    for field in required_fields:
        if field not in prompt_template:
            st.error(f"Prompt template missing field: {field}")
            return

    with open(dataset_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    city_data = json.load(open(json_data["metadata"]["cities_data_path"], "r", encoding="utf-8"))
    all_data = json_data["problems"]
    total_data_num = len(all_data)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "meeting", "response")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"meeting_response_{current_time}.json")

    metadata = {
        "dataset_path": dataset_path,
        "cities_data_path": json_data["metadata"]["cities_data_path"],
        "api_url": api_url,
        "model_engine": model_engine,
        "total_data_num": total_data_num,
        "prompt_template": prompt_template,
        "max_new_token": max_new_token,
        "temperature": temperature,
        "top_p": top_p
    }

    client = OpenAI(api_key=api_key, base_url=api_url)

    with open(save_file, "w") as f:
        json.dump({"metadata": metadata, "responses": []}, f, indent=4, ensure_ascii=False)

    my_bar = st.progress(0)
    progress = 0

    def build_prompt(item):
        places = item['problem']['places']
        available_times = ""
        commuting_time = ""

        for place, info in item['problem']['visit_info'].items():
            available_times += f"You'd like to meet at {place} for {info['stay_duration']} minutes. Your friend will be here from {info['available_time'][0]} to {info['available_time'][1]}\n"

        for i in range(len(places)):
            for j in range(len(places)):
                if i != j:
                    start, end = places[i], places[j]
                    time = get_commuting_time(city_data[item['problem']['city']], start, end)
                    commuting_time += f"It takes {time} minutes to travel from {start} to {end}. "

        prompt = prompt_template.replace("{{original_position}}", item['problem']['origin_place']) \
            .replace("{{available_times}}", available_times) \
            .replace("{{commuting_time}}", commuting_time)

        return prompt

    def process_single(index):
        item = all_data[index]
        prompt = build_prompt(item)
        output_original = ""
        parsed_output = ""

        try:
            response = client.chat.completions.create(
                model=model_engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_token,
                temperature=temperature,
                top_p=top_p
            )
            llm_output = response.choices[0].message.content.strip().replace("\n", "")
            output_original = llm_output

            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}")
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in output.")
            parsed_output = json.loads(llm_output[json_start:json_end + 1])

            result = {
                "metadata": item["metadata"],
                "success": True,
                "problem": item["problem"],
                "prompt": prompt,
                "origin_output": output_original,
                "llm_output": parsed_output
            }

        except Exception as e:
            print(f"[ERROR index {index}]:", e)
            result = {
                "metadata": item["metadata"],
                "success": False,
                "problem": item["problem"],
                "prompt": prompt,
                "origin_output": output_original,
                "llm_output": ""
            }

        return index, result

    if if_parallel:
        with ThreadPoolExecutor(max_workers=parallel_num) as executor:
            for start in range(0, total_data_num, parallel_num):
                indices = list(range(start, min(start + parallel_num, total_data_num)))
                future_to_index = {executor.submit(process_single, i): i for i in indices}
                results = {}

                for future in as_completed(future_to_index):
                    idx, res = future.result()
                    results[idx] = res

                with open(save_file, "r") as f:
                    saved = json.load(f)

                for i in indices:
                    saved["responses"].append(results[i])
                    progress += 1
                    my_bar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(saved, f, indent=4, ensure_ascii=False)
    else:
        for i in range(total_data_num):
            _, res = process_single(i)

            with open(save_file, "r") as f:
                saved = json.load(f)

            saved["responses"].append(res)
            progress += 1
            my_bar.progress(progress / total_data_num, text=f"Processing {progress}/{total_data_num}")

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(saved, f, indent=4, ensure_ascii=False)

    st.success(f"Meeting response generation completed. See {save_file}")

def parse_time(time_str):
    """Convert time string to datetime object"""
    return datetime.datetime.strptime(time_str, "%H:%M")

def evaluate_response(response, cities_data):
    """Evaluate a single response"""
    if not response["success"]:
        return response

    city = response["metadata"]["city"]
    visit_info = response["problem"]["visit_info"]
    llm_output = response["llm_output"]
    best_route = response["problem"]["best_route"]
    origin_place = response["problem"]["origin_place"]

    # Initialize evaluation result
    result = {
        "metadata": response["metadata"],
        "success": True,
        "problem": response["problem"],
        "prompt": response["prompt"],
        "origin_output": response["origin_output"],
        "llm_output": response["llm_output"],
        "evaluation": {
            "passed": True,
            "errors": []
        }
    }

    # Check if places in llm_output are in visit_info
    for place in llm_output.keys():
        if place not in visit_info:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Invalid Place",
                "reason": f"Place {place} is not in visit_info"
            })
            # return result

    # Sort llm_output by time
    sorted_visits = sorted(llm_output.items(), key=lambda x: parse_time(x[1]["start"]))

    # If the first place is not origin_place, check travel time from origin_place to the first place
    first_place = sorted_visits[0][0]
    if first_place != origin_place:
        travel_time = get_travel_time(cities_data, city, origin_place, first_place)
        if travel_time is None:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Missing Travel Time",
                "reason": f"Cannot find travel time from {origin_place} to {first_place}"
            })
            # return result

        # first_start = parse_time(sorted_visits[0][1]["start"])
        # origin_available_time = visit_info[origin_place]["available_time"]
        # origin_available_end = parse_time(origin_available_time[1])
        #
        # if origin_available_end + datetime.timedelta(minutes=travel_time) > first_start:
        #     result["evaluation"]["passed"] = False
        #     result["evaluation"]["errors"].append({
        #         "type": "Time Conflict",
        #         "reason": f"Travel time from {origin_place} to {first_place} causes a time conflict"
        #     })
            # return result
        first_start = parse_time(sorted_visits[0][1]["start"])
        origin_to_first_travel_time = get_travel_time(cities_data, city, origin_place, first_place)
        if origin_to_first_travel_time is None:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Missing Travel Time",
                "reason": f"Cannot find travel time from {origin_place} to {first_place}"
            })
            # return result

        elif parse_time("9:00") + datetime.timedelta(minutes=origin_to_first_travel_time) > first_start:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Time Conflict",
                "reason": f"Travel time from {origin_place} to {first_place} causes a time conflict"
            })

    # Check if the number of places in llm_output matches the number in best_route
    if len(llm_output) < len(best_route):
        result["evaluation"]["passed"] = False
        result["evaluation"]["errors"].append({
            "type": "Suboptimal Solution",
            "reason": "The number of places in llm_output is less than in best_route, not an optimal solution"
        })
        # return result

    # Check visit times and travel times for each place
    for i in range(len(sorted_visits) - 1):
        current_place, current_time = sorted_visits[i]
        next_place, next_time = sorted_visits[i + 1]

        if current_place not in visit_info:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Invalid Place",
                "reason": f"Place {current_place} is not in visit_info"
            })
            continue

        # Check if the current place's visit time is within available_time
        current_start = parse_time(current_time["start"])
        current_end = parse_time(current_time["end"])
        available_time = visit_info[current_place]["available_time"]


        if current_start < parse_time(available_time[0]) or current_end > parse_time(available_time[1]):
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Invalid Visit Time",
                "reason": f"Visit time for {current_place} is not within available_time"
            })

        # Check travel time
        travel_time = get_travel_time(cities_data, city, current_place, next_place)
        if travel_time is None:
            result["evaluation"]["passed"] = False
            result["evaluation"]["errors"].append({
                "type": "Missing Travel Time",
                "reason": f"Cannot find travel time from {current_place} to {next_place}"
            })
        else:
            # Check if the next place's start time is reasonable
            next_start = parse_time(next_time["start"])
            if current_end + datetime.timedelta(minutes=travel_time) > next_start:
                result["evaluation"]["passed"] = False
                result["evaluation"]["errors"].append({
                    "type": "Time Conflict",
                    "reason": f"Travel time from {current_place} to {next_place} causes a time conflict"
                })

    return result

def evaluate_all_responses(response_path, task_name):
    """Evaluate all responses"""
    with open(response_path, 'r') as f:
        data = json.load(f)

    cities_data_path = data["metadata"]["cities_data_path"]
    with open(cities_data_path, 'r') as f:
        cities_data = json.load(f)

    results = []
    for response in data["responses"]:
        results.append(evaluate_response(response, cities_data))

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = os.path.join(os.getcwd(), "tasks", task_name, "planning", "meeting", "evaluation")
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"meeting_evaluate_{current_time}.json")

    output_result = {
        "metadata": data["metadata"],
        "results": results
    }
    with open(save_file, 'w') as f:
        json.dump(output_result, f, indent=4)

    st.success(f"Meeting response evaluation completed. See {save_file}.")
