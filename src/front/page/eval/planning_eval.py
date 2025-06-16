import streamlit as st
import sys
import os
import configparser
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import json
import pandas as pd
from src.eval.planning.natural_plan.trip import generate_trip_problem, generate_trip_response_model, generate_trip_response_api, evaluate_trip_response
from src.eval.planning.natural_plan.meeting import generate_meeting_problem, generate_meeting_response_model, generate_meeting_response_api, evaluate_all_responses
from src.eval.planning.natural_plan.calender import generate_calender_problem, generate_calender_response, evaluate_calender_response

config = configparser.ConfigParser()
config.read('src/front/page/eval/eval_config.ini')

st.header("Planning Evaluation")

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

with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface/local", "API"])

    if model_source == "huggingface/local":
        with model_path_col:
            st.write("Model path")
            model_path = st.text_input(
                "Path to pretrained model or model identifier from Hugging Face"
            )
        original_model_adapter_path = None
        original_model_use_adapter = st.toggle("Use adapter", value=False)
        if original_model_use_adapter:
            original_model_adapter_path = st.text_input("Adapter path", value="",
                                                        key="original_model_adapter_path")

    if model_source == "API":
        api_url = st.text_input("API URL")
        api_key = st.text_input("API key")
        model_engine = st.text_input("Model engine")


with st.container(border=True):
    st.write("Natrual-Plan benchmark")
    trip_tab, meeting_tab, calender_tab = st.tabs(["Trip", "Meeting", "Calender"])

    with trip_tab:
        st.write("Trip")

        if 'task' not in st.session_state:
            st.error("Please select a task first.")

        elif not os.path.exists(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json")):
            st.error("Please make sure the data_config.json is in the dataset/planning folder.")

        else:
            with open(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json"), "r") as f:
                data_config = json.load(f)
            trip_data_config = data_config["trip"]

            if_generate_new_trip = st.toggle("Generate new trip", value=False)
            if if_generate_new_trip:
                generate_config = pd.DataFrame(
                    [
                        {"num_cities": 5, "min_stay": 2, "max_stay": 7, "constraint_days": 1, "direct_flight_rate": 0.3, "trips_amount": 20},
                        {"num_cities": 5, "min_stay": 2, "max_stay": 7, "constraint_days": 3, "direct_flight_rate": 0.5, "trips_amount": 20},
                        {"num_cities": 8, "min_stay": 2, "max_stay": 7, "constraint_days": 1, "direct_flight_rate": 0.3, "trips_amount": 20},
                        {"num_cities": 8, "min_stay": 2, "max_stay": 7, "constraint_days": 2, "direct_flight_rate": 0.5, "trips_amount": 20},
                        {"num_cities": 8, "min_stay": 2, "max_stay": 7, "constraint_days": 3, "direct_flight_rate": 0.8, "trips_amount": 20},
                    ]
                )

                st.write("Generate config")
                edited_generate_config = st.data_editor(
                    generate_config,
                    num_rows="dynamic",
                    key="generate_config",
                )

                with st.expander("Generate config help"):
                    st.markdown(
                        """
                        - `num_cities`: number of cities in the trip (range from 3 to 50, 3 to 10 is recommended)
                        - `min_stay`: minimum stay days in each city (2 to 7 is recommended)
                        -  `max_stay`: maximum stay days in each city (2 to 7 is recommended)
                        -  `constraint_days`: days of the constraint (no more than num_cities)
                        -  `direct_flight_rate`: the rate of direct flight (0.3 to 0.5 is recommended)
                        -  `trips_amount`: total amount of trips to generate of this config
                        """
                    )

                if st.button("Generate", key="generate_trip"):
                    generate_trip_problem(trip_data_config, edited_generate_config)

            st.divider()

            st.write("Generate Response")

            trip_prompt_template = config.get("planning", "trip_prompt_template")
            prompt = st.text_area(
                "Prompt",
                trip_prompt_template.strip().replace("\\n", "\n"),
                height=200,
            )

            selected_dataset = st.selectbox("Dataset", [f for f in os.listdir(trip_data_config["trip_data_folder"]) if not f.startswith('.')])
            dataset_path = os.path.join(trip_data_config["trip_data_folder"], selected_dataset)
            if st.button("Preview dataset"):
                with open(dataset_path, "r") as f:
                    dataset = json.load(f)
                    metadata = dataset["metadata"]["config"]
                st.dataframe(pd.DataFrame(metadata))

            response_generate_args_col = st.columns(3)
            with response_generate_args_col[0]:
                max_tokens = st.number_input("Max tokens", value=600, min_value=1)
            with response_generate_args_col[1]:
                temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1)
            with response_generate_args_col[2]:
                top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1)

            if model_source == "API":
                if_parallel = st.toggle("Parallel", value=False, key="trip_if_parallel")
                parallel_num = 0
                if if_parallel:
                    parallel_num = st.number_input("Parallel num", value=4, min_value=1, step=1, key="trip_parallel_num")
            if st.button("Generate", key="generate_trip_response"):
                if 'task' not in st.session_state or st.session_state['task'] is None:
                    st.error("Please select a task first.")
                else:
                    if model_source == "API":
                        generate_trip_response_api(
                            dataset_path=dataset_path,
                            api_key=api_key,
                            api_url=api_url,
                            model_engine=model_engine,
                            if_parallel=if_parallel,
                            parallel_num=parallel_num,
                            task_name=st.session_state['task'],
                            prompt_template=prompt,
                            max_new_token=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                    else:
                        generate_trip_response_model(
                            dataset_path,
                            model_path,
                            original_model_adapter_path if original_model_use_adapter else None,
                            st.session_state['task'],
                            prompt,
                            max_tokens,
                            temperature,
                            top_p
                        )

        st.divider()

        st.write("Trip Evaluation")
        if 'task' not in st.session_state or st.session_state['task'] is None:
            st.error("Please select a task first.")
        elif not os.path.exists(os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "trip", "response")
):
            st.error(f"Please make sure the trip response of this task is generated. See {os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "trip", "response")}")
        else:
            trip_folder = os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "trip", "response")
            trip_files = os.listdir(trip_folder)
            selected_trip_file = st.selectbox("Select a trip file", trip_files)
            if selected_trip_file is not None:
                trip_file_path = os.path.join(trip_folder, selected_trip_file)
                # if st.button("Preview metadata", key="preview_trip_response"):
                if trip_file_path is not None:
                    with open(trip_file_path, "r") as f:
                        trip_response = json.load(f)
                    trip_response = trip_response["metadata"]
                    st.json(trip_response)

                if st.button("Evaluate", key="evaluate_trip_response"):
                    evaluate_trip_response(trip_file_path, st.session_state['task'])
                    st.success(f"Trip response {selected_trip_file} evaluated successfully.")

    with meeting_tab:
        st.write("Meeting")

        if not os.path.exists(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json")):
            st.error("Please make sure the data_config.json is in the dataset/planning folder.")

        if 'task' not in st.session_state:
            st.error("Please select a task first.")

        else:
            with open(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json"), "r") as f:
                data_config = json.load(f)
            meeting_data_config = data_config["meeting"]
            if_generate_new_meeting = st.toggle("Generate new meeting", value=False)
            if if_generate_new_meeting:
                generate_config = pd.DataFrame(
                    [
                        {"num_places": 5, "num_conflicts": 0,"num_meetings": 20},
                        {"num_places": 5, "num_conflicts": 1,"num_meetings": 20},
                        {"num_places": 8, "num_conflicts": 2,"num_meetings": 20},
                        {"num_places": 8, "num_conflicts": 3,"num_meetings": 20},
                    ]
                )

                st.write("Generate config")
                edited_generate_config = st.data_editor(
                    generate_config,
                    num_rows="dynamic",
                    key="generate_config",
                )

                with st.expander("Generate config help"):
                    st.markdown(
                        """
                        - `num_positions`: number of positions in the meeting (range from 2 to 10)
                        - `num_conflicts`: number of conflicts in the meeting (range from 0 to num_positions // 2)
                        - `num_meetings`: total amount of meetings to generate of this config
                        """
                    )


                if st.button("Generate", key="generate_meeting"):
                    generate_meeting_problem(meeting_data_config, edited_generate_config)

            st.divider()

            st.write("Generate Response")

            meeting_prompt_template = config.get("planning", "meeting_prompt_template")
            prompt = st.text_area(
                "Prompt",
                meeting_prompt_template.strip().replace("\\n", "\n"),
                height=400,
            )

            selected_dataset = st.selectbox("Dataset", [f for f in os.listdir(meeting_data_config["meeting_data_folder"]) if not f.startswith('.')])
            dataset_path = os.path.join(meeting_data_config["meeting_data_folder"], selected_dataset)

            if st.button("Preview dataset", key="preview_meeting_dataset"):
                with open(dataset_path, "r") as f:
                    dataset = json.load(f)
                    metadata = dataset["metadata"]['config']
                st.dataframe(pd.DataFrame(metadata))

            response_generate_args_col = st.columns(3)
            with response_generate_args_col[0]:
                max_tokens = st.number_input("Max tokens", value=1200, min_value=1, key="meeting_max_tokens")
            with response_generate_args_col[1]:
                temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1, key="meeting_temperature")
            with response_generate_args_col[2]:
                top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1, key="meeting_top_p")

            if model_source == "API":
                if_parallel = st.toggle("Parallel", value=False, key="meeting_if_parallel")
                parallel_num = 0
                if if_parallel:
                    parallel_num = st.number_input("Parallel num", value=4, min_value=1, step=1, key="meeting_parallel_num")
            if st.button("Generate", key="generate_meeting_response"):
                if 'task' not in st.session_state or st.session_state['task'] is None:
                    st.error("Please select a task first.")
                else:
                    if model_source == "API":
                        generate_meeting_response_api(
                            dataset_path=dataset_path,
                            api_key=api_key,
                            api_url=api_url,
                            if_parallel=if_parallel,
                            parallel_num=parallel_num,
                            model_engine=model_engine,
                            task_name=st.session_state['task'],
                            prompt_template=prompt,
                            max_new_token=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        pass
                    else:
                        generate_meeting_response_model(
                            dataset_path,
                            model_path,
                            original_model_adapter_path if original_model_use_adapter else None,
                            st.session_state['task'],
                            prompt,
                            max_tokens,
                            temperature,
                            top_p
                        )

            st.divider()

            st.write("Meeting Evaluation")

            if 'task' not in st.session_state or st.session_state['task'] is None:
                st.error("Please select a task first.")
            elif not os.path.exists(os.path.join(os.getcwd(), "tasks",
                                                 st.session_state['task'], "planning", "meeting", "response")):
                st.error(f"Please make sure the meeting response of this task is generated. See {os.path.join(os.getcwd(), 'tasks', st.session_state['task'], 'planning', 'meeting', 'response')}")
            else:
                meeting_folder = os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "meeting", "response")
                meeting_files = os.listdir(meeting_folder)
                selected_meeting_file = st.selectbox("Select a meeting file", meeting_files)
                if selected_meeting_file is not None:
                    meeting_file_path = os.path.join(meeting_folder, selected_meeting_file)
                    if st.button("Preview metadata", key="preview_meeting_response"):
                        with open(meeting_file_path, "r") as f:
                            meeting_response = json.load(f)
                        meeting_response = meeting_response["metadata"]
                        st.json(meeting_response)

                    if st.button("Evaluate", key="evaluate_meeting_response"):
                        evaluate_all_responses(meeting_file_path, st.session_state['task'])

    with calender_tab:
        st.write("Calender")

        if not os.path.exists(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json")):
            st.error("Please make sure the data_config.json is in the dataset/planning folder.")

        if 'task' not in st.session_state:
            st.error("Please select a task first.")

        else:
            with open(os.path.join(os.getcwd(), "dataset", "planning", "data_config.json"), "r") as f:
                data_config = json.load(f)
            calender_data_config = data_config["calender"]
            if_generate_new_calender = st.toggle("Generate new calender", value=False)
            if if_generate_new_calender:
                generate_config = pd.DataFrame(
                    [
                        {"num_people": 3, "duration_mins": 30, "total_days": 1, "busy_blocks": 3, "num_data": 20},
                        {"num_people": 3, "duration_mins": 60, "total_days": 3, "busy_blocks": 5, "num_data": 20},
                        {"num_people": 5, "duration_mins": 60, "total_days": 5, "busy_blocks": 10, "num_data": 20},
                        {"num_people": 5, "duration_mins": 60, "total_days": 6, "busy_blocks": 15, "num_data": 20},
                    ]
                )

                st.write("Generate config")
                edited_generate_config = st.data_editor(
                    generate_config,
                    column_config={
                        "num_people": st.column_config.NumberColumn(
                            "num_people",
                            min_value=2,
                            max_value=10,
                            step=1,
                            required=True,
                        ),
                        "duration_mins": st.column_config.SelectboxColumn(
                        "duration_mins",
                            options=[
                                "30",
                                "60",
                            ],
                            required=True,
                        ),
                        "total_days": st.column_config.NumberColumn(
                            "total_days",
                            min_value=1,
                            max_value=7,
                            step=1,
                            required=True,
                        ),
                        "busy_blocks": st.column_config.NumberColumn(
                            "busy_blocks",
                            min_value=1,
                            step=1,
                            required=True,
                        ),
                        "num_data": st.column_config.NumberColumn(
                            "num_data",
                            min_value=1,
                            step=1,
                            required=True,
                        ),
                    },
                    num_rows="dynamic",
                    key="generate_calender_config",
                )

                with st.expander("Generate config help"):
                    st.markdown(
                        """
                        - `num_people`: number of people in the meeting (range from 2 to 10)
                        - `duration_mins`: duration of the meeting (in minutes)
                        - `total_days`: total days of the meeting (range from 1 to 7)
                        - `busy_blocks`: number of busy blocks for each person in total days
                        - `num_data`: total amount of data to generate of this config
                        """
                    )

                if st.button("Generate", key="generate_calender"):
                    generate_calender_problem(calender_data_config, edited_generate_config)

            st.divider()

            st.write("Generate Response")

            calender_prompt_template = config.get("planning", "calender_prompt_template")
            prompt = st.text_area(
                "Prompt",
                calender_prompt_template.strip().replace("\\n", "\n"),
                height=300,
            )

            selected_dataset = st.selectbox("Dataset", [f for f in os.listdir(calender_data_config["calender_data_folder"]) if not f.startswith('.')])
            dataset_path = os.path.join(calender_data_config["calender_data_folder"], selected_dataset)

            if st.button("Preview dataset", key="preview_calender_dataset"):
                with open(dataset_path, "r") as f:
                    dataset = json.load(f)
                    metadata = dataset["metadata"]
                st.dataframe(pd.DataFrame(metadata))

            response_generate_args_col = st.columns(3)
            with response_generate_args_col[0]:
                max_tokens = st.number_input("Max tokens", value=600, min_value=1, key="calender_max_tokens")
            with response_generate_args_col[1]:
                temperature = st.number_input("Temperature", value=0.1, min_value=0.0, step=0.1, key="calender_temperature")
            with response_generate_args_col[2]:
                top_p = st.number_input("Top p", value=1.0, min_value=0.0, step=0.1, key="calender_top_p")

            if_parallel = st.toggle("Parallel", value=False, key="calender_parallel")
            parallel_num = 0
            if if_parallel:
                parallel_num = st.number_input("Parallel num", value=4, min_value=1, step=1, key="calender_parallel_num")

            if st.button("Generate", key="generate_calender_response"):
                if 'task' not in st.session_state or st.session_state['task'] is None:
                    st.error("Please select a task first.")
                else:
                    if model_source == "API":
                        generate_calender_response(
                            dataset_path=dataset_path,
                            call_method=model_source,
                            api_key=api_key,
                            api_url=api_url,
                            if_parallel=if_parallel,
                            parallel_num=parallel_num,
                            model_engine=model_engine,
                            task_name=st.session_state['task'],
                            prompt_template=prompt,
                            max_new_token=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                    else:
                        generate_calender_response(
                            dataset_path=dataset_path,
                            call_method=model_source,
                            model_name=model_path,
                            model_adapter=original_model_adapter_path if original_model_use_adapter else None,
                            task_name=st.session_state['task'],
                            prompt_template=prompt,
                            max_new_token=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )

            st.divider()

            st.write("Calender Evaluation")

            if 'task' not in st.session_state or st.session_state['task'] is None:
                st.error("Please select a task first.")
            elif not os.path.exists(os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "calender", "response")):
                st.error(f"Please make sure the calender response of this task is generated. See {os.path.join(os.getcwd(), 'tasks', st.session_state['task'], 'planning', 'calender', 'response')}")
            else:
                calender_folder = os.path.join(os.getcwd(), "tasks", st.session_state['task'], "planning", "calender", "response")
                calender_files = os.listdir(calender_folder)
                selected_calender_file = st.selectbox("Select a calender file", calender_files)
                if selected_calender_file is not None:
                    calender_file_path = os.path.join(calender_folder, selected_calender_file)
                    if st.button("Preview metadata", key="preview_calender_response"):
                        with open(calender_file_path, "r") as f:
                            calender_response = json.load(f)
                        calender_response = calender_response["metadata"]
                        st.json(calender_response)

                    if st.button("Evaluate", key="evaluate_calender_response"):
                        evaluate_calender_response(calender_file_path, st.session_state['task'])