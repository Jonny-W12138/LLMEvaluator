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
import ast

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
# 将根目录添加到sys.path
sys.path.append(root_dir)
import config as conf

ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']

def llm_fact_checking_judge_model(model_path, selected_dataset, task_name, prompt_template, selected_summary_file_path,
                    field_mapping, max_tokens, temperature, top_p, adapter_path=None):
    """
    Generate key facts using LLM model.
    """
    if (field_mapping["Field Type"] == "Ref Summary").sum() != 1 \
        or (field_mapping['Field Type'] == 'Transcript').sum() != 1:
        st.error("Field mapping must contain one 'Ref Summary' and one 'Transcript' field.")
        return

    if prompt_template.count('{{summary_to_judge}}') != 1:
        st.error("Prompt template must contain one '{{summary_to_judge}}' placeholder.")
        return

    if prompt_template.count('{{input_text}}') != 1:
        st.error("Prompt template must contain one '{{input_text}}' placeholder.")
        return

    results = []

    with open(selected_summary_file_path, 'r') as f:
        raw_summaries = json.load(f)
    summaries_to_judge = pd.DataFrame(raw_summaries["results"])

    total_data_num = summaries_to_judge.shape[0]

    my_bar = st.progress(0)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    success_num = 0

    for i in range(total_data_num):
        current_row = summaries_to_judge.iloc[i]
        prompt = prompt_template.replace("{{input_text}}", current_row["transcript"])

        prompt = prompt.replace("{{summary_to_judge}}", current_row["summary"])

        messages = [
            {"role": "user", "content": prompt}
        ]

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

        pred_labels, pred_types, pred_quote = [], [], []

        if_success = True

        try:
            start_idx = generated_text.find('[')

            if start_idx != -1:
                end_idx = generated_text.find(']')
                output = generated_text[start_idx:end_idx + 1]
                output = output.replace('\n', '')
                output = ast.literal_eval(output)

                for out in output:
                    category = out["category"]
                    category = category.replace('\n', '').replace('[', '').replace(']', '')
                    if category.lower() == "no error":
                        pred_labels.append(0)
                    else:
                        pred_labels.append(1)
                    pred_types.append(category)
                    pred_quote.append(out["quote"])

            else:
                start_idx = generated_text.find('{')
                end_idx = generated_text.find('}')
                output = generated_text[start_idx:end_idx + 1]
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
                pred_labels = []
                pred_types = []
                if_success = False

        result_data = {
            "record_index": i,
            "success": if_success,
            "transcript": current_row["transcript"],
            "summary": current_row["summary"],
            "model_output": generated_text,
            "pred_labels": pred_labels,
            "pred_types": pred_types,
            "pred_quote": pred_quote,
        }

        if if_success:
            success_num += 1

        results.append(result_data)

        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(os.getcwd(), "tasks", f"{task_name}", "summarization",
                               f"keyfact_check-{current_time}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    st.success(f"Results saved to {output_file}")


def llm_fact_checking_judge_api(task_name, selected_dataset, prompt_template, selected_summary_file_path, api_url, api_key, model_engine,
                       field_mapping, max_tokens, temperature, top_p):
    """
    Generate key facts using LLM model.
    """
    if (field_mapping["Field Type"] == "Ref Summary").sum() != 1 \
        or (field_mapping['Field Type'] == 'Transcript').sum() != 1:
        st.error("Field mapping must contain one 'Ref Summary' and one 'Transcript' field.")
        return

    if prompt_template.count('{{summary_to_judge}}') != 1:
        st.error("Prompt template must contain one '{{summary_to_judge}}' placeholder.")
        return

    if prompt_template.count('{{input_text}}') != 1:
        st.error("Prompt template must contain one '{{input_text}}' placeholder.")
        return

    results = []

    with open(selected_summary_file_path, 'r') as f:
        raw_summaries = json.load(f)
    summaries_to_judge = pd.DataFrame(raw_summaries["results"])

    total_data_num = summaries_to_judge.shape[0]

    my_bar = st.progress(0)

    client = OpenAI(api_key=api_key, base_url=api_url)

    success_num = 0

    for i in range(total_data_num):
        current_row = summaries_to_judge.iloc[i]
        prompt = prompt_template.replace("{{input_text}}", current_row["transcript"])

        prompt = prompt.replace("{{summary_to_judge}}", current_row["summary"])

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

        pred_labels, pred_types, pred_quote = [], [], []

        if_success = True

        try:
            start_idx = generated_text.find('[')

            if start_idx != -1:
                end_idx = generated_text.find(']')
                output = generated_text[start_idx:end_idx + 1]
                output = output.replace('\n', '')
                output = ast.literal_eval(output)


                for out in output:
                    category = out["category"]
                    category = category.replace('\n', '').replace('[', '').replace(']', '')
                    if category.lower() == "no error":
                        pred_labels.append(0)
                    else:
                        pred_labels.append(1)
                    pred_types.append(category)
                    pred_quote.append(out["quote"])

            else:
                start_idx = generated_text.find('{')
                end_idx = generated_text.find('}')
                output = generated_text[start_idx:end_idx + 1]
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
                pred_labels = []
                pred_types = []
                if_success = False

        result_data = {
            "record_index": i,
            "success": if_success,
            "transcript": current_row["transcript"],
            "summary": current_row["summary"],
            "model_output": generated_text,
            "pred_labels": pred_labels,
            "pred_types": pred_types,
            "pred_quote": pred_quote,
        }

        if if_success:
            success_num += 1

        results.append(result_data)

        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(os.getcwd(), "tasks", f"{task_name}", "summarization", f"keyfact_check-{current_time}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    st.success(f"Results saved to {output_file}")

def llm_fact_alignment_judge_model(model_path, selected_dataset, task_name, prompt_template, selected_summary_file_path,
                    field_mapping, max_tokens, temperature, top_p, adapter_path=None):
    """
    Generate key facts using LLM model.
    """
    if prompt_template.count('{{summary_to_judge}}') != 1:
        st.error("Prompt template must contain one '{{summary_to_judge}}' placeholder.")
        return

    if prompt_template.count('{{ref_keyfacts}}') != 1:
        st.error("Prompt template must contain one '{{ref_keyfacts}}' placeholder.")
        return

    with open(os.path.join(os.getcwd(), "dataset", "summarization", "data_config.json")) as f:
        datasets = json.load(f)
        if selected_dataset not in datasets:
            st.error("Selected dataset not found in data_config.json.")
            return
        if "keyfacts_path" not in datasets[selected_dataset]:
            st.error("Keyfacts not generated for the selected dataset.\n\n"
                     "Check dataset/summarization/data_config.json.")
            return

        keyfacts_data_path = datasets[selected_dataset]["keyfacts_path"]
    with open(keyfacts_data_path, 'r') as f:
        keyfacts_data = json.load(f)
    keyfacts = pd.DataFrame(keyfacts_data["results"])


    results = []

    with open(selected_summary_file_path, 'r') as f:
        raw_summaries = json.load(f)
    summaries_to_judge = pd.DataFrame(raw_summaries["results"])

    total_data_num = summaries_to_judge.shape[0]

    if keyfacts.shape[0] != total_data_num:
        st.error("Number of keyfacts and summaries do not match.\n\n"
                 f"Check {keyfacts_data_path} and {selected_summary_file_path}.")
        return

    my_bar = st.progress(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    success_num = 0


    for i in range(total_data_num):
        summary_current_row = summaries_to_judge.iloc[i]
        keyfacts_current_row = keyfacts.iloc[i]

        prompt = prompt_template.replace("{{summary_to_judge}}", summary_current_row["summary"])

        keyfacts_list = keyfacts_current_row["key_facts"]
        keyfacts_str = "\n".join(keyfacts_list)

        prompt = prompt.replace("{{ref_keyfacts}}", keyfacts_str)

        messages = [
            {"role": "user", "content": prompt}
        ]

        kwargs = {
            "text_inputs": messages,
            "max_new_tokens": max_tokens,
            "top_p": top_p
        }

        if temperature == 0.0:
            kwargs["do_sample"] = False
        else:
            kwargs["temperature"] = temperature

        # outputs = pipe(**kwargs)
        outputs = pipe(messages, max_new_tokens=max_tokens, top_p=top_p)
        generated_text = outputs[0]["generated_text"][-1]['content'].strip().replace("\n", "")

        output = generated_text

        if_success = True

        try:
            output = output.replace('```', '')
            start_idx = output.find('[')
            output = output[start_idx:]
            output = ast.literal_eval(output)

            matched_lines = set()
            pred_labels = []

            for out in output:
                category = out["response"]

                if category.lower() == "yes":
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

                if 'line number' in out:
                    line_nums = out["line number"]

                    for line_num in line_nums:
                        if type(line_num) is str:
                            line_num = line_num.replace('[', '').replace(']', '')
                        matched_lines.add(int(line_num))

            result = {
                "record_index": i,
                "success": if_success,
                "summary": summary_current_row["summary"],
                "keyfacts": keyfacts_current_row["key_facts"],
                "model_output": generated_text,
                "pred_labels": pred_labels,
                "matched_lines": list(matched_lines)
            }

        except Exception as e:
            print('parsing error:', e)
            result = {
                "record_index": i,
                "success": False,
                "summary": summary_current_row["summary"],
                "keyfacts": keyfacts_current_row["key_facts"],
                "model_output": generated_text,
                "pred_labels": [],
                "matched_lines": []
            }

        results.append(result)

        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

    output_result = {
        "metadata": {
            "model_source": "API",
            "model_engine": model_path,
            "total_data_num": total_data_num,
            "success_num": success_num,
            "failed_num": total_data_num - success_num,
            "selected_dataset": selected_dataset,
            "prompt_template": prompt_template,
            "field_mapping": field_mapping.to_dict(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        },
        "results": results
    }

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(os.getcwd(), "tasks", f"{task_name}", "summarization",
                               f"keyfact_alignment-{current_time}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    st.success(f"Results saved to {output_file}")


def llm_fact_alignment_judge_api(task_name, selected_dataset, prompt_template, selected_summary_file_path, api_url, api_key, model_engine,
                       field_mapping, max_tokens, temperature, top_p):
    if prompt_template.count('{{summary_to_judge}}') != 1:
        st.error("Prompt template must contain one '{{summary_to_judge}}' placeholder.")
        return

    if prompt_template.count('{{ref_keyfacts}}') != 1:
        st.error("Prompt template must contain one '{{ref_keyfacts}}' placeholder.")
        return

    with open(os.path.join(os.getcwd(), "dataset", "summarization", "data_config.json")) as f:
        datasets = json.load(f)
        if selected_dataset not in datasets:
            st.error("Selected dataset not found in data_config.json.")
            return
        if "keyfacts_path" not in datasets[selected_dataset]:
            st.error("Keyfacts not generated for the selected dataset.\n\n"
                     "Check dataset/summarization/data_config.json.")
            return

        keyfacts_data_path = datasets[selected_dataset]["keyfacts_path"]
    with open(keyfacts_data_path, 'r') as f:
        keyfacts_data = json.load(f)
    keyfacts = pd.DataFrame(keyfacts_data["results"])


    results = []

    with open(selected_summary_file_path, 'r') as f:
        raw_summaries = json.load(f)
    summaries_to_judge = pd.DataFrame(raw_summaries["results"])

    total_data_num = summaries_to_judge.shape[0]

    if keyfacts.shape[0] != total_data_num:
        st.error("Number of keyfacts and summaries do not match.\n\n"
                 f"Check {keyfacts_data_path} and {selected_summary_file_path}.")
        return

    my_bar = st.progress(0)

    client = OpenAI(api_key=api_key, base_url=api_url)

    success_num = 0


    for i in range(total_data_num):
        summary_current_row = summaries_to_judge.iloc[i]
        keyfacts_current_row = keyfacts.iloc[i]

        prompt = prompt_template.replace("{{summary_to_judge}}", summary_current_row["summary"])

        keyfacts_list = keyfacts_current_row["key_facts"]
        keyfacts_str = "\n".join(keyfacts_list)

        prompt = prompt.replace("{{ref_keyfacts}}", keyfacts_str)

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

        output = generated_text

        if_success = True

        try:
            output = output.replace('```', '')
            start_idx = output.find('[')
            output = output[start_idx:]
            output = ast.literal_eval(output)

            matched_lines = set()
            pred_labels = []

            for out in output:
                category = out["response"]

                if category.lower() == "yes":
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

                if 'line number' in out:
                    line_nums = out["line number"]

                    for line_num in line_nums:
                        if type(line_num) is str:
                            line_num = line_num.replace('[', '').replace(']', '')
                        matched_lines.add(int(line_num))

            result = {
                "record_index": i,
                "success": if_success,
                "summary": summary_current_row["summary"],
                "keyfacts": keyfacts_current_row["key_facts"],
                "model_output": generated_text,
                "pred_labels": pred_labels,
                "matched_lines": list(matched_lines)
            }

        except Exception as e:
            print('parsing error:', e)
            result = {
                "record_index": i,
                "success": False,
                "summary": summary_current_row["summary"],
                "keyfacts": keyfacts_current_row["key_facts"],
                "model_output": generated_text,
                "pred_labels": [],
                "matched_lines": []
            }

        results.append(result)

        my_bar.progress((i + 1) / total_data_num, text=f"Processing {i + 1}/{total_data_num}")

    output_data = {
        "metadata": {
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
        },
        "results": results
    }

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(os.getcwd(), "tasks", f"{task_name}", "summarization",
                               f"keyfact_alignment-{current_time}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    st.success(f"Results saved to {output_file}")