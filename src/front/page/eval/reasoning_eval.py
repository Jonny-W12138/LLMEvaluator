import subprocess
import streamlit as st
from streamlit import columns, session_state
import sys
import os

# 添加项目根目录到Python搜索路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_dir)

# 导入配置
from config import TASKS_DIR, DATASET_DIR

from src.eval.reasoning.response_generate import ReasoningResponseGenerator
from src.eval.reasoning.reasoning_evaluator import ReasoningEvaluator
from src.utils.model_loader import load_model
from src.utils.jsonl_validator import validate_and_fix_jsonl
import json
import pandas as pd
import time
from datetime import datetime

## st.header("Reasoning Evaluate")
# 替换标题，添加居中效果和样式
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="color: #4361ee; font-weight: 600;">Reasoning Evaluate</h1>
    <p style="color: #666; font-size: 1em;">Assess model reasoning capabilities with systematic benchmarks</p>
</div>
""", unsafe_allow_html=True)

# 1. 任务选择
with st.container(border=True):
    st.write("Task")
    task_type = st.segmented_control("task type", ["New task", "Existing task"], 
                                   default=st.session_state.get("task_type", "New task"), 
                                   label_visibility="collapsed")
    task = None

    if task_type == "New task":
        input_task = st.text_input("Task name", value=st.session_state.get("input_task", ""))
        if st.button("Create"):
            if input_task is None or input_task == "":
                st.error("Task name cannot be empty.")
            else:
                task_path = os.path.join(TASKS_DIR, input_task)
                if os.path.exists(task_path):
                    st.error(f"Task {input_task} already exists.")
                else:
                    os.makedirs(os.path.join(task_path, "reasoning"))
                    st.success(f"Task {input_task} created.")
                    task = input_task
                    if task is not None or task != "":
                        st.session_state["task"] = task
                        st.session_state["task_type"] = task_type
                        st.session_state["input_task"] = input_task
    else:
        # 获取任务列表时过滤掉非任务文件
        def get_task_list():
            tasks = []
            for item in os.listdir(TASKS_DIR):
                # 排除.gitkeep等隐藏文件和非目录项
                if not item.startswith('.') and os.path.isdir(os.path.join(TASKS_DIR, item)):
                    tasks.append(item)
            return tasks

        tasks = get_task_list()
        if tasks:  # 只有当有可用任务时才显示下拉菜单
            selected_task = st.selectbox("Select Existing Task", tasks, 
                                       index=tasks.index(st.session_state.get("selected_task", tasks[0])) if st.session_state.get("selected_task") in tasks else 0)
            if st.button("Select", key="select_task"):
                if selected_task is not None and selected_task != "":
                    st.session_state["task"] = selected_task
                    st.session_state["task_type"] = task_type
                    st.session_state["selected_task"] = selected_task
        else:
            st.info("No existing tasks found. Please create a new task first.")
            selected_task = None

# 2. 模型选择
with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model source")
        model_source = st.selectbox("Choose the source of the model",
                     ["huggingface/local", "API"],
                     index=0 if st.session_state.get("model_source") == "huggingface/local" else 1)

    with model_path_col:
        if model_source != "API":
            st.write("Model path")
            model_path = st.text_input(
                "Path to pretrained model or model identifier from Hugging Face",
                value=st.session_state.get("model_path", "")
            )

    if model_source == "API":
        api_url = st.text_input("API URL", value=st.session_state.get("api_url", ""))
        api_key = st.text_input("API key", type="password", value=st.session_state.get("api_key", ""))
        model_engine = st.text_input("Model engine", value=st.session_state.get("model_engine", ""))

# 3. 评估模型配置
with st.container(border=True):
    st.write("Evaluation Model")
    
    # 添加开关选项，控制是否启用评估模型
    use_eval_model = st.checkbox("Use Model for Step Evaluation", 
                                value=st.session_state.get("use_eval_model", False),
                                help="When enabled, uses the specified AI model to evaluate reasoning step similarity instead of the default algorithm")
    
    # 根据开关状态显示或隐藏评估模型配置
    if use_eval_model:
        eval_model_source = st.selectbox("Choose the source of the evaluation model",
                        ["huggingface/local", "API"], 
                        index=0 if st.session_state.get("eval_model_source") == "huggingface/local" else 1,
                        key="eval_model_source")

        if eval_model_source == "API":
            eval_api_url = st.text_input("Evaluation API URL", 
                                       value=st.session_state.get("eval_api_url", ""),
                                       key="eval_api_url")
            eval_api_key = st.text_input("Evaluation API key", 
                                       type="password", 
                                       value=st.session_state.get("eval_api_key", ""),
                                       key="eval_api_key")
            eval_model_engine = st.text_input("Evaluation Model engine", 
                                            value=st.session_state.get("eval_model_engine", ""),
                                            key="eval_model_engine")
        else:
            eval_model_path = st.text_input(
                "Path to evaluation model or model identifier from Hugging Face",
                value=st.session_state.get("eval_model_path", ""),
                key="eval_model_path"
            )
    else:
        # 如果不使用评估模型，清空相关会话状态
        for key in ["eval_model_source", "eval_api_url", "eval_api_key", "eval_model_engine", "eval_model_path"]:
            if key in st.session_state:
                del st.session_state[key]

# 4. 数据集选择
with st.container(border=True):
    st.write("Dataset")
    
    # 添加文件上传功能
    uploaded_file = st.file_uploader("Upload JSONL Format Dataset", 
                                   type=["jsonl"],
                                   key="uploaded_file")
    
    if uploaded_file is not None:
        try:
            # 读取上传的文件内容
            content = uploaded_file.getvalue().decode("utf-8")
            lines = content.strip().split("\n")
            
            # 验证JSONL格式
            preview_data = []
            for i, line in enumerate(lines[:5]):  # 只验证前5行
                try:
                    data = json.loads(line)
                    
                    # 验证必要字段
                    required_fields = ["qid", "contexts", "reference"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        st.error(f"Line {i+1} missing required fields: {', '.join(missing_fields)}")
                        break
                        
                    # 验证reference字段结构
                    if not isinstance(data["reference"], dict) or not {"answer", "process"}.issubset(data["reference"].keys()):
                        st.error(f"Line {i+1} has invalid reference field format, must contain answer and process fields")
                        break
                        
                    preview_data.append(data)
                except json.JSONDecodeError:
                    st.error(f"Line {i+1} is not a valid JSON format")
                    break
            
            # 如果验证通过，保存临时文件
            if len(preview_data) > 0:
                # 创建临时目录保存上传的文件
                temp_dir = os.path.join(DATASET_DIR, "reasoning", "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                
                # 生成唯一文件名
                temp_filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                # 保存文件
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                st.success(f"Dataset validation successful, {len(lines)} records in total")
                st.session_state["dataset_preview"] = preview_data
                st.session_state["dataset_path"] = temp_path
                st.session_state["selected_dataset"] = "user_uploaded"
                
                # 显示数据集预览
                st.write("Dataset Preview")
                st.dataframe(pd.DataFrame(preview_data))
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
                
    # 数据集格式说明部分
    with st.expander("Dataset Format Instructions"):
        st.markdown("<span style='font-size: 1.1em; font-weight: bold;'>Dataset Format Requirements</span>", unsafe_allow_html=True)
        st.markdown("The uploaded dataset must be in JSONL format (one JSON object per line). Each JSON object should contain the following fields:")
        st.markdown("""
        - `qid`: Question unique identifier
        - `contexts`: Question context (string or list of strings)
        - `reference`: Reference answer, containing the following sub-fields:
            - `answer`: Correct answer
            - `process`: Reasoning steps
        - `level`: Question difficulty level (integer, optional)
        - `examples`: Examples list (optional)
        - `category`: Question category (optional)
        """)
        st.markdown("<span style='font-size: 1.1em; font-weight: bold;'>Example Format</span>", unsafe_allow_html=True)
        st.code("""{"qid":"logic_123","contexts":["Question description"],"reference":{"answer":["0"],"process":["Step1","Step2"]},"level":0,"examples":["Example1"]}""", language="json")

# 4. 推理评估 - 修改字体大小
with st.container(border=True):
    st.write("Response Generation")  # 将subheader改为write，使字体大小一致
    
    # 使用两列布局，左侧为提示模板，右侧为参数设置
    col1, col2 = st.columns([3, 2])
    
    with col1:
        prompt_template = st.session_state.get("prompt_template", """\
The system has built-in default prompt template. 
If you need additional prompt words, please add them in this module.""")

        st.write("**Prompt Template**")
        prompt_template = st.text_area("", 
                                     value=prompt_template, 
                                     height=300, 
                                     label_visibility="collapsed",
                                     key="prompt_template")
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;">
            <strong>Note:</strong>
            <ul style="margin-top: 5px; margin-bottom: 5px;">
                <li>The output must be in <strong> JSON </strong> format</li>
                <li>The evaluation results will be saved in the <code>tasks</code> folder</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("**Generation Parameters**")
        max_tokens = st.slider("Max Tokens", 
                             min_value=100, 
                             max_value=2000, 
                             value=st.session_state.get("max_tokens", 400), 
                             step=100,
                             key="max_tokens")
        temperature = st.slider("Temperature", 
                              min_value=0.0, 
                              max_value=1.0, 
                              value=st.session_state.get("temperature", 0.0), 
                              step=0.1,
                              key="temperature")
        top_p = st.slider("Top P", 
                         min_value=0.1, 
                         max_value=1.0, 
                         value=st.session_state.get("top_p", 1.0), 
                         step=0.1,
                         key="top_p")
        
        # 参数说明使用相同的字体大小
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;">
            <strong>Parameter Description:</strong>
            <ul style="margin-top: 5px; margin-bottom: 5px;">
                <li><strong>Max Tokens</strong>: Controls the maximum length of generated text</li>
                <li><strong>Temperature</strong>: Controls randomness, lower values make output more deterministic</li>
                <li><strong>Top P</strong>: Controls the range of token selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 添加一些空间
    st.write("")
    
    # 生成评估按钮
    generate_btn = st.button("Generate & Evaluate", use_container_width=True)
    if generate_btn:
        if "dataset_path" not in st.session_state:
            st.error("Please select or upload a dataset first")
        elif "task" not in st.session_state:
            st.error("Please select a task first")
        else:
            with st.spinner("Generating evaluation results..."):
                # 保存当前参数到session state，但避免修改widget使用的键
                st.session_state["model_source"] = model_source
                if model_source != "API":
                    st.session_state["model_path"] = model_path
                else:
                    st.session_state["api_url"] = api_url
                    st.session_state["api_key"] = api_key
                    st.session_state["model_engine"] = model_engine
                
                # 不再直接修改widget使用的session state键值
                # 而是使用临时变量来存储评估模型配置
                eval_model_config = None
                if use_eval_model:
                    if eval_model_source == "API":
                        eval_model_config = {
                            "api_url": eval_api_url,
                            "api_key": eval_api_key,
                            "model_engine": eval_model_engine
                        }
                    else:
                        eval_model_config = {
                            "model_path": eval_model_path
                        }

                # 1. 加载模型
                if model_source == "API":
                    model = {
                        "api_url": api_url,
                        "api_key": api_key,
                        "model_engine": model_engine
                    }
                else:
                    model = load_model(model_path)

                # 2. 加载评估模型(如果启用)
                eval_model = None
                if use_eval_model and eval_model_config:
                    if eval_model_source == "API":
                        eval_model = {
                            "api_url": eval_model_config["api_url"],
                            "api_key": eval_model_config["api_key"],
                            "model_engine": eval_model_config["model_engine"]
                        }
                    else:
                        eval_model = load_model(eval_model_config["model_path"])

                # 3. 加载数据集
                dataset = []
                dataset_path = st.session_state["dataset_path"]
                
                # 先尝试验证和修复JSONL文件
                with st.spinner("Validating and fixing dataset file..."):
                    result = validate_and_fix_jsonl(dataset_path, verbose=False)
                    
                    if result["status"] == "error":
                        st.error(f"Dataset loading failed: {result.get('message', 'Unknown error')}")
                        st.stop()
                    
                    stats = result["stats"]
                    fixed_dataset_path = result["output_file"]
                    
                    # 显示验证结果
                    if stats["fixed_lines"] > 0 or stats["unfixable_lines"] > 0:
                        st.warning(
                            f"Dataset file needs repair: "
                            f"Total lines {stats['total_lines']}, "
                            f"Valid lines {stats['valid_lines']}, "
                            f"Fixed lines {stats['fixed_lines']}, "
                            f"Unfixable lines {stats['unfixable_lines']}"
                        )
                        
                        # 如果有无法修复的行，显示警告
                        if stats["unfixable_lines"] > 0:
                            st.error(f"Dataset has {stats['unfixable_lines']} lines that cannot be parsed, which may affect evaluation")
                    
                    # 加载修复后的文件
                    with open(fixed_dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:  # 跳过空行
                                dataset.append(json.loads(line))
                
                # 检查加载的数据
                if not dataset:
                    st.error("Dataset is empty or in incorrect format")
                    st.stop()
                else:
                    st.success(f"Successfully loaded {len(dataset)} data items")
                    
                    # 检查必要字段
                    required_fields = ["qid", "contexts", "reference"]
                    for item in dataset:
                        missing_fields = [field for field in required_fields if field not in item]
                        if missing_fields:
                            st.error(f"Data item {item.get('qid', 'unknown')} missing required fields: {', '.join(missing_fields)}")
                            st.stop()
                        
                        # 检查reference字段
                        if not isinstance(item["reference"], dict) or not {"answer", "process"}.issubset(item["reference"].keys()):
                            st.error(f"Data item {item.get('qid', 'unknown')} has incorrect reference field format, must contain answer and process fields")
                            st.stop()

                # 3. 执行评估
                evaluator = ReasoningEvaluator()
                results = evaluator.evaluate_batch(
                    model=model,
                    dataset=dataset,
                    task_name=st.session_state["task"],
                    eval_model=eval_model if use_eval_model else None  # 只在启用时传入评估模型
                )

                # 4. 显示结果摘要
                st.success("Evaluation complete!")
                st.write("Evaluation Results Summary:")
                
                # 显示评估方式
                if use_eval_model:
                    eval_method = f"Using model evaluation: {eval_model_engine if eval_model_source == 'API' else eval_model_path}"
                else:
                    eval_method = "Using default algorithm"
                st.info(f"Step evaluation method: {eval_method}")
                
                st.json({
                    "Total Samples": len(results),
                    "Answer Accuracy (A-Acc)": f"{sum(r['answer_accuracy'] for r in results) / len(results):.2%}",
                    "Step Accuracy (P-Acc)": f"{sum(r['step_accuracy'] for r in results) / len(results):.2%}",
                    "Overall Accuracy (AP-Acc)": f"{sum(r['ap_accuracy'] for r in results) / len(results):.2%}"
                })

                # 显示详细评估结果
                st.subheader("Detailed Evaluation Results")
                if 'results' in locals() and results:
                    # 创建可折叠部分显示每个样本的详细评估结果
                    for i, result in enumerate(results):
                        with st.expander(f"Sample {i+1}: {result.get('qid', 'unknown')} (Level: {result.get('level', 'unknown')})"):
                            # 显示原始问题
                            st.markdown("**Question:**")
                            if isinstance(dataset[i]["contexts"], list):
                                # 如果是列表，显示每个上下文
                                for j, context in enumerate(dataset[i]["contexts"]):
                                    st.text(f"Context {j+1}:\n{context}")
                            else:
                                st.text(dataset[i]["contexts"])
                            
                            # 显示示例（如果有）
                            if "examples" in dataset[i] and dataset[i]["examples"]:
                                st.markdown("**Examples:**")
                                for j, example in enumerate(dataset[i]["examples"]):
                                    st.text(f"Example {j+1}:\n{example}")
                            
                            # 显示模型响应
                            st.markdown("**Model Response:**")
                            st.text(result.get("model_response", "No response received"))
                            
                            # 尝试解析和美化显示JSON响应
                            try:
                                if isinstance(result.get("model_response"), str):
                                    # 尝试提取JSON部分
                                    json_start = result["model_response"].find("{")
                                    json_end = result["model_response"].rfind("}") + 1
                                    if json_start >= 0 and json_end > json_start:
                                        json_str = result["model_response"][json_start:json_end]
                                        parsed_json = json.loads(json_str)
                                        st.json(parsed_json)
                            except:
                                pass
                            
                            # 显示参考答案
                            st.markdown("**Reference Answer:**")
                            reference = dataset[i]["reference"]
                            # 将列表格式的answer和process转换为字符串
                            answer = reference["answer"][0] if isinstance(reference["answer"], list) else reference["answer"]
                            process = reference["process"][0] if isinstance(reference["process"], list) else reference["process"]
                            
                            # 使用json格式化显示，保持与Model Response一致的格式
                            formatted_reference = {
                                "answer": answer,
                                "process": process
                            }
                            st.json(formatted_reference)
                            
                            # 显示评估结果框
                            st.markdown("**Evaluation Results:**")
                            
                            # 显示答案评估结果
                            st.markdown("**Answer:**")
                            if result.get('answer_accuracy'):
                                st.write("✓ Correct")
                            else:
                                st.write("✗ Incorrect")
                            
                            # 检查模型是否提供了推理步骤
                            model_response = result.get("model_response", "")
                            has_steps = False
                            if isinstance(model_response, dict):
                                has_steps = bool(model_response.get("process"))
                            elif isinstance(model_response, str):
                                try:
                                    # 尝试解析JSON字符串
                                    json_start = model_response.find("{")
                                    json_end = model_response.rfind("}") + 1
                                    if json_start >= 0 and json_end > json_start:
                                        json_str = model_response[json_start:json_end]
                                        parsed_json = json.loads(json_str)
                                        has_steps = bool(parsed_json.get("process"))
                                except:
                                    pass
                            
                            if has_steps:
                                # 获取模型和参考答案的步骤
                                model_steps = ""
                                if isinstance(model_response, dict):
                                    model_steps = model_response.get("process", "")
                                elif isinstance(model_response, str):
                                    try:
                                        json_start = model_response.find("{")
                                        json_end = model_response.rfind("}") + 1
                                        if json_start >= 0 and json_end > json_start:
                                            json_str = model_response[json_start:json_end]
                                            parsed_json = json.loads(json_str)
                                            model_steps = parsed_json.get("process", "")
                                    except:
                                        pass
                                
                                # 计算步骤相似度
                                step_similarity = 0
                                if "step_similarity" in result:
                                    # 直接从结果中获取相似度
                                    step_similarity = result.get("step_similarity", 0)
                                else:
                                    # 回退到本地计算
                                    try:
                                        from src.eval.reasoning.step_checker import StepChecker
                                        checker = StepChecker()
                                        step_similarity = checker._calculate_step_similarity(model_steps, process)
                                    except Exception as e:
                                        st.error(f"Error calculating similarity: {e}")
                                
                                # 显示详细的步骤评估信息
                                st.write("**Reasoning Steps:**")
                                if result.get('step_accuracy'):
                                    st.write("✓ Correct")
                                    st.write(f"Process Similarity: {step_similarity:.2%}")
                                else:
                                    st.write("✗ Incorrect")
                                    st.write(f"Process Similarity: {step_similarity:.2%}")
                                
                            else:
                                st.write("**Reasoning Steps:** Not Provided")
                                
                            st.markdown("**Overall Assessment:**")
                            st.write(f"{'✓ Correct' if result.get('ap_accuracy') else '✗ Incorrect'}")
                            
                # 如果有错误，显示错误信息
                if "error" in result:
                    st.error(f"Evaluation Error: {result['error']}")

if st.session_state.get("evaluation_complete", False):
    st.success("Evaluation complete!")
    st.write("Evaluation Results Summary:")
    
    # 显示评估方式
    if st.session_state.get("use_eval_model", False):
        eval_method = f"Using model evaluation: {st.session_state.get('eval_model_engine') if st.session_state.get('eval_model_source') == 'API' else st.session_state.get('eval_model_path')}"
    else:
        eval_method = "Using default algorithm"
    st.info(f"Step evaluation method: {eval_method}")
    
    results = st.session_state.get("evaluation_results", [])
    st.json({
        "Total Samples": len(results),
        "Answer Accuracy (A-Acc)": f"{sum(r['answer_accuracy'] for r in results) / len(results):.2%}",
        "Step Accuracy (P-Acc)": f"{sum(r['step_accuracy'] for r in results) / len(results):.2%}",
        "Overall Accuracy (AP-Acc)": f"{sum(r['ap_accuracy'] for r in results) / len(results):.2%}"
    })

    # 显示详细评估结果
    st.subheader("Detailed Evaluation Results")
    if results:
        for i, result in enumerate(results):
            with st.expander(f"Sample {i+1}: {result.get('qid', 'unknown')} (Level: {result.get('level', 'unknown')})"):
                # 显示原始问题
                st.markdown("**Question:**")
                if isinstance(dataset[i]["contexts"], list):
                    # 如果是列表，显示每个上下文
                    for j, context in enumerate(dataset[i]["contexts"]):
                        st.text(f"Context {j+1}:\n{context}")
                else:
                    st.text(dataset[i]["contexts"])
                
                # 显示示例（如果有）
                if "examples" in dataset[i] and dataset[i]["examples"]:
                    st.markdown("**Examples:**")
                    for j, example in enumerate(dataset[i]["examples"]):
                        st.text(f"Example {j+1}:\n{example}")
                
                # 显示模型响应
                st.markdown("**Model Response:**")
                st.text(result.get("model_response", "No response received"))
                
                # 尝试解析和美化显示JSON响应
                try:
                    if isinstance(result.get("model_response"), str):
                        # 尝试提取JSON部分
                        json_start = result["model_response"].find("{")
                        json_end = result["model_response"].rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = result["model_response"][json_start:json_end]
                            parsed_json = json.loads(json_str)
                            st.json(parsed_json)
                except:
                    pass
                
                # 显示参考答案
                st.markdown("**Reference Answer:**")
                reference = dataset[i]["reference"]
                # 将列表格式的answer和process转换为字符串
                answer = reference["answer"][0] if isinstance(reference["answer"], list) else reference["answer"]
                process = reference["process"][0] if isinstance(reference["process"], list) else reference["process"]
                
                # 使用json格式化显示，保持与Model Response一致的格式
                formatted_reference = {
                    "answer": answer,
                    "process": process
                }
                st.json(formatted_reference)
                
                # 显示评估结果框
                st.markdown("**Evaluation Results:**")
                
                # 显示答案评估结果
                st.markdown("**Answer:**")
                if result.get('answer_accuracy'):
                    st.write("✓ Correct")
                else:
                    st.write("✗ Incorrect")
                
                # 检查模型是否提供了推理步骤
                model_response = result.get("model_response", "")
                has_steps = False
                if isinstance(model_response, dict):
                    has_steps = bool(model_response.get("process"))
                elif isinstance(model_response, str):
                    try:
                        # 尝试解析JSON字符串
                        json_start = model_response.find("{")
                        json_end = model_response.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = model_response[json_start:json_end]
                            parsed_json = json.loads(json_str)
                            has_steps = bool(parsed_json.get("process"))
                    except:
                        pass
                
                if has_steps:
                    # 获取模型和参考答案的步骤
                    model_steps = ""
                    if isinstance(model_response, dict):
                        model_steps = model_response.get("process", "")
                    elif isinstance(model_response, str):
                        try:
                            json_start = model_response.find("{")
                            json_end = model_response.rfind("}") + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = model_response[json_start:json_end]
                                parsed_json = json.loads(json_str)
                                model_steps = parsed_json.get("process", "")
                        except:
                            pass
                    
                    # 计算步骤相似度
                    step_similarity = 0
                    if "step_similarity" in result:
                        # 直接从结果中获取相似度
                        step_similarity = result.get("step_similarity", 0)
                    else:
                        # 回退到本地计算
                        try:
                            from src.eval.reasoning.step_checker import StepChecker
                            checker = StepChecker()
                            step_similarity = checker._calculate_step_similarity(model_steps, process)
                        except Exception as e:
                            st.error(f"Error calculating similarity: {e}")
                    
                    # 显示详细的步骤评估信息
                    st.write("**Reasoning Steps:**")
                    if result.get('step_accuracy'):
                        st.write("✓ Correct")
                        st.write(f"Process Similarity: {step_similarity:.2%}")
                    else:
                        st.write("✗ Incorrect")
                        st.write(f"Process Similarity: {step_similarity:.2%}")
                    
                else:
                    st.write("**Reasoning Steps:** Not Provided")
                    
                st.markdown("**Overall Assessment:**")
                st.write(f"{'✓ Correct' if result.get('ap_accuracy') else '✗ Incorrect'}")
                
                # 如果有错误，显示错误信息
                if "error" in result:
                    st.error(f"Evaluation Error: {result['error']}")
