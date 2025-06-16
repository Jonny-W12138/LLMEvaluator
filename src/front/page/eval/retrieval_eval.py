"""
Retrieval Evaluation Page
"""

import os
import json
import streamlit as st
import pandas as pd
import sys
from datetime import datetime
import random
from typing import List, Dict, Any, Optional

# Add project root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_dir)

# Import configuration
from config import TASKS_DIR, DATASET_DIR

from src.eval.retrieval.retrieval_evaluator import RetrievalEvaluator
from src.utils.dir_utils import ensure_retrieval_directories, get_current_task
from src.utils.model_loader import load_model

# Ensure directories exist
task_name = get_current_task()
ensure_retrieval_directories(task_name)

# Page title and description with styling
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="color: #4361ee; font-weight: 600;">Retrieval Evaluate</h1>
    <p style="color: #666; font-size: 1em;">Assess model information retrieval capabilities from long contexts</p>
</div>
""", unsafe_allow_html=True)

# Initialize selected_task variable
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None

# 1. Task Selection
with st.container(border=True):
    st.write("Task")
    task_type = st.segmented_control("task type", ["New task", "Existing task"], 
                                   default=st.session_state.get("task_type", "New task"), 
                                   label_visibility="collapsed")
    task = None
    selected_task = st.session_state.selected_task

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
                    os.makedirs(os.path.join(task_path, "retrieval"), exist_ok=True)
                    st.success(f"Task {input_task} created.")
                    # 设置任务但不自动刷新页面
                    task = input_task
                    selected_task = input_task
                    st.session_state.selected_task = input_task
                    st.session_state.task_type = task_type
                    st.session_state.input_task = input_task
    else:
        # Get task list filtering out non-task files
        def get_task_list():
            tasks = []
            for item in os.listdir(TASKS_DIR):
                # Exclude hidden files and non-directory items
                if not item.startswith('.') and os.path.isdir(os.path.join(TASKS_DIR, item)):
                    tasks.append(item)
            return tasks

        tasks = get_task_list()
        if tasks:  # Only show dropdown when tasks are available
            task_index = 0
            if st.session_state.selected_task in tasks:
                task_index = tasks.index(st.session_state.selected_task)
                
            selected_task = st.selectbox("Select existing task", tasks, 
                                       index=task_index)
            if st.button("Select", key="select_task"):
                if selected_task is not None and selected_task != "":
                    st.session_state.selected_task = selected_task
                    st.session_state.task_type = task_type
        else:
            st.info("No existing tasks found. Please create a new task first.")
            selected_task = None

# 2. Model Selection
with st.container(border=True):
    model_source_col, model_path_col = st.columns(2)
    with model_source_col:
        st.write("Model Source")
        model_source = st.selectbox("Select model source",
                     ["huggingface/local", "API"],
                     index=0 if st.session_state.get("model_source") == "huggingface/local" else 1)

    with model_path_col:
        if model_source != "API":
            st.write("Model Path")
            model_path = st.text_input(
                "Hugging Face model identifier or local model path",
                value=st.session_state.get("model_path", "")
            )

    if model_source == "API":
        api_url = st.text_input("API URL", value=st.session_state.get("api_url", ""))
        api_key = st.text_input("API key", type="password", value=st.session_state.get("api_key", ""))
        model_engine = st.text_input("Model engine", value=st.session_state.get("model_engine", ""))

# 3. Evaluation Model (Optional)
with st.container(border=True):
    st.write("Evaluation Model (Optional)")
    
    use_eval_model = st.checkbox("Use separate evaluation model", 
                               value=False,
                               help="For retrieval tasks, answer evaluation is typically straightforward (exact matching). A separate evaluation model is only needed for complex answer extraction or specialized evaluation scenarios.")
    
    if use_eval_model:
        eval_model_col1, eval_model_col2 = st.columns(2)
        
        with eval_model_col1:
            eval_model_source = st.selectbox("Evaluation Model Source",
                             ["huggingface/local", "API"],
                             index=0 if st.session_state.get("eval_model_source") == "huggingface/local" else 1,
                             key="eval_model_source")
            
        with eval_model_col2:
            if eval_model_source != "API":
                eval_model_path = st.text_input(
                    "Evaluation Model Path",
                    value=st.session_state.get("eval_model_path", ""),
                    key="eval_model_path"
                )
            else:
                eval_model_engine = st.text_input("Evaluation Model Engine", 
                                               value=st.session_state.get("eval_model_engine", ""),
                                               key="eval_model_engine")
        
        if eval_model_source == "API":
            eval_api_url = st.text_input("Evaluation API URL", 
                                      value=st.session_state.get("eval_api_url", ""),
                                      key="eval_api_url")
            eval_api_key = st.text_input("Evaluation API Key", 
                                      type="password",
                                      value=st.session_state.get("eval_api_key", ""),
                                      key="eval_api_key")
        
        # Custom evaluation extraction patterns
        st.write("Answer Extraction Patterns")
        
        use_custom_patterns = st.checkbox("Use custom answer extraction patterns", 
                                        value=False,
                                        help="Define custom regex patterns to extract answer options from model responses")
        
        if use_custom_patterns:
            custom_patterns = st.text_area("Custom extraction patterns (one per line, regex supported)",
                                        value=st.session_state.get("custom_patterns", 
                                        """(?:选择|选项|答案|回答|答案是|我选择|正确答案是|应该选择)[\s]*[:：]?[\s]*([ABCD])
([ABCD])[\s]*(?:是正确的|是正确答案)
^[\s]*([ABCD])[\s]*$
我认为答案是[\s]*([ABCD])"""),
                                        height=150,
                                        help="Each line will be treated as a separate regex pattern. The first capturing group should match the answer option (A/B/C/D).")
            
            st.info("The system will try these patterns in order until it finds a match. If no pattern matches, it will fallback to searching for standalone A/B/C/D in the response.")

# 4. Dataset Selection
with st.container(border=True):
    st.write("Dataset")
    
    # 初始化数据集变量
    dataset = None
    dataset_path = None
    dataset_source = "default"
    
    # File upload functionality
    uploaded_file = st.file_uploader("Upload custom dataset (JSON)", type=["json"])
    if uploaded_file is not None:
        try:
            # 创建临时目录
            temp_dir = os.path.join(DATASET_DIR, "retrieval", "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成带时间戳的临时文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = os.path.join(temp_dir, f"temp_dataset_{timestamp}.json")
            
            # 保存上传的文件
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 尝试加载数据集验证格式
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                if isinstance(dataset, list) and len(dataset) > 0:
                    st.success(f"Dataset uploaded successfully: {len(dataset)} questions")
                else:
                    st.warning("Uploaded file does not contain a valid dataset array")
            except json.JSONDecodeError:
                st.error("The uploaded file is not valid JSON")
                
            # 设置为自定义数据集
            dataset_path = temp_file
            dataset_source = "custom"
        except Exception as e:
            st.error(f"Error uploading dataset: {str(e)}")

    # 数据集上传部分
    with st.expander("Dataset Format Instructions", expanded=False):
        # 上传控件代码...
        
        # 添加格式说明
        st.markdown("""
        Your dataset should be a JSON file with the following structure:
        ```json
        {
            "items": [
                {
                    "item_id": "q1",
                    "question": "What is the capital of France?",
                    "options": {
                        "A": "London",
                        "B": "Paris",
                        "C": "Berlin",
                        "D": "Rome"
                    },
                    "ground_truth": "B",
                    "domain": "Geography",
                    "difficulty": "easy",
                    "length": "short"
                },
                ...
            ]
        }
        ```
        
        **Required fields**:
        - `item_id`: Unique identifier for the question
        - `question`: The question text
        - `ground_truth`: The correct answer (either option key or text)
        
        **Optional fields**:
        - `options`: Dictionary of answer options (for multiple choice)
        - `domain`: Domain/category of the question
        - `difficulty`: Difficulty level (easy, medium, hard)
        - `length`: Content length (short, medium, long)
        """)

# 5. Response Generation
with st.container(border=True):
    st.write("Response Generation")
    
    # Custom prompt template
    default_prompt = """
You are a professional assistant who needs to answer questions based on given context.
Please read the following context carefully and then answer the question. 
Only respond with the letter A, B, C, or D without explanation.

Context:
{context}

Question:
{question}

Options:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Your answer (just the option letter):
"""
    
    use_custom_prompt = st.checkbox("Use custom prompt template")
    
    if use_custom_prompt:
        custom_prompt = st.text_area("Custom prompt template", 
                                    value=default_prompt, 
                                    height=200,
                                    help="Make sure the template includes placeholders: {context}, {question}, {choice_A}, {choice_B}, {choice_C}, and {choice_D}")
    else:
        custom_prompt = default_prompt

    # Preview prompt (optional toggle)
    if st.checkbox("Preview processed prompt") and dataset is not None:
        st.write("**Sample prompt:**")
        sample = dataset[0]
        preview_prompt = custom_prompt.format(
            context=sample.get("context", "")[:200] + "...",
            question=sample.get("question", ""),
            choice_A=sample.get("choice_A", ""),
            choice_B=sample.get("choice_B", ""),
            choice_C=sample.get("choice_C", ""),
            choice_D=sample.get("choice_D", "")
        )
        st.code(preview_prompt)
    
    # Sampling options
    sample_count = st.number_input("Number of samples to evaluate (0 for all)", 
                                 min_value=0, 
                                 max_value=1000,
                                 value=st.session_state.get("sample_count", 0),
                                 help="Set to 0 to evaluate the entire dataset")
    
    if sample_count > 0:
        random_seed = st.number_input("Random seed", 
                                    min_value=1, 
                                    max_value=9999,
                                    value=st.session_state.get("random_seed", 42),
                                    help="Controls random sampling. Using the same seed ensures consistent sample selection across different evaluation runs, allowing for fair comparisons between models.")
    
    # Start evaluation button - full width default style
    if st.button("Generate & Evaluate", use_container_width=True):
        # 检查任务选择
        if task is None and selected_task is None:
            st.error("Please select or create a task first.")
        else:
            # 设置当前任务
            current_task = task if task else selected_task
            
            # 检查数据集
            if dataset_source == "default":
                # 使用默认数据集
                try:
                    dataset_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "retrieval", "default_retrieval_dataset")) 
                                   if f.endswith(".json")]
                    
                    if not dataset_files:
                        st.error("No default datasets found.")
                        st.stop()
                    
                    # 使用第一个默认数据集
                    default_dataset_path = os.path.join(DATASET_DIR, "retrieval", "default_retrieval_dataset", dataset_files[0])
                    
                    with open(default_dataset_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                        
                    st.info(f"Using default dataset: {os.path.basename(default_dataset_path)} with {len(dataset)} questions")
                    
                except Exception as e:
                    st.error(f"Error loading default dataset: {str(e)}")
                    st.stop()
                
            elif dataset_path and dataset is None:
                # 已有路径但尚未加载
                try:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                except Exception as e:
                    st.error(f"Error loading dataset from {dataset_path}: {str(e)}")
                    st.stop()
            
            # 最终检查
            if dataset is None or len(dataset) == 0:
                st.error("No valid dataset available for evaluation.")
                st.stop()
            
            # 继续进行评估...
            try:
                # 设置进度跟踪
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("初始化评估...")
                
                # 创建输出目录
                from src.utils.dir_utils import ensure_retrieval_directories
                ensure_retrieval_directories(current_task)
                output_dir = os.path.join(TASKS_DIR, current_task, "retrieval")
                
                # 保存配置用于跟踪
                st.session_state["task"] = current_task
                
                # 准备模型配置
                if model_source == "API":
                    model_config = {
                        "model_type": "api",
                        "api_url": api_url,
                        "api_key": api_key,
                        "model_engine": model_engine
                    }
                else:  # huggingface/local
                    model_config = {
                        "model_type": "huggingface",
                        "model_path": model_path
                    }
                
                # 准备自定义提取模式
                custom_extraction_patterns = None
                if use_eval_model and use_custom_patterns and custom_patterns:
                    custom_extraction_patterns = custom_patterns.strip().split("\n")
                
                # 定义进度回调
                def update_progress(progress, status):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                # 初始化评估器
                status_text.text("初始化模型...")
                try:
                    from src.eval.retrieval.retrieval_evaluator import RetrievalEvaluator
                    evaluator = RetrievalEvaluator(
                        model_config,
                        prompt_template=custom_prompt if use_custom_prompt else None,
                        custom_patterns=custom_extraction_patterns
                    )
                    status_text.text("模型初始化成功！")
                except Exception as e:
                    progress_bar.progress(1.0)
                    status_text.text("模型初始化失败！")
                    st.error(f"初始化模型失败: {str(e)}")
                    st.error("请检查您的模型设置并重试。")
                    with st.expander("详细错误"):
                        import traceback
                        st.code(traceback.format_exc())
                    st.stop()
                
                # 如果指定，对数据集进行采样
                if sample_count > 0 and sample_count < len(dataset):
                    random.seed(random_seed)
                    dataset = random.sample(dataset, sample_count)
                    status_text.text(f"使用数据集中的 {sample_count} 个样本（种子: {random_seed}）...")
                
                # 执行评估
                status_text.text("开始评估...")
                result = evaluator.evaluate(dataset, output_dir, progress_callback=update_progress)
                
                # 检查结果
                if "error" in result:
                    progress_bar.progress(1.0)
                    status_text.text(f"评估失败: {result['error']}")
                    st.error(f"评估失败: {result['error']}")
                    if "traceback" in result:
                        with st.expander("错误详情"):
                            st.code(result["traceback"])
                    st.stop()
                
                # 清除之前的输出信息
                status_text.empty()  # 清除状态文本
                
                # 只显示一次结果摘要
                st.markdown("**Evaluation Results Summary**")
                
                metrics = result.get("metrics", {})
                responses = result.get("responses", [])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Questions", f"{metrics.get('total', 0)}")
                
                with col2:
                    st.metric("Correct Answers", f"{metrics.get('correct', 0)}")
                
                with col3:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
                
                with col4:
                    st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
                
                # 简洁显示详细指标
                st.write("**Detailed Metrics:**")
                
                # 按难度显示
                by_difficulty = metrics.get("by_difficulty", {})
                if by_difficulty:
                    st.write("**Accuracy by Difficulty:**")
                    for level, data in by_difficulty.items():
                        st.write(f"- {level.capitalize()}: {data.get('accuracy', 0)*100:.1f}% ({data.get('correct', 0)}/{data.get('total', 0)})")
                
                # 按长度显示
                by_length = metrics.get("by_length", {})
                if by_length:
                    st.write("**Accuracy by Context Length:**")
                    for length, data in by_length.items():
                        st.write(f"- {length.capitalize()}: {data.get('accuracy', 0)*100:.1f}% ({data.get('correct', 0)}/{data.get('total', 0)})")
                
                # 使用可折叠的expander显示样本响应
                with st.expander("View Sample Responses", expanded=False):
                    # 限制只显示前5个样本
                    for i, response in enumerate(responses[:5]):
                        st.markdown(f"**Question {i+1}**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Question:** {response.get('question', '')}")
                            st.markdown("**Options:**")
                            for key, value in response.get("options", {}).items():
                                st.markdown(f"- {key}: {value}")
                            st.markdown(f"**Correct Answer:** {response.get('ground_truth', '')}")
                        
                        with col2:
                            st.markdown("**Model Response:**")
                            if "error" in response:
                                st.error(f"Error: {response['error']}")
                            else:
                                model_resp = response.get('model_response', '')
                                
                                # 检查并显示模型响应
                                if model_resp and model_resp.strip():
                                    st.text_area("Raw response", value=model_resp, 
                                                height=100, disabled=True, key=f"resp_{i}")
                                else:
                                    st.warning("No response received from model")
                                
                                extracted = response.get('extracted_answer', None)
                                is_correct = response.get('is_correct', False)
                                
                                if extracted:
                                    st.markdown(f"**Extracted Answer:** {extracted}")
                                else:
                                    st.markdown("**Extracted Answer:** Not extracted")
                                
                                if is_correct:
                                    st.markdown("**Result:** ✓ Correct")
                                else:
                                    st.markdown("**Result:** ✗ Incorrect")
                                
                                st.markdown(f"**Response Time:** {response.get('response_time', 0):.2f} seconds")
                        
                        st.markdown("---")  # 分隔线

                # 成功消息 - 只显示一次
                st.success("Evaluation completed!")
                try:
                    # 使用相对路径显示结果位置
                    st.markdown(f"Results saved to: **tasks/{current_task}/retrieval**")
                except:
                    # 回退到显示原始路径
                    st.markdown(f"Results saved to: {output_dir}")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
