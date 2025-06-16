import streamlit as st
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_dir)

from config import TASKS_DIR

# Set page title and description, consistent with evaluation page style
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="color: #4361ee; font-weight: 600;">Reasoning Evaluation Report</h1>
    <p style="color: #666; font-size: 1em;">Visualize and analyze model reasoning capability assessment results</p>
</div>
""", unsafe_allow_html=True)

# Function: Get task list
def get_task_list():
    tasks = []
    for item in os.listdir(TASKS_DIR):
        # Exclude hidden files and non-directory items
        if not item.startswith('.') and os.path.isdir(os.path.join(TASKS_DIR, item)):
            # Check if reasoning subdirectory exists
            if os.path.exists(os.path.join(TASKS_DIR, item, "reasoning")):
                tasks.append(item)
    return tasks

# Function: Get task evaluation result files
def get_task_result_files(task_name):
    task_dir = os.path.join(TASKS_DIR, task_name, "reasoning")
    files = {
        "answer_accuracy": [],
        "step_accuracy": [],
        "ap_accuracy": []
    }
    
    if not os.path.exists(task_dir):
        return files
    
    for file in os.listdir(task_dir):
        for metric in files.keys():
            if file.startswith(f"{metric}_results_") and file.endswith(".json"):
                files[metric].append({
                    "path": os.path.join(task_dir, file),
                    "timestamp": file.replace(f"{metric}_results_", "").replace(".json", "")
                })
    
    # Sort by timestamp
    for metric in files:
        files[metric] = sorted(files[metric], key=lambda x: x["timestamp"], reverse=True)
    
    return files

# Function: Load evaluation results
def load_results(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 提取模型和评估模型信息
            model_info = {}
            for key in ["model_type", "model_engine", "api_url", "model_path",
                       "eval_model_type", "eval_model_engine", "eval_api_url", "eval_model_path"]:
                if key in data:
                    model_info[key] = data[key]
                    
            return data, model_info
    except Exception as e:
        st.error(f"Failed to load result file: {e}")
        return None, {}

# Function: Generate bar chart
def create_bar_chart(data, title, y_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data.keys(), data.values(), color='#4361ee')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Add value labels on bars
    for i, v in enumerate(data.values()):
        ax.text(i, v + 0.03, f"{v:.2%}", ha='center', fontsize=10)
    return fig

# Function: Generate line chart (by difficulty level)
def create_line_chart_by_level(results, title):
    # Group by difficulty level
    levels = {}
    for item in results:
        level = item.get("level", 0)
        if level not in levels:
            levels[level] = []
        levels[level].append(float(item["score"]))
    
    # Calculate average accuracy for each level
    level_avg = {level: np.mean(scores) for level, scores in levels.items()}
    
    # Sort by difficulty
    sorted_levels = sorted(level_avg.items(), key=lambda x: x[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [str(l[0]) for l in sorted_levels]
    scores = [l[1] for l in sorted_levels]
    
    ax.plot(levels, scores, marker='o', linewidth=2, color='#4361ee')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, v in enumerate(scores):
        ax.text(i, v + 0.03, f"{v:.2%}", ha='center', fontsize=10)
    
    return fig

# Function: Generate pie chart (correct/incorrect ratio)
def create_pie_chart(correct_count, total_count, title):
    labels = ['Correct', 'Incorrect']
    sizes = [correct_count, total_count - correct_count]
    colors = ['#4CAF50', '#F44336']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_title(title, fontsize=14, pad=20)
    
    return fig

# Function: Create confusion matrix (answer vs. step accuracy)
def create_confusion_matrix(results_answer, results_step):
    # Create mapping from qid to results
    answer_map = {r["qid"]: r["score"] for r in results_answer}
    step_map = {r["qid"]: r["score"] for r in results_step}
    
    # Find common qids
    common_qids = set(answer_map.keys()).intersection(set(step_map.keys()))
    
    # Calculate confusion matrix
    confusion = {
        "Answer Correct & Steps Correct": 0,
        "Answer Correct & Steps Incorrect": 0,
        "Answer Incorrect & Steps Correct": 0,
        "Answer Incorrect & Steps Incorrect": 0
    }
    
    for qid in common_qids:
        answer_correct = answer_map[qid] > 0
        step_correct = step_map[qid] > 0
        
        if answer_correct and step_correct:
            confusion["Answer Correct & Steps Correct"] += 1
        elif answer_correct and not step_correct:
            confusion["Answer Correct & Steps Incorrect"] += 1
        elif not answer_correct and step_correct:
            confusion["Answer Incorrect & Steps Correct"] += 1
        else:
            confusion["Answer Incorrect & Steps Incorrect"] += 1
    
    total = sum(confusion.values())
    percentages = {k: v/total*100 for k, v in confusion.items()}
    
    # Create 2x2 matrix
    matrix = np.zeros((2, 2))
    matrix[0, 0] = confusion["Answer Correct & Steps Correct"]
    matrix[0, 1] = confusion["Answer Correct & Steps Incorrect"]
    matrix[1, 0] = confusion["Answer Incorrect & Steps Correct"]
    matrix[1, 1] = confusion["Answer Incorrect & Steps Incorrect"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                xticklabels=["Steps Correct", "Steps Incorrect"],
                yticklabels=["Answer Correct", "Answer Incorrect"])
    
    ax.set_title("Answer vs. Reasoning Steps Matrix", fontsize=14, pad=20)
    
    return fig, confusion, percentages

# Function: Convert matplotlib figure to base64 encoded image
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    return img_str

# Function: Export report as HTML
def export_report_html(task_name, timestamp, metrics, results, selected_task):
    # Extract model information from results
    model_info = st.session_state.get("model_info", {})
    
    # Generate charts for the report
    # 1. Bar chart for metrics
    metric_chart = create_bar_chart(metrics, "Evaluation Metrics Comparison", "Accuracy")
    metric_chart_base64 = fig_to_base64(metric_chart)
    
    # 2. Confusion matrix
    confusion_chart, confusion, percentages = create_confusion_matrix(
        results["answer_accuracy"]["results"],
        results["step_accuracy"]["results"]
    )
    confusion_chart_base64 = fig_to_base64(confusion_chart)
    
    # 3. Pie charts
    # Answer accuracy pie chart
    correct_count = sum(1 for r in results["answer_accuracy"]["results"] if r["score"] > 0)
    total_count = len(results["answer_accuracy"]["results"])
    answer_pie = create_pie_chart(correct_count, total_count, "Answer Accuracy Distribution")
    answer_pie_base64 = fig_to_base64(answer_pie)
    
    # Steps accuracy pie chart
    correct_count = sum(1 for r in results["step_accuracy"]["results"] if r["score"] > 0)
    total_count = len(results["step_accuracy"]["results"])
    steps_pie = create_pie_chart(correct_count, total_count, "Reasoning Steps Accuracy Distribution")
    steps_pie_base64 = fig_to_base64(steps_pie)
    
    # Overall accuracy pie chart
    correct_count = sum(1 for r in results["ap_accuracy"]["results"] if r["score"] > 0)
    total_count = len(results["ap_accuracy"]["results"])
    overall_pie = create_pie_chart(correct_count, total_count, "Overall Accuracy Distribution")
    overall_pie_base64 = fig_to_base64(overall_pie)
    
    # 4. Difficulty level analysis charts
    # Answer accuracy by difficulty level
    answer_level_chart = create_line_chart_by_level(
        results["answer_accuracy"]["results"], 
        "Answer Accuracy by Difficulty Level"
    )
    answer_level_chart_base64 = fig_to_base64(answer_level_chart)
    
    # Steps accuracy by difficulty level
    steps_level_chart = create_line_chart_by_level(
        results["step_accuracy"]["results"], 
        "Reasoning Steps Accuracy by Difficulty Level"
    )
    steps_level_chart_base64 = fig_to_base64(steps_level_chart)
    
    # Overall accuracy by difficulty level
    overall_level_chart = create_line_chart_by_level(
        results["ap_accuracy"]["results"], 
        "Overall Accuracy by Difficulty Level"
    )
    overall_level_chart_base64 = fig_to_base64(overall_level_chart)
    
    # Create HTML content with enhanced styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reasoning Evaluation Report - {task_name}</title>
        <style>
            /* Modern clean styling */
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                color: #333;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            header {{
                background-color: #4361ee;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
                margin-bottom: 20px;
            }}
            h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 600;
            }}
            h2 {{
                color: #4361ee;
                border-bottom: 2px solid #4361ee;
                padding-bottom: 10px;
                margin-top: 30px;
                font-size: 22px;
            }}
            h3 {{
                color: #555;
                font-size: 18px;
                margin-top: 20px;
            }}
            .summary-box {{
                background-color: #e7f0fd;
                border-left: 4px solid #4361ee;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #4361ee;
                margin: 10px 0;
            }}
            .metric-label {{
                color: #666;
                font-size: 14px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: 600;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .chart-title {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #555;
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 20px;
            }}
            .chart-grid-3 {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 12px;
                color: #666;
                padding: 20px;
                border-top: 1px solid #eee;
            }}
            .correct {{
                color: #4CAF50;
                font-weight: bold;
            }}
            .incorrect {{
                color: #F44336;
                font-weight: bold;
            }}
            .model-info {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
                border: 1px solid #ddd;
            }}
            .model-info-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .model-card {{
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .insight-box {{
                background-color: #e7f0fd;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
                border-left: 4px solid #4361ee;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Reasoning Evaluation Report</h1>
                <p>Comprehensive analysis of model reasoning capability assessment</p>
            </header>
            
            <div class="summary-box">
                <h3>Report Overview</h3>
                <p><strong>Task Name:</strong> {task_name}</p>
                <p><strong>Evaluation Time:</strong> {timestamp.replace("_", " ")}</p>
                <p><strong>Total Samples:</strong> {len(results['answer_accuracy']['results'])}</p>
            </div>
            
            <!-- Model Information Section -->
            <h2>Model Information</h2>
            <div class="model-info-grid">
                <div class="model-card">
                    <h3>Evaluated Model</h3>
                    {
                    ('<p><strong>Type:</strong> ' + ('API Model' if model_info.get('model_type') == 'API' else 'Local/Huggingface Model') + '</p>' +
                    '<p><strong>' + ('Model Engine' if model_info.get('model_type') == 'API' else 'Model Path') + ':</strong> ' + 
                    (model_info.get('model_engine') if model_info.get('model_type') == 'API' else model_info.get('model_path', 'Unknown')) + '</p>' +
                    ('<p><strong>API URL:</strong> ' + model_info.get("api_url", "Default OpenAI API") + '</p>' if model_info.get('model_type') == 'API' else ''))
                    if "model_type" in model_info else "<p>Model information not available</p>"
                    }
                </div>
                <div class="model-card">
                    <h3>Evaluation Model</h3>
                    {
                    ('<p><strong>Type:</strong> ' + ('API Model' if model_info.get('eval_model_type') == 'API' else 'Local/Huggingface Model') + '</p>' +
                    '<p><strong>' + ('Model Engine' if model_info.get('eval_model_type') == 'API' else 'Model Path') + ':</strong> ' + 
                    (model_info.get('eval_model_engine') if model_info.get('eval_model_type') == 'API' else model_info.get('eval_model_path', 'Unknown')) + '</p>' +
                    ('<p><strong>API URL:</strong> ' + model_info.get("eval_api_url", "Default OpenAI API") + '</p>' if model_info.get('eval_model_type') == 'API' else ''))
                    if "eval_model_type" in model_info else "<p>Using Default Algorithm for Evaluation (No Evaluation Model)</p>"
                    }
                </div>
            </div>
            
            <!-- Metrics Summary Section -->
            <h2>Evaluation Metrics Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Answer Accuracy (A-Acc)</div>
                    <div class="metric-value">{metrics['answer_accuracy']:.2%}</div>
                    <div>Correct answer rate</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Reasoning Steps Accuracy (P-Acc)</div>
                    <div class="metric-value">{metrics['step_accuracy']:.2%}</div>
                    <div>Correct reasoning process rate</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Overall Accuracy (AP-Acc)</div>
                    <div class="metric-value">{metrics['ap_accuracy']:.2%}</div>
                    <div>Both answer and reasoning correct</div>
                </div>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{metric_chart_base64}" alt="Metrics Comparison Chart" style="max-width:100%;">
            </div>
            
            <!-- Relationship Analysis Section -->
            <h2>Answer vs. Reasoning Steps Relationship</h2>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <img src="data:image/png;base64,{confusion_chart_base64}" alt="Confusion Matrix" style="max-width:100%;">
                </div>
                
                <div>
                    <h3>Relationship Distribution:</h3>
                    <ul>
                        {' '.join(f'<li><strong>{k}:</strong> {v:.1f}%</li>' for k, v in percentages.items())}
                    </ul>
                    
                    <!-- Analysis insights -->
                    <div class="insight-box">
                        <h3>Insights</h3>
                        {f'<p>The model may have issues providing proper reasoning despite giving correct answers. Consider improving reasoning capabilities.</p>' if percentages["Answer Correct & Steps Incorrect"] > 20 else ''}
                        {f'<p>The model seems to have correct reasoning logic but wrong final answers. This may indicate issues in calculation or conclusion derivation.</p>' if percentages["Answer Incorrect & Steps Correct"] > 10 else ''}
                    </div>
                </div>
            </div>
            
            <!-- Accuracy Distribution Section -->
            <h2>Accuracy Distribution</h2>
            
            <div class="chart-grid-3">
                <div class="chart-container">
                    <img src="data:image/png;base64,{answer_pie_base64}" alt="Answer Accuracy Pie Chart" style="max-width:100%;">
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{steps_pie_base64}" alt="Reasoning Steps Accuracy Pie Chart" style="max-width:100%;">
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{overall_pie_base64}" alt="Overall Accuracy Pie Chart" style="max-width:100%;">
                </div>
            </div>
            
            <!-- Difficulty Level Analysis Section -->
            <h2>Difficulty Level Analysis</h2>
            <p>These charts show how the model performance varies with the difficulty level of the questions.</p>
            
            <div class="chart-grid-3">
                <div class="chart-container">
                    <img src="data:image/png;base64,{answer_level_chart_base64}" alt="Answer Accuracy by Difficulty Level" style="max-width:100%;">
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{steps_level_chart_base64}" alt="Reasoning Steps Accuracy by Difficulty Level" style="max-width:100%;">
                </div>
                
                <div class="chart-container">
                    <img src="data:image/png;base64,{overall_level_chart_base64}" alt="Overall Accuracy by Difficulty Level" style="max-width:100%;">
                </div>
            </div>
            
            <!-- Detailed Results Section -->
            <h2>Detailed Evaluation Results</h2>
            
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Question ID</th>
                            <th>Difficulty</th>
                            <th>Category</th>
                            <th>Answer</th>
                            <th>Reasoning Steps</th>
                            <th>Overall</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Create mapping from qid to results
    answer_map = {r["qid"]: r for r in results['answer_accuracy']['results']}
    step_map = {r["qid"]: r for r in results['step_accuracy']['results']}
    ap_map = {r["qid"]: r for r in results['ap_accuracy']['results']}
    
    # Find common qids
    common_qids = set(answer_map.keys()).intersection(set(step_map.keys())).intersection(set(ap_map.keys()))
    
    # Add table rows
    for qid in common_qids:
        answer = answer_map[qid]
        step = step_map[qid]
        ap = ap_map[qid]
        
        html_content += f"""
                        <tr>
                            <td>{qid}</td>
                            <td>{answer['level']}</td>
                            <td>{answer.get('category', '')}</td>
                            <td class="{'correct' if answer['score'] else 'incorrect'}">{'✓ Correct' if answer['score'] else '✗ Incorrect'}</td>
                            <td class="{'correct' if step['score'] else 'incorrect'}">{'✓ Correct' if step['score'] else '✗ Incorrect'}</td>
                            <td class="{'correct' if ap['score'] else 'incorrect'}">{'✓ Correct' if ap['score'] else '✗ Incorrect'}</td>
                        </tr>
        """
    
    # Close table and document
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p>LLM Evaluator - Reasoning Assessment Tool</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Main interface layout
# 1. Task selection
with st.container(border=True):
    st.write("Task Selection")
    
    tasks = get_task_list()
    if not tasks:
        st.info("No evaluation tasks found. Please create and run an evaluation task first.")
    else:
        # Use index parameter instead of directly modifying session state
        default_index = 0
        if "report_task" in st.session_state and st.session_state.get("report_task") in tasks:
            default_index = tasks.index(st.session_state.get("report_task"))
        
        selected_task = st.selectbox("Task Name", tasks, 
                                    index=default_index,
                                    key="report_task")
        
        # Get evaluation result files for the selected task
        result_files = get_task_result_files(selected_task)
        
        # Check if evaluation results exist
        has_results = any(len(files) > 0 for files in result_files.values())
        
        if not has_results:
            st.warning(f"Task '{selected_task}' has no reasoning evaluation results. Please run an evaluation first.")
        else:
            # Select evaluation timestamp
            timestamps = []
            for metric, files in result_files.items():
                for file in files:
                    if file["timestamp"] not in timestamps:
                        timestamps.append(file["timestamp"])
            
            timestamps = sorted(timestamps, reverse=True)
            selected_timestamp = st.selectbox(
                "Evaluation Time", 
                timestamps,
                format_func=lambda x: x.replace("_", " ").replace("results", "")
            )
            
            # Load selected evaluation results
            tmp_results = {}
            model_info = {}  # 添加模型信息变量
            
            for metric, files in result_files.items():
                for file in files:
                    if file["timestamp"] == selected_timestamp:
                        result_data, file_model_info = load_results(file["path"])
                        if result_data:
                            tmp_results[metric] = result_data
                            # 合并模型信息
                            model_info.update(file_model_info)
            
            # Check if all three metrics were loaded
            if len(tmp_results) == 3:
                # Store results in session state WITHOUT using the widget key
                if "loaded_results" not in st.session_state:
                    st.session_state["loaded_results"] = {}
                if "result_metrics" not in st.session_state:
                    st.session_state["result_metrics"] = {}
                
                st.session_state["loaded_results"] = tmp_results
                st.session_state["result_metrics"] = {
                    "answer_accuracy": sum(r["score"] for r in tmp_results["answer_accuracy"]["results"]) / len(tmp_results["answer_accuracy"]["results"]),
                    "step_accuracy": sum(r["score"] for r in tmp_results["step_accuracy"]["results"]) / len(tmp_results["step_accuracy"]["results"]),
                    "ap_accuracy": sum(r["score"] for r in tmp_results["ap_accuracy"]["results"]) / len(tmp_results["ap_accuracy"]["results"])
                }
                st.session_state["model_info"] = model_info  # 保存模型信息
                st.success(f"Successfully loaded evaluation results for {selected_task}")

# 2. Results summary
if "loaded_results" in st.session_state and "result_metrics" in st.session_state:
    results = st.session_state["loaded_results"]
    metrics = st.session_state["result_metrics"]
    model_info = st.session_state.get("model_info", {})  # 获取模型信息
    
    # 首先显示模型信息
    with st.container(border=True):
        st.write("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Evaluated Model:**")
            if "model_type" in model_info:
                if model_info["model_type"] == "API":
                    st.write(f"Type: API Model")
                    st.write(f"Model Engine: {model_info.get('model_engine', 'Unknown')}")
                    st.write(f"API URL: {model_info.get('api_url', 'Default OpenAI API')}")
                else:
                    st.write(f"Type: Local/Huggingface Model")
                    st.write(f"Model Path: {model_info.get('model_path', 'Unknown')}")
            else:
                st.write("Model information not available")
        
        with col2:
            # 如果有评估模型信息，显示评估模型信息
            st.markdown("**Evaluation Model:**")
            if "eval_model_type" in model_info:
                if model_info["eval_model_type"] == "API":
                    st.write(f"Type: API Model")
                    st.write(f"Model Engine: {model_info.get('eval_model_engine', 'Unknown')}")
                    st.write(f"API URL: {model_info.get('eval_api_url', 'Default OpenAI API')}")
                else:
                    st.write(f"Type: Local/Huggingface Model")
                    st.write(f"Model Path: {model_info.get('eval_model_path', 'Unknown')}")
            else:
                st.write("Using Default Algorithm for Evaluation (No Evaluation Model)")
    
    # 然后显示评估结果摘要
    with st.container(border=True):
        st.write("Results Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Answer Accuracy (A-Acc)", f"{metrics['answer_accuracy']:.2%}")
        
        with col2:
            st.metric("Reasoning Steps Accuracy (P-Acc)", f"{metrics['step_accuracy']:.2%}")
        
        with col3:
            st.metric("Overall Accuracy (AP-Acc)", f"{metrics['ap_accuracy']:.2%}")
        
        # Display total samples
        st.info(f"Total evaluation samples: {len(results['answer_accuracy']['results'])}")
        
        # Create bar chart
        fig = create_bar_chart(metrics, "Evaluation Metrics Comparison", "Accuracy")
        st.pyplot(fig)

# 3. Detailed analysis
if "loaded_results" in st.session_state:
    results = st.session_state["loaded_results"]
    
    with st.container(border=True):
        st.write("Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Relationship Matrix", "Accuracy Distribution", "Difficulty Analysis", "Detailed Results"])
        
        with tab1:
            st.write("Answer vs. Reasoning Steps Relationship Analysis")
            
            # Create confusion matrix
            fig, confusion, percentages = create_confusion_matrix(
                results["answer_accuracy"]["results"],
                results["step_accuracy"]["results"]
            )
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.pyplot(fig)
            
            with col2:
                st.write("Distribution:")
                for k, v in percentages.items():
                    st.write(f"{k}: {v:.1f}%")
                
                # Analysis insights
                if percentages["Answer Correct & Steps Incorrect"] > 20:
                    st.info("The model may have issues providing proper reasoning despite giving correct answers. Consider improving reasoning capabilities.")
                
                if percentages["Answer Incorrect & Steps Correct"] > 10:
                    st.info("The model seems to have correct reasoning logic but wrong final answers. This may indicate issues in calculation or conclusion derivation.")
        
        with tab2:
            st.write("Accuracy Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Answer correct/incorrect ratio pie chart
                correct_count = sum(1 for r in results["answer_accuracy"]["results"] if r["score"] > 0)
                total_count = len(results["answer_accuracy"]["results"])
                fig = create_pie_chart(correct_count, total_count, "Answer Accuracy Distribution")
                st.pyplot(fig)
            
            with col2:
                # Steps correct/incorrect ratio pie chart
                correct_count = sum(1 for r in results["step_accuracy"]["results"] if r["score"] > 0)
                total_count = len(results["step_accuracy"]["results"])
                fig = create_pie_chart(correct_count, total_count, "Reasoning Steps Accuracy Distribution")
                st.pyplot(fig)
            
            with col3:
                # Overall correct/incorrect ratio pie chart
                correct_count = sum(1 for r in results["ap_accuracy"]["results"] if r["score"] > 0)
                total_count = len(results["ap_accuracy"]["results"])
                fig = create_pie_chart(correct_count, total_count, "Overall Accuracy Distribution")
                st.pyplot(fig)
        
        with tab3:
            st.write("Performance by Difficulty Level")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Answer accuracy by difficulty level
                fig = create_line_chart_by_level(
                    results["answer_accuracy"]["results"], 
                    "Answer Accuracy by Difficulty Level"
                )
                st.pyplot(fig)
                st.info("This chart shows how answer accuracy varies with question difficulty.")
                
            with col2:
                # Steps accuracy by difficulty level
                fig = create_line_chart_by_level(
                    results["step_accuracy"]["results"], 
                    "Steps Accuracy by Difficulty Level"
                )
                st.pyplot(fig)
                st.info("This chart shows how reasoning accuracy varies with question difficulty.")
                
            with col3:
                # Overall accuracy by difficulty level
                fig = create_line_chart_by_level(
                    results["ap_accuracy"]["results"], 
                    "Overall Accuracy by Difficulty Level"
                )
                st.pyplot(fig)
                st.info("This chart shows how overall performance varies with question difficulty.")
        
        with tab4:
            st.write("Detailed Evaluation Results")
            
            # Create detailed results table
            data = []
            
            # Create mapping from qid to results
            answer_map = {r["qid"]: r for r in results["answer_accuracy"]["results"]}
            step_map = {r["qid"]: r for r in results["step_accuracy"]["results"]}
            ap_map = {r["qid"]: r for r in results["ap_accuracy"]["results"]}
            
            # Find common qids
            common_qids = set(answer_map.keys()).intersection(set(step_map.keys())).intersection(set(ap_map.keys()))
            
            for qid in common_qids:
                data.append({
                    "Question ID": qid,
                    "Difficulty": answer_map[qid]["level"],
                    "Category": answer_map[qid].get("category", ""),
                    "Answer Correct": "✓" if answer_map[qid]["score"] else "✗",
                    "Steps Correct": "✓" if step_map[qid]["score"] else "✗",
                    "Overall Assessment": "✓" if ap_map[qid]["score"] else "✗"
                })
            
            # Convert to DataFrame and display
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

# 4. Report export
if "loaded_results" in st.session_state and "result_metrics" in st.session_state:
    results = st.session_state["loaded_results"]
    metrics = st.session_state["result_metrics"]
    
    with st.container(border=True):
        st.write("Report Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate HTML Report", use_container_width=True):
                timestamp = results["answer_accuracy"]["timestamp"]
                html_content = export_report_html(selected_task, timestamp, metrics, results, selected_task)
                
                # Create download link
                b64 = base64.b64encode(html_content.encode()).decode()
                file_name = f"{selected_task}_reasoning_report_{timestamp}.html"
                href = f'<a href="data:text/html;base64,{b64}" download="{file_name}">Click to download HTML report</a>'
                
                st.markdown(href, unsafe_allow_html=True)
                st.success("Report generated successfully! Click the link above to download.")
        
        with col2:
            if st.button("Export CSV Data", use_container_width=True):
                # Create detailed results table
                data = []
                
                # Create mapping from qid to results
                answer_map = {r["qid"]: r for r in results["answer_accuracy"]["results"]}
                step_map = {r["qid"]: r for r in results["step_accuracy"]["results"]}
                ap_map = {r["qid"]: r for r in results["ap_accuracy"]["results"]}
                
                # Find common qids
                common_qids = set(answer_map.keys()).intersection(set(step_map.keys())).intersection(set(ap_map.keys()))
                
                for qid in common_qids:
                    data.append({
                        "Question ID": qid,
                        "Difficulty": answer_map[qid]["level"],
                        "Category": answer_map[qid].get("category", ""),
                        "Answer Correct": 1 if answer_map[qid]["score"] else 0,
                        "Steps Correct": 1 if step_map[qid]["score"] else 0,
                        "Overall Assessment": 1 if ap_map[qid]["score"] else 0
                    })
                
                # Convert to DataFrame and export
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                
                timestamp = results["answer_accuracy"]["timestamp"]
                file_name = f"{selected_task}_reasoning_results_{timestamp}.csv"
                
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:text/csv;base64,{b64}" download="{file_name}">Click to download CSV data</a>'
                
                st.markdown(href, unsafe_allow_html=True)
                st.success("CSV data generated successfully! Click the link above to download.")

# 显示评估结果
if "loaded_results" in st.session_state and "result_metrics" in st.session_state:
    results = st.session_state["loaded_results"]
    metrics = st.session_state["result_metrics"]
    model_info = st.session_state.get("model_info", {})  # 获取模型信息
