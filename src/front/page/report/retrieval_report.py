"""
检索能力评估报告页面
"""

import os
import streamlit as st
import pandas as pd
import json
import glob
import sys
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import math
import re

# 添加项目根目录到Python搜索路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_dir)

from config import TASKS_DIR

from src.report.retrieval.retrieval_results_render import RetrievalResultsRenderer
from src.report.retrieval.retrieval_report_generate import RetrievalReportGenerator
from src.utils.dir_utils import ensure_retrieval_directories, get_current_task

# 设置页面标题和描述，与评估页面风格一致
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="color: #4361ee; font-weight: 600;">Retrieval Evaluation Report</h1>
    <p style="color: #666; font-size: 1em;">Visualize and analyze model information retrieval capability assessment results</p>
</div>
""", unsafe_allow_html=True)

# 辅助函数
def get_task_list():
    """获取任务列表"""
    tasks = []
    for item in os.listdir(TASKS_DIR):
        # 排除隐藏文件和非目录项
        if not item.startswith('.') and os.path.isdir(os.path.join(TASKS_DIR, item)):
            # 检查是否存在检索子目录
            if os.path.exists(os.path.join(TASKS_DIR, item, "retrieval")):
                tasks.append(item)
    return tasks

def load_retrieval_results(task_path: str) -> List[Dict[str, Any]]:
    """
    加载检索评估结果 - 支持多种文件命名格式
    """
    retrieval_dir = os.path.join(task_path, "retrieval")
    if not os.path.exists(retrieval_dir):
        st.error(f"检索目录不存在: {retrieval_dir}")
        return []
    
    # 移除调试输出
    # st.write("**Debug:** 检查目录内容")
    all_files = os.listdir(retrieval_dir)
    # st.json(all_files)
    
    # 查找所有JSON文件
    json_files = [f for f in all_files if f.endswith('.json') and os.path.isfile(os.path.join(retrieval_dir, f))]
    
    # 查找结果文件，支持两种格式:
    # 1. 指标文件 (metrics_*.json) + 响应文件 (responses_*.json)
    # 2. 合并结果文件 (retrieval_results_*.json)
    results = []
    
    # 处理第一种格式 (指标 + 响应文件)
    metrics_files = [f for f in json_files if f.startswith("metrics_")]
    for metrics_file in metrics_files:
        try:
            # 提取时间戳
            timestamp_str = metrics_file.replace("metrics_", "").replace(".json", "")
            
            # 查找对应的响应文件
            responses_file = os.path.join(retrieval_dir, f"responses_{timestamp_str}.json")
            if not os.path.exists(responses_file):
                continue
                
            # 处理文件...
            with open(os.path.join(retrieval_dir, metrics_file), 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                
            results.append({
                "timestamp": timestamp_str,
                "formatted_time": format_timestamp(timestamp_str),
                "metrics_file": os.path.join(retrieval_dir, metrics_file),
                "responses_file": responses_file,
                "total": metrics.get("total", 0),
                "accuracy": metrics.get("accuracy", 0) * 100,
                "result_format": "separate"
            })
        except Exception as e:
            st.error(f"Error loading metrics file {metrics_file}: {str(e)}")
    
    # 处理第二种格式 (合并结果文件)
    result_files = [f for f in json_files if f.startswith("retrieval_results_")]
    for result_file in result_files:
        try:
            # 提取时间戳
            timestamp_str = result_file.replace("retrieval_results_", "").replace(".json", "")
            
            # 加载结果文件
            file_path = os.path.join(retrieval_dir, result_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                response_data = json.load(f)
            
            # 检查文件结构并提取指标
            metrics = {}
            if "metrics" in response_data:
                # 如果结果文件包含metrics字段
                metrics = response_data["metrics"]
            else:
                # 计算基本指标
                responses = response_data.get("responses", [])
                correct = sum(1 for r in responses if r.get("is_correct", False))
                total = len(responses)
                metrics = {
                    "total": total,
                    "correct": correct,
                    "accuracy": correct / total if total > 0 else 0
                }
            
            # 添加到结果列表
            results.append({
                "timestamp": timestamp_str,
                "formatted_time": format_timestamp(timestamp_str),
                "metrics_file": file_path,  # 使用同一文件
                "responses_file": file_path,  # 使用同一文件
                "total": metrics.get("total", 0),
                "accuracy": metrics.get("accuracy", 0) * 100,
                "result_format": "combined"
            })
        except Exception as e:
            st.error(f"Error loading result file {result_file}: {str(e)}")
    
    # 按时间戳排序
    return sorted(results, key=lambda x: x["timestamp"], reverse=True)

def format_timestamp(timestamp_str):
    """格式化时间戳字符串为人类可读格式"""
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def create_difficulty_chart(responses, figsize=(5, 3)):
    """创建按难度分类的准确率图表"""
    # 按难度分组
    difficulty_groups = {}
    for resp in responses:
        difficulty = resp.get("difficulty", "unknown")
        if difficulty not in difficulty_groups:
            difficulty_groups[difficulty] = {"total": 0, "correct": 0}
        
        difficulty_groups[difficulty]["total"] += 1
        if resp.get("is_correct", False):
            difficulty_groups[difficulty]["correct"] += 1
    
    # 计算准确率
    difficulties = []
    accuracies = []
    counts = []
    for difficulty, stats in difficulty_groups.items():
        if stats["total"] > 0:
            difficulties.append(difficulty.capitalize())
            accuracies.append(stats["correct"] / stats["total"] * 100)
            counts.append(stats["total"])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(difficulties, accuracies, color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylim(0, max(100, max(accuracies) * 1.1 if accuracies else 100))
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('Accuracy by Difficulty Level', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def create_length_chart(responses, figsize=(5, 3)):
    """创建按内容长度的准确率图表"""
    # 按长度分组
    length_groups = {}
    for resp in responses:
        length = resp.get("length", "unknown")
        if length not in length_groups:
            length_groups[length] = {"total": 0, "correct": 0}
        
        length_groups[length]["total"] += 1
        if resp.get("is_correct", False):
            length_groups[length]["correct"] += 1
    
    # 计算准确率
    lengths = []
    accuracies = []
    counts = []
    for length, stats in sorted(length_groups.items()):
        if stats["total"] > 0:
            lengths.append(length.capitalize())
            accuracies.append(stats["correct"] / stats["total"] * 100)
            counts.append(stats["total"])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(lengths, accuracies, color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylim(0, max(100, max(accuracies) * 1.1 if accuracies else 100))
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('Accuracy by Content Length', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def create_domain_chart(responses):
    """创建按领域的准确率图表"""
    # 按领域分组
    domain_groups = {}
    for resp in responses:
        domain = resp.get("domain", "unknown")
        if domain not in domain_groups:
            domain_groups[domain] = {"total": 0, "correct": 0}
        
        domain_groups[domain]["total"] += 1
        if resp.get("is_correct", False):
            domain_groups[domain]["correct"] += 1
    
    # 计算准确率
    domains = []
    accuracies = []
    counts = []
    for domain, stats in sorted(domain_groups.items()):
        if stats["total"] > 0:
            domains.append(domain.capitalize())
            accuracies.append(stats["correct"] / stats["total"] * 100)
            counts.append(stats["total"])
    
    # 创建图表 - 尺寸减半
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(domains, accuracies, color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylim(0, max(100, max(accuracies) * 1.1 if accuracies else 100))
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('Accuracy by Domain', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def create_error_analysis(responses):
    """创建错误分析图表"""
    # 计算错误类型
    error_types = {
        "Wrong Answer": 0,
        "Format Error": 0,
        "Other": 0
    }
    
    for resp in responses:
        if not resp.get("is_correct", True):
            error_reason = resp.get("error_reason", "")
            if "format" in error_reason.lower():
                error_types["Format Error"] += 1
            elif error_reason:
                error_types["Wrong Answer"] += 1
            else:
                error_types["Other"] += 1
    
    # 只保留非零类型
    labels = []
    values = []
    for label, value in error_types.items():
        if value > 0:
            labels.append(label)
            values.append(value)
    
    # 如果没有错误数据，添加一个占位符
    if not labels:
        labels = ["No Error Data"]
        values = [1]
    
    # 创建图表 - 尺寸减半
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    ax.axis('equal')  # 确保绘制为圆形
    ax.set_title('Error Type Distribution', fontsize=11)
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def create_accuracy_chart(metrics, figsize=(5, 3)):
    """创建总体准确率条形图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准确率数据
    accuracy = metrics.get('accuracy', 0) * 100
    
    # 创建条形图
    bars = ax.bar(["Accuracy"], [accuracy], color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 设置轴和标签
    ax.set_ylim(0, max(100, accuracy * 1.1))
    ax.set_ylabel('Percentage (%)', fontsize=9)
    ax.set_title('Overall Accuracy', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def create_pie_chart(correct, total, title="Response Distribution", figsize=(5, 3)):
    """创建正确/错误分布饼图"""
    incorrect = total - correct
    
    # 创建饼图
    fig, ax = plt.subplots(figsize=figsize)
    labels = ['Correct', 'Incorrect']
    sizes = [correct, incorrect]
    colors = ['#4CAF50', '#F44336']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
           startangle=90, textprops={'fontsize': 9})
    
    # 添加标题并确保圆形展示
    ax.set_title(title, fontsize=11)
    ax.axis('equal')
    
    # 增加紧凑性
    plt.tight_layout()
    
    return fig

def get_length_chart_base64(responses):
    """生成按长度分类的准确率图表"""
    # 按长度分组
    length_groups = {}
    for resp in responses:
        length = resp.get("length", "unknown")
        if length not in length_groups:
            length_groups[length] = {"total": 0, "correct": 0}
        
        length_groups[length]["total"] += 1
        if resp.get("is_correct", False):
            length_groups[length]["correct"] += 1
    
    # 计算准确率
    lengths = []
    accuracies = []
    for length, stats in length_groups.items():
        if stats["total"] > 0:
            lengths.append(length.capitalize())
            accuracies.append(stats["correct"] / stats["total"] * 100)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(lengths, accuracies, color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Context Length')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return get_chart_base64(fig)

def get_domain_chart_base64(responses):
    """生成按领域分类的准确率图表"""
    # 按领域分组
    domain_groups = {}
    for resp in responses:
        domain = resp.get("domain", "unknown")
        if domain not in domain_groups:
            domain_groups[domain] = {"total": 0, "correct": 0}
        
        domain_groups[domain]["total"] += 1
        if resp.get("is_correct", False):
            domain_groups[domain]["correct"] += 1
    
    # 计算准确率
    domains = []
    accuracies = []
    for domain, stats in domain_groups.items():
        if stats["total"] > 0:
            domains.append(domain)
            accuracies.append(stats["correct"] / stats["total"] * 100)
    
    # 按准确率排序
    sorted_data = sorted(zip(domains, accuracies), key=lambda x: x[1], reverse=True)
    domains = [x[0] for x in sorted_data]
    accuracies = [x[1] for x in sorted_data]
    
    # 如果领域太多，只显示前10个
    if len(domains) > 10:
        domains = domains[:10]
        accuracies = accuracies[:10]
    
    # 创建水平条形图
    fig, ax = plt.subplots(figsize=(10, max(6, len(domains) * 0.5)))
    bars = ax.barh(domains, accuracies, color='#4361ee')
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%', ha='left', va='center')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Accuracy by Domain')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    return get_chart_base64(fig)

def get_error_chart_base64(responses):
    """生成错误类型分析饼图"""
    # 统计错误类型
    error_types = {
        "Not Extracted": 0,
        "Wrong Answer": 0
    }
    
    for resp in responses:
        if not resp.get("is_correct", False):
            if resp.get("extracted_answer") is None:
                error_types["Not Extracted"] += 1
            else:
                error_types["Wrong Answer"] += 1
    
    # 只有当有错误时才创建图表
    if sum(error_types.values()) == 0:
        # 创建一个简单的消息图表
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No errors found!", ha='center', va='center', fontsize=16)
        ax.axis('off')
        return get_chart_base64(fig)
    
    # 创建饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        error_types.values(),
        labels=error_types.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=['#FF9800', '#F44336']
    )
    
    # 设置字体大小
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color('white')
    
    ax.axis('equal')  # 相等的纵横比确保饼图绘制为圆形
    ax.set_title('Error Type Distribution', fontsize=14, pad=20)
    
    return get_chart_base64(fig)

def generate_examples_html(responses):
    """为HTML报告生成样例响应HTML"""
    examples_html = ""
    
    for i, response in enumerate(responses):
        is_correct = response.get("is_correct", False)
        result_class = "correct" if is_correct else "incorrect"
        result_text = "✓ Correct" if is_correct else "✗ Incorrect"
        
        options_html = ""
        for key, value in response.get("options", {}).items():
            options_html += f"<li><strong>{key}</strong>: {value}</li>"
        
        examples_html += f"""
        <div class="example">
            <h3>Example {i+1}</h3>
            <p><strong>Question:</strong> {response.get("question", "")}</p>
            <p><strong>Domain:</strong> {response.get("domain", "")}</p>
            <p><strong>Difficulty:</strong> {response.get("difficulty", "").capitalize()}</p>
            <p><strong>Context Length:</strong> {response.get("length", "").capitalize()}</p>
            
            <div class="options">
                <p><strong>Options:</strong></p>
                <ul>
                    {options_html}
                </ul>
            </div>
            
            <p><strong>Correct Answer:</strong> {response.get("ground_truth", "")}</p>
            <p><strong>Model Response:</strong> {response.get("model_response", "")[:200]}...</p>
            <p><strong>Extracted Answer:</strong> {response.get("extracted_answer", "Not extracted")}</p>
            <p><strong>Result:</strong> <span class="{result_class}">{result_text}</span></p>
            <p><strong>Response Time:</strong> {response.get("response_time", 0):.2f} seconds</p>
        </div>
        """
    
    return examples_html

def export_report_html(task_name, timestamp, metrics, responses, model_info):
    """生成简化版HTML报告，修复图表尺寸一致性问题"""
    # 获取基本指标
    total = metrics.get("total", 0)
    correct = metrics.get("correct", 0)
    accuracy = metrics.get("accuracy", 0) * 100
    avg_time = metrics.get("avg_response_time", 0)
    
    # 收集数据用于图表
    # 处理难度数据
    difficulty_stats = {}
    length_stats = {}
    domain_stats = {}
    error_types = {"Wrong Answer": 0, "Format Error": 0, "Other": 0}
    
    for resp in responses:
        # 难度统计
        difficulty = resp.get("difficulty", "unknown")
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {"total": 0, "correct": 0}
        difficulty_stats[difficulty]["total"] += 1
        if resp.get("is_correct", False):
            difficulty_stats[difficulty]["correct"] += 1
        
        # 长度统计
        length = resp.get("length", "unknown")
        if length not in length_stats:
            length_stats[length] = {"total": 0, "correct": 0}
        length_stats[length]["total"] += 1
        if resp.get("is_correct", False):
            length_stats[length]["correct"] += 1
        
        # 领域统计
        domain = resp.get("domain", "unknown")
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "correct": 0}
        domain_stats[domain]["total"] += 1
        if resp.get("is_correct", False):
            domain_stats[domain]["correct"] += 1
        
        # 错误类型统计
        if not resp.get("is_correct", True):
            error_reason = resp.get("error_reason", "")
            if "format" in error_reason.lower():
                error_types["Format Error"] += 1
            elif error_reason:
                error_types["Wrong Answer"] += 1
            else:
                error_types["Other"] += 1
    
    # 找出所有图表的最大值，用于统一Y轴
    max_accuracy = 0
    
    # 准备各个图表的数据
    difficulty_labels = []
    difficulty_data = []
    
    for difficulty, stats in sorted(difficulty_stats.items()):
        if stats["total"] > 0:
            difficulty_labels.append(f'"{difficulty.capitalize()}"')
            accuracy_val = stats["correct"] / stats["total"] * 100
            difficulty_data.append(f'{accuracy_val:.1f}')
            max_accuracy = max(max_accuracy, accuracy_val)
    
    length_labels = []
    length_data = []
    
    for length, stats in sorted(length_stats.items()):
        if stats["total"] > 0:
            length_labels.append(f'"{length.capitalize()}"')
            accuracy_val = stats["correct"] / stats["total"] * 100
            length_data.append(f'{accuracy_val:.1f}')
            max_accuracy = max(max_accuracy, accuracy_val)
    
    domain_labels = []
    domain_data = []
    
    for domain, stats in sorted(domain_stats.items()):
        if stats["total"] > 0:
            domain_labels.append(f'"{domain.capitalize()}"')
            accuracy_val = stats["correct"] / stats["total"] * 100
            domain_data.append(f'{accuracy_val:.1f}')
            max_accuracy = max(max_accuracy, accuracy_val)
    
    # 计算Y轴的最大值 (向上取整到最接近的10的倍数，并确保至少比数据最大值大10%)
    y_axis_max = min(100, math.ceil((max_accuracy * 1.1) / 10) * 10)
    
    error_labels = []
    error_values = []
    
    for label, value in error_types.items():
        if value > 0:
            error_labels.append(f'"{label}"')
            error_values.append(str(value))
    
    # 如果没有错误数据，提供一个默认值
    if not error_labels:
        error_labels = ['"No Errors"']
        error_values = ['1']
    
    # 模型类型和引擎信息
    model_type_value = "API Model" if model_info.get("model_type") == "api" else "Local/Huggingface Model"
    
    if model_info.get("model_type") == "api":
        engine_value = model_info.get('model_engine') or model_info.get('model_name', 'Unknown')
        api_url_value = model_info.get('api_url', 'Default API')
    else:
        engine_value = "N/A"
        api_url_value = model_info.get('model_path', 'Unknown')
    
    # 初始化替换字典 - 关键是在这里先初始化
    replacements = {
        'TASK_NAME': task_name,
        'EVALUATION_ID': timestamp,
        'CURRENT_DATE': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'MODEL_TYPE_VALUE': model_type_value,
        'MODEL_ENGINE_VALUE': engine_value,
        'API_URL_VALUE': api_url_value,
        'TOTAL_QUESTIONS': str(total),
        'CORRECT_ANSWERS': str(correct),
        'ACCURACY_VALUE': f"{accuracy:.1f}",
        'AVG_TIME_VALUE': f"{avg_time:.2f}",
        'CORRECT_COUNT': str(correct),
        'INCORRECT_COUNT': str(total - correct),
        'Y_AXIS_MAX': str(y_axis_max)  # 添加Y轴最大值设置
    }
    
    # 添加图表数据替换项
    replacements["DISTRIBUTION_DATA"] = f"[{correct}, {total - correct}]"
    
    # 添加其他图表数据
    replacements.update({
        'DIFFICULTY_LABELS': "[" + ", ".join(difficulty_labels) + "]",
        'DIFFICULTY_DATA': "[" + ", ".join(difficulty_data) + "]",
        'LENGTH_LABELS': "[" + ", ".join(length_labels) + "]",
        'LENGTH_DATA': "[" + ", ".join(length_data) + "]",
        'DOMAIN_LABELS': "[" + ", ".join(domain_labels) + "]",
        'DOMAIN_DATA': "[" + ", ".join(domain_data) + "]",
        'ERROR_LABELS': "[" + ", ".join(error_labels) + "]",
        'ERROR_VALUES': "[" + ", ".join(error_values) + "]"
    })
    
    # HTML模板和其他代码...
    html_template_updated = '''<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retrieval Evaluation Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #333;
                background-color: #f8f9fa;
                line-height: 1.6;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.05);
                border-radius: 10px;
            }
            .report-header {
                background: linear-gradient(135deg, #3a0ca3, #4361ee);
                color: white;
                text-align: center;
                margin-bottom: 30px;
                padding: 40px 20px;
                border-radius: 8px;
            }
            .report-header h1 {
                font-weight: 700;
                margin-bottom: 10px;
                font-size: 2.2rem;
            }
            .report-header p {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
            }
            .section {
                margin-bottom: 25px;
                padding: 0;
            }
            .section-header {
                color: #3a0ca3;
                font-weight: 600;
                font-size: 1.3rem;
                padding-bottom: 8px;
                margin-bottom: 15px;
                border-bottom: 2px solid #3a0ca3;
            }
            .metric-value {
                font-size: 24px;
                font-weight: 700;
                color: #3a0ca3;
            }
            .metric-label {
                font-size: 14px;
                color: #666;
                font-weight: 500;
            }
            .chart-container {
                height: 300px;
                margin-bottom: 20px;
            }
            .info-card {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .data-row {
                display: flex;
                flex-wrap: wrap;
                margin: 0 -10px;
            }
            .data-col {
                flex: 1;
                padding: 0 10px;
                min-width: 200px;
                margin-bottom: 20px;
            }
            .chart-wrapper {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .chart-title {
                font-size: 16px;
                font-weight: 600;
                color: #3a0ca3;
                margin-bottom: 10px;
                text-align: center;
            }
            @media (max-width: 768px) {
                .data-col {
                    flex: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 报告标题 -->
            <div class="report-header">
                <h1>Retrieval Evaluation Report</h1>
                <p>Comprehensive analysis of model information retrieval capability assessment</p>
            </div>
            
            <!-- 模型信息 -->
            <div class="section">
                <div class="section-header">Model Information</div>
                <div class="info-card">
                    <div class="row">
                        <div class="col-md-6">
                            <h5>Model Details</h5>
                            <p><strong>Type:</strong> MODEL_TYPE_VALUE</p>
                            <p><strong>Engine:</strong> MODEL_ENGINE_VALUE</p>
                            <p><strong>API URL:</strong> API_URL_VALUE</p>
                        </div>
                        <div class="col-md-6">
                            <h5>Task Information</h5>
                            <p><strong>Task:</strong> TASK_NAME</p>
                            <p><strong>Evaluation ID:</strong> EVALUATION_ID</p>
                            <p><strong>Generated on:</strong> CURRENT_DATE</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 结果摘要 -->
            <div class="section">
                <div class="section-header">Results Summary</div>
                <div class="row mb-4">
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="info-card text-center h-100">
                            <div class="metric-label">Total Questions</div>
                            <div class="metric-value">TOTAL_QUESTIONS</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="info-card text-center h-100">
                            <div class="metric-label">Correct Answers</div>
                            <div class="metric-value">CORRECT_ANSWERS</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="info-card text-center h-100">
                            <div class="metric-label">Accuracy</div>
                            <div class="metric-value">ACCURACY_VALUE%</div>
                        </div>
                    </div>
                    <div class="col-md-3 col-sm-6 mb-3">
                        <div class="info-card text-center h-100">
                            <div class="metric-label">Avg Response Time</div>
                            <div class="metric-value">AVG_TIME_VALUEs</div>
                        </div>
                    </div>
                </div>
                
                <!-- 结果摘要图表 -->
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Overall Performance</div>
                            <canvas id="accuracyChart" height="250"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Response Distribution</div>
                            <canvas id="distributionChart" height="250"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 详细分析 -->
            <div class="section">
                <div class="section-header">Detailed Analysis</div>
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Accuracy by Difficulty Level</div>
                            <canvas id="difficultyChart" height="250"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Accuracy by Content Length</div>
                            <canvas id="lengthChart" height="250"></canvas>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Accuracy by Domain</div>
                            <canvas id="domainChart" height="250"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <div class="chart-wrapper">
                            <div class="chart-title">Error Analysis</div>
                            <canvas id="errorChart" height="250"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="text-center text-muted mt-4 mb-2">
                <small>Generated by LLM Evaluator Platform</small>
            </div>
        </div>

        <!-- JavaScript 图表生成 -->
        <script>
            // 准确率条形图
            var accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
            var accuracyChart = new Chart(accuracyCtx, {
                type: 'bar',
                data: {
                    labels: ['Accuracy'],
                    datasets: [{
                        label: 'Percentage (%)',
                        data: [ACCURACY_VALUE],
                        backgroundColor: '#4361ee',
                        borderColor: '#3a0ca3',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: Y_AXIS_MAX,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            
            // 修改分布饼图配置
            var distributionCtx = document.getElementById('distributionChart').getContext('2d');
            var distributionChart = new Chart(distributionCtx, {
                type: 'pie',
                data: {
                    labels: ['Correct', 'Incorrect'],
                    datasets: [{
                        data: DISTRIBUTION_DATA,
                        backgroundColor: ['#4CAF50', '#F44336'],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.6,  // 添加固定的宽高比
                    layout: {
                        padding: {
                            left: 10,
                            right: 10,
                            top: 0,
                            bottom: 10
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                boxWidth: 15
                            }
                        }
                    }
                }
            });
            
            // 难度级别准确率图表
            var difficultyCtx = document.getElementById('difficultyChart').getContext('2d');
            var difficultyChart = new Chart(difficultyCtx, {
                type: 'bar',
                data: {
                    labels: DIFFICULTY_LABELS,
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: DIFFICULTY_DATA,
                        backgroundColor: '#4361ee',
                        borderColor: '#3a0ca3',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: Y_AXIS_MAX,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            
            // 内容长度准确率图表
            var lengthCtx = document.getElementById('lengthChart').getContext('2d');
            var lengthChart = new Chart(lengthCtx, {
                type: 'bar',
                data: {
                    labels: LENGTH_LABELS,
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: LENGTH_DATA,
                        backgroundColor: '#4361ee',
                        borderColor: '#3a0ca3',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: Y_AXIS_MAX,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            
            // 领域准确率图表
            var domainCtx = document.getElementById('domainChart').getContext('2d');
            var domainChart = new Chart(domainCtx, {
                type: 'bar',
                data: {
                    labels: DOMAIN_LABELS,
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: DOMAIN_DATA,
                        backgroundColor: '#4361ee',
                        borderColor: '#3a0ca3',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: Y_AXIS_MAX,
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            
            // 修改错误分析饼图配置
            var errorCtx = document.getElementById('errorChart').getContext('2d');
            var errorChart = new Chart(errorCtx, {
                type: 'pie',
                data: {
                    labels: ERROR_LABELS,
                    datasets: [{
                        data: ERROR_VALUES,
                        backgroundColor: ['#F44336', '#FF9800', '#9C27B0'],
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.6,  // 添加固定的宽高比
                    layout: {
                        padding: {
                            left: 10,
                            right: 10,
                            top: 0,
                            bottom: 10
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: {
                                font: {
                                    size: 12
                                },
                                boxWidth: 15
                            }
                        }
                    }
                }
            });
        </script>
    </body>
    </html>'''
    
    # 应用所有替换
    for key, value in replacements.items():
        html_template_updated = html_template_updated.replace(key, value)
    
    return html_template_updated

def export_csv_data(responses):
    """生成CSV数据并返回base64编码的下载链接"""
    try:
        df = pd.DataFrame([
            {
                "Question ID": r.get("item_id", ""),
                "Question": r.get("question", ""),
                "Domain": r.get("domain", ""),
                "Difficulty": r.get("difficulty", ""),
                "Length": r.get("length", ""),
                "Correct Answer": r.get("ground_truth", ""),
                "Model Answer": r.get("extracted_answer", ""),
                "Is Correct": 1 if r.get("is_correct", False) else 0,
                "Response Time (s)": r.get("response_time", 0)
            } for r in responses
        ])
        
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return b64
    except Exception as e:
        st.error(f"Error generating CSV data: {str(e)}")
        return None

def extract_model_info(response_data):
    """增强的模型信息提取函数，特别优化DashScope API的识别"""
    model_info = {}
    
    # 1. 检查顶层配置信息
    top_level_fields = [
        "model", "model_name", "engine", "model_engine", 
        "api_url", "model_path", "model_type"
    ]
    
    for field in top_level_fields:
        if field in response_data:
            model_info[field] = response_data[field]
    
    # 2. 检查评估配置信息 (关键改进：检查config字段)
    if "config" in response_data and isinstance(response_data["config"], dict):
        config = response_data["config"]
        for field in top_level_fields + ["url"]:  # 增加检查url字段
            if field in config and field not in model_info:
                if field == "url":  # 处理url字段映射到api_url
                    model_info["api_url"] = config["url"]
                else:
                    model_info[field] = config[field]
    
    # 3. 检查元数据部分
    if "metadata" in response_data and isinstance(response_data["metadata"], dict):
        metadata = response_data["metadata"]
        
        # 检查metadata字段
        for field in top_level_fields:
            if field in metadata and field not in model_info:
                model_info[field] = metadata[field]
        
        # 检查model子对象
        if "model" in metadata:
            if isinstance(metadata["model"], dict):
                for field, value in metadata["model"].items():
                    if field not in model_info:
                        model_info[field] = value
            else:
                # 如果model是字符串值
                model_info["model_name"] = str(metadata["model"])
        
        # 检查config部分
        if "config" in metadata and isinstance(metadata["config"], dict):
            config = metadata["config"]
            for field in top_level_fields:
                if field in config and field not in model_info:
                    model_info[field] = config[field]
    
    # 4. 递归搜索关键字段 (新增功能)
    # 这是一个深度搜索，将找到嵌套在任何位置的信息
    def search_dict(d, target_keys, path=""):
        if not isinstance(d, dict):
            return {}
        
        results = {}
        for k, v in d.items():
            if k in target_keys and k not in model_info:
                results[k] = v
            
            # 特殊处理DashScope
            if k == "url" and "api_url" not in model_info and isinstance(v, str) and "dashscope" in v.lower():
                results["api_url"] = v
            
            if k == "engine" and "model_engine" not in model_info and isinstance(v, str) and "qwen" in v.lower():
                results["model_engine"] = v
            
            # 递归搜索
            if isinstance(v, dict):
                nested = search_dict(v, target_keys, path + "." + k if path else k)
                for nk, nv in nested.items():
                    if nk not in results:
                        results[nk] = nv
        
        return results
    
    # 执行递归搜索
    search_results = search_dict(response_data, top_level_fields + ["url", "engine"])
    for k, v in search_results.items():
        if k not in model_info:
            model_info[k] = v
    
    # 5. 特别处理DashScope API - 检查输入文本
    response_str = str(response_data)
    if "dashscope" in response_str.lower() or "qwen" in response_str.lower():
        if "model_type" not in model_info:
            model_info["model_type"] = "api"
        
        # 尝试从字符串中提取URL和engine
        # 提取URL
        url_match = re.search(r'https?://[^\"\')\s]+dashscope[^\"\')\s]*', response_str)
        if url_match and "api_url" not in model_info:
            model_info["api_url"] = url_match.group(0)
        
        # 提取qwen模型名称
        qwen_match = re.search(r'qwen[\-\w]+', response_str, re.IGNORECASE)
        if qwen_match and "model_engine" not in model_info:
            model_info["model_engine"] = qwen_match.group(0)
    
    # 6. 标准化字段
    if "engine" in model_info and "model_engine" not in model_info:
        model_info["model_engine"] = model_info["engine"]
    
    if "model" in model_info and "model_name" not in model_info:
        model_info["model_name"] = model_info["model"]
    
    if "url" in model_info and "api_url" not in model_info:
        model_info["api_url"] = model_info["url"]
    
    # 7. 根据其他字段推断模型类型
    if "model_type" not in model_info:
        if any(api in str(model_info.get("model_name", "")).lower() for api in ["gpt", "openai", "claude", "palm", "qwen", "gemini"]):
            model_info["model_type"] = "api"
        elif "model_path" in model_info:
            model_info["model_type"] = "local"
        elif "api_url" in model_info:
            model_info["model_type"] = "api"
    
    return model_info

def create_detailed_analysis(responses):
    """创建详细结果的数据框"""
    if not responses:
        return pd.DataFrame()
    
    # 构建详细结果数据
    details = []
    for i, resp in enumerate(responses):
        details.append({
            "Question ID": resp.get("item_id", f"Q{i+1}"),
            "Question": resp.get("question", ""),
            "Domain": resp.get("domain", "general").capitalize(),
            "Difficulty": resp.get("difficulty", "medium").capitalize(),
            "Correct Answer": resp.get("ground_truth", ""),
            "Model Answer": resp.get("extracted_answer", ""),
            "Is Correct": "✓" if resp.get("is_correct", False) else "✗",
            "Response Time (s)": f"{resp.get('response_time', 0):.2f}"
        })
    
    # 创建数据框
    return pd.DataFrame(details)

# 获取任务列表并进行检查
tasks = get_task_list()

if not tasks:
    st.warning("No tasks with retrieval evaluation results found. Please run a retrieval evaluation first.")
    st.stop()

with st.container(border=True):
    st.write("Task Selection")
    
    # 选择任务
    selected_task = st.selectbox("Select a task", tasks, index=0)
    
    # 加载该任务的评估结果
    task_path = os.path.join(TASKS_DIR, selected_task)
    evaluation_results = load_retrieval_results(task_path)
    
    if not evaluation_results:
        st.warning(f"No retrieval evaluation results found for task '{selected_task}'. Please run a retrieval evaluation for this task first.")
        st.stop()
    
    # 选择评估结果 - 现在直接放在任务选择下方
    result_options = [f"{r['formatted_time']} - Accuracy: {r['accuracy']:.1f}%" for r in evaluation_results]
    selected_index = st.selectbox("Select evaluation result", range(len(result_options)), 
                                 format_func=lambda i: result_options[i])
    
    selected_result = evaluation_results[selected_index]
    selected_timestamp = selected_result["timestamp"]
    
    st.info(f"Selected evaluation: {selected_result['formatted_time']}")

# 加载指标和响应数据
if selected_result["result_format"] == "combined":
    with open(selected_result["responses_file"], 'r', encoding='utf-8') as f:
        response_data = json.load(f)
    
    if "metrics" in response_data:
        metrics = response_data["metrics"]
    else:
        # 尝试从响应计算指标
        responses = response_data.get("responses", [])
        correct = sum(1 for r in responses if r.get("is_correct", False))
        total = len(responses)
        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0
        }
    
    responses = response_data.get("responses", [])
    # 使用增强的提取函数
    model_info = extract_model_info(response_data)

else:
    # 分离文件格式
    with open(selected_result["metrics_file"], 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    with open(selected_result["responses_file"], 'r', encoding='utf-8') as f:
        responses_data = json.load(f)
        responses = responses_data.get("responses", [])
        model_info = {k: responses_data.get(k) for k in ["model_type", "model_engine", "api_url", "model_path"] 
                     if k in responses_data}

# 保存到session state用于报告生成
st.session_state["loaded_metrics"] = metrics
st.session_state["loaded_responses"] = responses
st.session_state["model_info"] = model_info
st.session_state["selected_task"] = selected_task

# 2. 模型信息和评估摘要
with st.container(border=True):
    st.write("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Details**")
        if model_info:
            if model_info.get("model_type") == "api":
                st.write(f"Type: API Model")
                
                # 优先使用model_engine，其次使用model_name
                engine = model_info.get('model_engine') or model_info.get('model_name', 'Unknown')
                st.write(f"Model Engine: {engine}")
                
                # 显示API URL
                api_url = model_info.get('api_url', 'Default API')
                st.write(f"API URL: {api_url}")
            else:
                st.write(f"Type: Local/Huggingface Model")
                st.write(f"Model Path: {model_info.get('model_path', 'Unknown')}")
            
            # 显示任何其他可能的模型信息
            if "additional_info" in model_info:
                st.write(f"Additional Info: {model_info['additional_info']}")
        else:
            st.write("Model information not available")

# 3. 评估结果摘要
with st.container(border=True):
    st.write("Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Questions", metrics.get("total", 0))
    
    with col2:
        st.metric("Correct Answers", metrics.get("correct", 0))
    
    with col3:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
    
    with col4:
        st.metric("Avg Response Time", f"{metrics.get('avg_response_time', 0):.2f}s")
    
    # 创建总体准确率图表和饼图
    col1, col2 = st.columns(2)
    
    with col1:
        # 总体准确率条形图
        fig = create_accuracy_chart(metrics)
        st.pyplot(fig)
    
    with col2:
        # 正确/错误分布饼图
        correct_count = metrics.get("correct", 0)
        total_count = metrics.get("total", 0)
        fig = create_pie_chart(correct_count, total_count, "Response Distribution")
        st.pyplot(fig)

# 4. 详细分析
if "loaded_metrics" in st.session_state and "loaded_responses" in st.session_state:
    metrics = st.session_state["loaded_metrics"]
    responses = st.session_state["loaded_responses"]
    
    with st.container(border=True):
        st.write("Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Accuracy Distribution", "Domain Analysis", "Error Analysis", "Detailed Results"])
        
        # Tab 1: 准确率分布
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # 使用更新后的难度图表函数
                fig = create_difficulty_chart(responses, figsize=(5, 3))
                st.pyplot(fig)
                st.info("This chart shows accuracy across different difficulty levels.")
            
            with col2:
                # 使用更新后的长度图表函数
                fig = create_length_chart(responses, figsize=(5, 3))
                st.pyplot(fig)
                st.info("This chart shows how accuracy varies with context length.")
        
        # Tab 2: 领域分析
        with tab2:
            # 使用新的领域图表函数
            fig = create_domain_chart(responses)
            st.pyplot(fig)
            st.info("This chart shows accuracy across different domains.")
        
        # Tab 3: 错误分析
        with tab3:
            # 使用新的错误分析图表函数
            fig = create_error_analysis(responses)
            st.pyplot(fig)
            st.info("Error analysis shows the distribution of error types in incorrect responses.")
        
        # Tab 4: 详细结果
        with tab4:
            # 创建详细的Pandas数据框
            detailed_df = create_detailed_analysis(responses)
            
            # 显示数据框
            st.dataframe(detailed_df, use_container_width=True)

# 5. 样本检查
if "loaded_responses" in st.session_state:
    responses = st.session_state["loaded_responses"]
    
    with st.container(border=True):
        st.write("Sample Inspection")
        
        # 添加按类型筛选选项
        filter_options = ["All", "Correct Only", "Incorrect Only"]
        filter_type = st.radio("Filter results", filter_options, horizontal=True)
        
        filtered_responses = []
        if filter_type == "Correct Only":
            filtered_responses = [r for r in responses if r.get("is_correct", False)]
        elif filter_type == "Incorrect Only":
            filtered_responses = [r for r in responses if not r.get("is_correct", False)]
        else:
            filtered_responses = responses
        
        # 显示前10个示例
        sample_count = min(10, len(filtered_responses))
        for i, response in enumerate(filtered_responses[:sample_count]):
            with st.expander(f"Question {i+1}: {response.get('question', '')[:50]}...", expanded=(i==0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Question Details**")
                    st.markdown(f"**ID**: {response.get('item_id', '')}")
                    st.markdown(f"**Domain**: {response.get('domain', '')}")
                    st.markdown(f"**Difficulty**: {response.get('difficulty', '').capitalize()}")
                    st.markdown(f"**Length**: {response.get('length', '').capitalize()}")
                    
                    st.markdown("**Options**:")
                    for key, value in response.get("options", {}).items():
                        st.markdown(f"- **{key}**: {value}")
                    
                    st.markdown(f"**Correct Answer**: {response.get('ground_truth', '')}")
                
                with col2:
                    st.markdown("**Model Response**")
                    st.text_area("Raw response", value=response.get('model_response', ''), 
                            height=150, disabled=True, key=f"resp_{i}")
                    
                    extracted = response.get('extracted_answer', None)
                    st.markdown(f"**Extracted Answer**: {extracted if extracted else 'Not extracted'}")
                    
                    is_correct = response.get('is_correct', False)
                    st.markdown(f"**Result**: {'✓ Correct' if is_correct else '✗ Incorrect'}")
                    
                    st.markdown(f"**Response Time**: {response.get('response_time', 0):.2f} seconds")

# 6. 报告导出
if all(k in st.session_state for k in ["loaded_metrics", "loaded_responses", "model_info", "selected_task"]):
    metrics = st.session_state["loaded_metrics"]
    responses = st.session_state["loaded_responses"]
    model_info = st.session_state["model_info"]
    selected_task = st.session_state["selected_task"]
    
    with st.container(border=True):
        st.write("Report Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate HTML Report", use_container_width=True):
                try:
                    with st.spinner("Generating report..."):
                        # 自定义HTML报告生成
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        html_content = export_report_html(selected_task, selected_timestamp, metrics, responses, model_info)
                        
                        # 将HTML内容编码为base64
                        b64 = base64.b64encode(html_content.encode()).decode()
                        file_name = f"{selected_task}_retrieval_report_{timestamp}.html"
                        
                        # 创建下载链接
                        href = f'<a href="data:text/html;base64,{b64}" download="{file_name}">Click to download HTML report</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success("Report generated successfully! Click the link above to download.")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
        
        with col2:
            if st.button("Export CSV Data", use_container_width=True):
                try:
                    with st.spinner("Generating CSV data..."):
                        # 生成CSV数据
                        b64_csv = export_csv_data(responses)
                        
                        if b64_csv:
                            # 创建时间戳
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"{selected_task}_retrieval_results_{timestamp}.csv"
                            
                            # 创建下载链接
                            href = f'<a href="data:text/csv;base64,{b64_csv}" download="{file_name}">Click to download CSV data</a>'
                            
                            # 显示链接和成功消息
                            st.markdown(href, unsafe_allow_html=True)
                            st.success("CSV data generated successfully! Click the link above to download.")
                except Exception as e:
                    st.error(f"Error exporting CSV: {e}")
                    import traceback
                    st.code(traceback.format_exc())

