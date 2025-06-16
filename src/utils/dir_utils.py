"""
目录和文件工具函数
"""

import os
import streamlit as st

def ensure_retrieval_directories(task_name="default_task"):
    """
    确保检索评估所需目录结构存在
    
    Args:
        task_name: 任务名称
    """
    # 确保路径是相对于项目根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    dirs = [
        os.path.join(base_dir, f"tasks/{task_name}/retrieval"),
        os.path.join(base_dir, f"tasks/{task_name}/retrieval/reports"),
        os.path.join(base_dir, f"tasks/{task_name}/retrieval/charts")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def get_current_task():
    """
    获取当前任务名称
    
    Returns:
        当前任务名称
    """
    if "current_task" not in st.session_state:
        st.session_state["current_task"] = "default_task"
    
    return st.session_state["current_task"] 