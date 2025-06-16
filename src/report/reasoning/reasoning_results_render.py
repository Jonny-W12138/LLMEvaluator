import streamlit as st
import json
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

def reasoning_results_render(file_path):
    """渲染推理评估结果"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 显示任务信息
    st.subheader("Task Information")
    st.write(f"Task Name: {data['task_name']}")
    st.write(f"Evaluation Metric: {data['metric']}")
    st.write(f"Evaluation Time: {data['timestamp']}")
    
    # 转换为DataFrame
    df = pd.DataFrame(data["results"])
    
    # 1. 总体统计
    st.subheader("Overall Statistics")
    total_num_col, avg_score_col = st.columns(2)
    with total_num_col:
        st.metric("Total Samples", df.shape[0])
    with avg_score_col:
        st.metric("Average Score", f"{df['score'].mean():.2%}")
    
    # 2. 难度级别分析
    st.subheader("Difficulty Level Analysis")
    level_stats = df.groupby("level")["score"].agg([
        ("Sample Count", "count"),
        ("Average Score", "mean")
    ]).reset_index()
    
    # 创建柱状图
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=level_stats["level"],
        y=level_stats["Average Score"],
        name="Score"
    ))
    
    fig.update_layout(
        title=f"{data['metric']} Score Distribution by Difficulty Level",
        xaxis_title="Difficulty Level",
        yaxis_title="Score",
        yaxis_tickformat=".2%"
    )
    
    st.plotly_chart(fig) 