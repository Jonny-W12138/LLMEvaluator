"""
检索评估结果可视化
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import os

class RetrievalResultsRenderer:
    def __init__(self, results_path: str):
        """
        初始化检索结果渲染器
        
        Args:
            results_path: 评估结果JSON文件路径
        """
        self.results_path = results_path
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """
        加载评估结果
        
        Returns:
            评估结果字典
        """
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_accuracy_chart(self, save_path: str = None) -> plt.Figure:
        """
        创建总体准确率图表
        
        Args:
            save_path: 保存图表的路径
            
        Returns:
            Matplotlib图表对象
        """
        metrics = self.results.get("metrics", {})
        accuracy = metrics.get("accuracy", 0) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(["准确率"], [accuracy], color='#4CAF50', width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('准确率 (%)')
        ax.set_title('检索能力评估总体准确率')
        
        # 在柱状图上添加标签
        ax.text(0, accuracy + 2, f"{accuracy:.1f}%", ha='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_difficulty_chart(self, save_path: str = None) -> plt.Figure:
        """
        创建按难度分类的准确率图表
        
        Args:
            save_path: 保存图表的路径
            
        Returns:
            Matplotlib图表对象
        """
        metrics = self.results.get("metrics", {})
        by_difficulty = metrics.get("by_difficulty", {})
        
        difficulties = []
        accuracies = []
        
        for level, data in by_difficulty.items():
            difficulties.append("简单" if level == "easy" else "困难")
            accuracies.append(data.get("accuracy", 0) * 100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(difficulties, accuracies, color=['#4CAF50', '#F44336'], width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('准确率 (%)')
        ax.set_title('按难度分类的检索准确率')
        
        # 在柱状图上添加标签
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 2, f"{acc:.1f}%", ha='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_length_chart(self, save_path: str = None) -> plt.Figure:
        """
        创建按长度分类的准确率图表
        
        Args:
            save_path: 保存图表的路径
            
        Returns:
            Matplotlib图表对象
        """
        metrics = self.results.get("metrics", {})
        by_length = metrics.get("by_length", {})
        
        lengths = []
        accuracies = []
        
        for length, data in by_length.items():
            if length == "short":
                display_name = "短"
            elif length == "medium":
                display_name = "中"
            else:
                display_name = "长"
            
            lengths.append(display_name)
            accuracies.append(data.get("accuracy", 0) * 100)
        
        # 按短、中、长排序
        order = {"短": 0, "中": 1, "长": 2}
        sorted_data = sorted(zip(lengths, accuracies), key=lambda x: order.get(x[0], 999))
        lengths = [x[0] for x in sorted_data]
        accuracies = [x[1] for x in sorted_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(lengths, accuracies, color=['#4CAF50', '#2196F3', '#9C27B0'], width=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('准确率 (%)')
        ax.set_title('按长度分类的检索准确率')
        
        # 在柱状图上添加标签
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 2, f"{acc:.1f}%", ha='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_domain_chart(self, save_path: str = None) -> plt.Figure:
        """
        创建按领域分类的准确率图表
        
        Args:
            save_path: 保存图表的路径
            
        Returns:
            Matplotlib图表对象
        """
        metrics = self.results.get("metrics", {})
        by_domain = metrics.get("by_domain", {})
        
        domains = []
        accuracies = []
        
        for domain, data in by_domain.items():
            domains.append(domain)
            accuracies.append(data.get("accuracy", 0) * 100)
        
        # 按准确率排序
        sorted_data = sorted(zip(domains, accuracies), key=lambda x: x[1], reverse=True)
        domains = [x[0] for x in sorted_data]
        accuracies = [x[1] for x in sorted_data]
        
        # 限制显示数量，如果太多
        if len(domains) > 10:
            domains = domains[:10]
            accuracies = accuracies[:10]
            plt.figure(figsize=(12, 8))
        else:
            plt.figure(figsize=(10, 6))
        
        # 水平条形图
        fig, ax = plt.subplots(figsize=(10, max(6, len(domains) * 0.5)))
        bars = ax.barh(domains, accuracies, color='#3F51B5')
        ax.set_xlim(0, 100)
        ax.set_xlabel('准确率 (%)')
        ax.set_title('按领域分类的检索准确率')
        
        # 在柱状图上添加标签
        for bar, acc in zip(bars, accuracies):
            ax.text(acc + 1, bar.get_y() + bar.get_height()/2, f"{acc:.1f}%", va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_error_analysis(self, save_path: str = None) -> plt.Figure:
        """
        创建错误分析图表
        
        Args:
            save_path: 保存图表的路径
            
        Returns:
            Matplotlib图表对象
        """
        responses = self.results.get("responses", [])
        
        # 初始化错误统计
        error_types = {
            "未提取到答案": 0,
            "提取错误答案": 0
        }
        
        for response in responses:
            if not response.get("is_correct", False):
                if response.get("extracted_answer") is None:
                    error_types["未提取到答案"] += 1
                else:
                    error_types["提取错误答案"] += 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(
            error_types.values(), 
            labels=error_types.keys(), 
            autopct='%1.1f%%',
            colors=['#FF9800', '#F44336'],
            startangle=90
        )
        ax.axis('equal')
        ax.set_title('检索错误类型分布')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def render_all_charts(self, output_dir: str) -> Dict[str, str]:
        """
        渲染所有图表并保存
        
        Args:
            output_dir: 输出目录
            
        Returns:
            图表路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {}
        
        # 生成并保存各种图表
        charts["accuracy"] = os.path.join(output_dir, "accuracy_chart.png")
        self.create_accuracy_chart(charts["accuracy"])
        
        charts["difficulty"] = os.path.join(output_dir, "difficulty_chart.png")
        self.create_difficulty_chart(charts["difficulty"])
        
        charts["length"] = os.path.join(output_dir, "length_chart.png")
        self.create_length_chart(charts["length"])
        
        metrics = self.results.get("metrics", {})
        if metrics.get("by_domain", {}):
            charts["domain"] = os.path.join(output_dir, "domain_chart.png")
            self.create_domain_chart(charts["domain"])
        
        charts["error"] = os.path.join(output_dir, "error_analysis.png")
        self.create_error_analysis(charts["error"])
        
        return charts
    
    def create_detailed_analysis(self) -> pd.DataFrame:
        """
        创建详细分析数据框
        
        Returns:
            包含详细分析的Pandas DataFrame
        """
        responses = self.results.get("responses", [])
        
        analysis_data = []
        for resp in responses:
            if "error" not in resp:  # 跳过处理错误的项
                analysis_data.append({
                    "ID": resp.get("item_id", ""),
                    "领域": resp.get("domain", ""),
                    "子领域": resp.get("sub_domain", ""),
                    "难度": resp.get("difficulty", ""),
                    "长度": resp.get("length", ""),
                    "问题": resp.get("question", "")[:100] + "...",
                    "正确答案": resp.get("ground_truth", ""),
                    "模型答案": resp.get("extracted_answer", ""),
                    "正确性": "✓" if resp.get("is_correct", False) else "✗",
                    "响应时间(秒)": round(resp.get("response_time", 0), 2)
                })
        
        return pd.DataFrame(analysis_data)
