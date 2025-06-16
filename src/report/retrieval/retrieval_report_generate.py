"""
检索评估报告生成
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from jinja2 import Template

from .retrieval_results_render import RetrievalResultsRenderer

class RetrievalReportGenerator:
    def __init__(self, results_path: str):
        """
        初始化检索报告生成器
        
        Args:
            results_path: 评估结果JSON文件路径
        """
        self.results_path = results_path
        self.renderer = RetrievalResultsRenderer(results_path)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """
        加载评估结果
        
        Returns:
            评估结果字典
        """
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_html_report(self, output_path: str) -> None:
        """
        Generate HTML format evaluation report
        
        Args:
            output_path: Output file path
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get evaluation metrics
        metrics = self.results.get("metrics", {})
        
        # Generate charts
        charts_dir = os.path.join(os.path.dirname(output_path), "charts")
        chart_paths = self.renderer.render_all_charts(charts_dir)
        
        # Convert to relative paths
        rel_chart_paths = {}
        for key, path in chart_paths.items():
            rel_chart_paths[key] = os.path.relpath(path, os.path.dirname(output_path))
        
        # Detailed analysis data
        df = self.renderer.create_detailed_analysis()
        detailed_table_html = df.to_html(classes="table table-striped", index=False, escape=False)
        
        # Get model response examples
        responses = self.results.get("responses", [])
        example_responses = responses[:5] if len(responses) > 5 else responses
        
        # HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Retrieval Capability Evaluation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #4361ee;
                }
                .summary-box {
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metrics {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .metric-card {
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    flex: 1;
                    min-width: 200px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4361ee;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .response-example {
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .response-question {
                    font-weight: bold;
                }
                .correct {
                    color: #4CAF50;
                }
                .incorrect {
                    color: #F44336;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Retrieval Capability Evaluation Report</h1>
                <p>Generated on: {{ timestamp }}</p>
                
                <div class="summary-box">
                    <h2>Evaluation Summary</h2>
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">{{ metrics.total }}</div>
                            <div>Total Questions</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ metrics.correct }}</div>
                            <div>Correct Answers</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.1f"|format(metrics.accuracy * 100) }}%</div>
                            <div>Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{{ "%.2f"|format(metrics.avg_response_time) }}s</div>
                            <div>Avg Response Time</div>
                        </div>
                    </div>
                </div>
                
                <h2>Visualization Analysis</h2>
                
                <h3>Overall Accuracy</h3>
                <div class="chart-container">
                    <img src="{{ charts.accuracy }}" alt="Overall Accuracy">
                </div>
                
                <h3>Accuracy by Difficulty</h3>
                <div class="chart-container">
                    <img src="{{ charts.difficulty }}" alt="Accuracy by Difficulty">
                </div>
                
                <h3>Accuracy by Context Length</h3>
                <div class="chart-container">
                    <img src="{{ charts.length }}" alt="Accuracy by Context Length">
                </div>
                
                {% if 'domain' in charts %}
                <h3>Accuracy by Domain</h3>
                <div class="chart-container">
                    <img src="{{ charts.domain }}" alt="Accuracy by Domain">
                </div>
                {% endif %}
                
                <h3>Error Analysis</h3>
                <div class="chart-container">
                    <img src="{{ charts.error }}" alt="Error Analysis">
                </div>
                
                <h2>Sample Responses</h2>
                
                {% for response in example_responses %}
                <div class="response-example">
                    <div class="response-question">Question: {{ response.question }}</div>
                    <p>ID: {{ response.item_id }}</p>
                    <p>Domain: {{ response.domain }}</p>
                    <p>Difficulty: {{ response.difficulty }}</p>
                    <p>Length: {{ response.length }}</p>
                    
                    <p><strong>Options:</strong></p>
                    <ul>
                    {% for key, value in response.options.items() %}
                        <li>{{ key }}: {{ value }}</li>
                    {% endfor %}
                    </ul>
                    
                    <p><strong>Correct Answer:</strong> {{ response.ground_truth }}</p>
                    <p><strong>Model Response:</strong> {{ response.model_response }}</p>
                    <p><strong>Extracted Answer:</strong> {{ response.extracted_answer }}</p>
                    <p><strong>Result:</strong> <span class="{{ 'correct' if response.is_correct else 'incorrect' }}">{{ "Correct" if response.is_correct else "Incorrect" }}</span></p>
                    <p><strong>Response Time:</strong> {{ "%.2f"|format(response.response_time) }} seconds</p>
                </div>
                {% endfor %}
                
                <h2>Detailed Results</h2>
                {{ detailed_table|safe }}
                
            </div>
        </body>
        </html>
        """
        
        # Fill template
        template = Template(template_str)
        html_content = template.render(
            metrics=metrics,
            charts=rel_chart_paths,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            example_responses=example_responses,
            detailed_table=detailed_table_html
        )
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")
    
    def generate_markdown_report(self, output_path: str) -> None:
        """
        Generate Markdown format evaluation report
        
        Args:
            output_path: Output file path
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get evaluation metrics
        metrics = self.results.get("metrics", {})
        
        # Generate charts
        charts_dir = os.path.join(os.path.dirname(output_path), "charts")
        chart_paths = self.renderer.render_all_charts(charts_dir)
        
        # Convert to relative paths
        rel_chart_paths = {}
        for key, path in chart_paths.items():
            rel_chart_paths[key] = os.path.relpath(path, os.path.dirname(output_path))
        
        # Get model response examples
        responses = self.results.get("responses", [])
        example_responses = responses[:3] if len(responses) > 3 else responses
        
        # Create Markdown content
        md_content = f"""# Retrieval Capability Evaluation Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Evaluation Summary

- **Total Questions**: {metrics.get("total", 0)}
- **Correct Answers**: {metrics.get("correct", 0)}
- **Accuracy**: {metrics.get("accuracy", 0) * 100:.1f}%
- **Average Response Time**: {metrics.get("avg_response_time", 0):.2f} seconds

## Visualization Analysis

### Overall Accuracy

![Overall Accuracy]({rel_chart_paths.get("accuracy", "")})

### Accuracy by Difficulty

![Accuracy by Difficulty]({rel_chart_paths.get("difficulty", "")})

### Accuracy by Context Length

![Accuracy by Context Length]({rel_chart_paths.get("length", "")})

"""
        
        # Add domain chart if available
        if "domain" in rel_chart_paths:
            md_content += f"""### Accuracy by Domain

![Accuracy by Domain]({rel_chart_paths.get("domain", "")})

"""
        
        md_content += f"""### Error Analysis

![Error Analysis]({rel_chart_paths.get("error", "")})

## Sample Responses

"""
        
        # Add response examples
        for i, response in enumerate(example_responses):
            is_correct = response.get("is_correct", False)
            result_str = "✓ Correct" if is_correct else "✗ Incorrect"
            
            md_content += f"""### Example {i+1}

**Question**: {response.get("question", "")}

**Options**:
"""
            
            for key, value in response.get("options", {}).items():
                md_content += f"- {key}: {value}\n"
            
            md_content += f"""
**Correct Answer**: {response.get("ground_truth", "")}

**Model Response**: {response.get("model_response", "")}

**Extracted Answer**: {response.get("extracted_answer", "Not extracted")}

**Result**: {result_str}

**Response Time**: {response.get("response_time", 0):.2f} seconds

"""
        
        # Write Markdown file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown report generated: {output_path}")
    
    def generate_report(self, output_dir: str, formats: List[str] = ["html", "md"]) -> Dict[str, str]:
        """
        生成评估报告
        
        Args:
            output_dir: 输出目录
            formats: 报告格式列表，可选"html"和"md"
            
        Returns:
            报告文件路径字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_paths = {}
        
        if "html" in formats:
            html_path = os.path.join(output_dir, f"retrieval_report_{timestamp}.html")
            self.generate_html_report(html_path)
            report_paths["html"] = html_path
        
        if "md" in formats:
            md_path = os.path.join(output_dir, f"retrieval_report_{timestamp}.md")
            self.generate_markdown_report(md_path)
            report_paths["md"] = md_path
        
        return report_paths
