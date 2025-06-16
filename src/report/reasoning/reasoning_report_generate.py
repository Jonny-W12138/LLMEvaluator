def init_template(task_name):
    html_template = '''\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reasoning Evaluation Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            width: 100%;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 style="text-align: center">Reasoning Evaluation Report</h1>
        <p style="text-align: center">Task name: {{task_name}}</p>
        <p style="text-align: center">Generated time: {{generated_time}}</p>
'''
    return html_template

def generate_reasoning_report(json_path, report_template):
    report_template += '''\
    <h3>Reasoning Evaluation Results</h3>
    <div class="card">
        <div class="row mt-4">
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Total Samples</h5>
                    <p class="display-6">{{total_samples}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Answer Accuracy</h5>
                    <p class="display-6">{{average_answer_accuracy}}</p>
                </div>
            </div>
            <div class="col-md-4">
                <div class="p-3">
                    <h5>Average Step Accuracy</h5>
                    <p class="display-6">{{average_step_accuracy}}</p>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col text-center">
                <h3>Accuracy Distribution by Level</h3>
                <canvas id="level_accuracy_chart"></canvas>
            </div>
        </div>
    </div>
    '''
    
    # 加载结果数据
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 计算统计信息
    total_samples = len(results)
    avg_answer_acc = sum(r["answer_accuracy"] for r in results) / total_samples
    avg_step_acc = sum(r["step_accuracy"] for r in results) / total_samples
    
    # 生成难度级别分布图
    levels = sorted(set(r["level"] for r in results))
    level_stats = {}
    for level in levels:
        level_results = [r for r in results if r["level"] == level]
        level_stats[level] = {
            "answer_acc": sum(r["answer_accuracy"] for r in level_results) / len(level_results),
            "step_acc": sum(r["step_accuracy"] for r in level_results) / len(level_results),
            "ap_acc": sum(r["ap_accuracy"] for r in level_results) / len(level_results)
        }
    
    # 生成图表
    plt.figure(figsize=(10, 6))
    x = np.arange(len(levels))
    width = 0.25
    
    plt.bar(x - width, [level_stats[l]["answer_acc"] for l in levels], width, label='Answer Accuracy')
    plt.bar(x, [level_stats[l]["step_acc"] for l in levels], width, label='Step Accuracy')
    plt.bar(x + width, [level_stats[l]["ap_acc"] for l in levels], width, label='AP Accuracy')
    
    plt.xlabel('Difficulty Level')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Difficulty Level')
    plt.xticks(x, [f'Level {l}' for l in levels])
    plt.legend()
    
    # 保存图表为base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    # 填充模板
    html_filled = report_template \
        .replace("{{total_samples}}", str(total_samples)) \
        .replace("{{average_answer_accuracy}}", f"{avg_answer_acc:.2%}") \
        .replace("{{average_step_accuracy}}", f"{avg_step_acc:.2%}") \
        .replace('<canvas id="level_accuracy_chart"></canvas>',
                f'<img src="data:image/png;base64,{img_base64}" class="img-fluid" alt="Level Accuracy Distribution">')
    
    return html_filled 