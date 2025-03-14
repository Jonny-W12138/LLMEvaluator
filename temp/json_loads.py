import json

with open("tasks/deepseekv3/computation/math/response/math_response_2025-03-09_10-29-13.json", "r", encoding="utf-8") as file:
    data = json.load(file)
print(data['response'][-1]['llm_output']['answer'].replace("\\\\", "\\"))