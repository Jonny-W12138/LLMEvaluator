from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")


def process_json(input_file, output_file, instruction):
    # 读取输入的 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 用于存储处理结果
    results = []

    # 遍历每条记录
    for record in tqdm(data, desc="Processing records", unit="record"):
        index = record.get("index")
        full_text = record.get("full-text")
        text_type = record.get("categorie")

        if text_type == "news" or text_type == "academic":
            instruction = """Please follow the given rules and summarize the given text:
rules:
[1] have at most 380 characters
[2] have the goal of replacing reading the text
[3] give participants/time/place/manner of events
[4] form a sentence rather than a fragment
[5] omit distracting information
[6] avoid entities or information not present in the text, even if we are fairly sure it is true
[7] reject synonyms for words in the text
text to summarize:\n"""

        elif text_type == "biography":
            instruction = """Please follow the given rules and summarize the given text:
rules:
[1] have at most 380 characters
[2] have the goal of replacing reading the text
[3] typically take the form “Kim is/was a French X who … ”
[4] typically include information about what this person is/was known for (“… best known
for …”)
[5] information about the time period and place is typically included (“a Japanese X”, “a
German X living in France”, “a 19th century Kenyan X”)
[6] form a sentence rather than a fragment
[7] omit distracting information
[8] avoid entities or information not present in the text, even if we are fairly sure it is true
[9] reject synonyms for words in the text
text to summarize:\n"""

        elif text_type == "travel guide":
            instruction = """Please follow the given rules and summarize the given text:
rules:
[1] have at most 380 characters
[2] have the goal of replacing reading the text
[3] typically take the form “xxx is/was a city … ”
[4] typically include information about what this city is/was known for (“… best known
for …”)
[5] information about the feature of the city
[6] form a sentence rather than a fragment
[7] omit distracting information
[8] avoid entities or information not present in the text, even if we are fairly sure it is true
[9] reject synonyms for words in the text
text to summarize:\n"""
        else:
            raise ValueError(f"ERR: index {index} has no categorie to match!")

        # 构造模型输入的消息
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{full_text}"}
        ]

        try:
            # 调用模型 API 获取输出
            response = client.chat.completions.create(
                messages=messages,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                temperature=0,  # 设置为0，提高生成内容的确定性
                top_p=1,  # 设置为1，确保不限制采样范围
                max_tokens=400,  # 限制生成内容的最大长度
                frequency_penalty=0,  # 控制生成内容中重复单词的惩罚程度
                presence_penalty=0  # 控制生成内容是否重复提及已有信息

            )
            # 获取模型的回复
            output = response.choices[0].message.content

            # 记录结果
            results.append({
                "index": index,
                "output": output
            })

        except Exception as e:
            print(f"处理 index {index} 时出错: {e}")
            results.append({
                "index": index,
                "output": f"Error: {e}"
            })

    # 将结果写入到输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file}")


# 示例用法
if __name__ == "__main__":
    input_file = "/root/autodl-tmp/code/summary_gen/my_summarization_dataset.json"  # 输入 JSON 文件路径
    output_file = "/root/autodl-tmp/code/summary_gen/my_summarization_output.json"  # 输出 JSON 文件路径
    instruction = "Please summarize the following text:"  # 用户给定的指令
    process_json(input_file, output_file, instruction)