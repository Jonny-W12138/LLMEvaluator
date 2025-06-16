import os
import json
from datetime import datetime
from .step_checker import StepChecker
from .response_generate import ReasoningResponseGenerator
from config import TASKS_DIR

class ReasoningEvaluator:
    """推理评估器基类"""
    
    def __init__(self):
        self.step_checker = StepChecker()
        self.response_generator = ReasoningResponseGenerator()
    
    def evaluate_batch(self, model, dataset, task_name, eval_model=None):
        """批量评估
        Args:
            model: 使用的模型
            dataset: 数据集
            task_name: 任务名称
            eval_model: 用于评估的模型，如果为None则使用step_checker进行评估
        Returns:
            评估结果dict
        """
        results = []
        for item in dataset:
            # 生成响应
            raw_response = self.response_generator.generate(
                model,
                item["contexts"],
                item.get("examples")
            )
            
            # 解析reference
            if isinstance(item["reference"], str):
                reference = json.loads(item["reference"])
            else:
                reference = item["reference"]
            
            # 评估结果
            result = self.evaluate(raw_response, reference, eval_model)
            result.update({
                "qid": item["qid"],
                "level": item.get("level", 0),
                "category": item.get("category", ""),
                "model_response": raw_response  # 保存原始响应用于显示
            })
            results.append(result)
            
        # 保存结果
        self._save_results(results, task_name, model, eval_model)
        return results
    
    def evaluate(self, response, reference, eval_model=None):
        """评估单条结果
        Args:
            response: 模型响应
            reference: 参考答案
            eval_model: 用于评估的模型，如果为None则使用step_checker进行评估
        """
        try:
            # 确保response是字典类型
            if isinstance(response, str):
                try:
                    # 尝试从文本中提取JSON部分
                    # 首先检查是否有```json和```包裹的部分
                    import re
                    json_pattern = r"```json\s*([\s\S]*?)\s*```"
                    matches = re.findall(json_pattern, response)
                    if matches:
                        json_str = matches[0]
                    else:
                        # 尝试直接查找JSON对象
                        start = response.find("{")
                        end = response.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_str = response[start:end]
                        else:
                            # 没有找到有效的JSON对象
                            raise ValueError("No valid JSON found")
                    
                    try:
                        # 尝试直接解析
                        response = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 处理非标准格式：带索引的键值对，如 "0:"
                        # 例如: {"answer": {"0:": "yes"}, "process": {"0:": "step 1", "1:": "step 2"}}
                        # 转换为: {"answer": ["yes"], "process": ["step 1", "step 2"]}
                        cleaned_json = {}
                        for field in ["answer", "process"]:
                            if f'"{field}":' in json_str or f"'{field}':" in json_str:
                                # 提取字段值
                                field_pattern = r'["\']' + field + r'["\']:\s*(\{[^{}]*\}|\[[^\[\]]*\]|"[^"]*"|\'[^\']*\')'
                                field_matches = re.findall(field_pattern, json_str)
                                
                                if field_matches:
                                    field_value = field_matches[0]
                                    
                                    # 如果是字典格式
                                    if field_value.startswith("{"):
                                        # 提取键值对
                                        pairs_pattern = r'["\'](\d+:?)["\']:\s*("[^"]*"|\'[^\']*\')'
                                        pairs = re.findall(pairs_pattern, field_value)
                                        
                                        # 按索引排序
                                        pairs.sort(key=lambda x: int(x[0].rstrip(':')))
                                        
                                        # 提取值并去除引号
                                        values = []
                                        for _, value in pairs:
                                            # 去除首尾引号
                                            if value.startswith('"') and value.endswith('"'):
                                                values.append(value[1:-1])
                                            elif value.startswith("'") and value.endswith("'"):
                                                values.append(value[1:-1])
                                            else:
                                                values.append(value)
                                        
                                        cleaned_json[field] = values
                                    
                                    # 如果直接是字符串
                                    elif field_value.startswith('"') or field_value.startswith("'"):
                                        # 去除首尾引号
                                        if field_value.startswith('"') and field_value.endswith('"'):
                                            cleaned_json[field] = field_value[1:-1]
                                        elif field_value.startswith("'") and field_value.endswith("'"):
                                            cleaned_json[field] = field_value[1:-1]
                                        else:
                                            cleaned_json[field] = field_value
                                    
                                    # 如果是列表格式
                                    elif field_value.startswith("["):
                                        try:
                                            # 尝试解析为列表
                                            cleaned_json[field] = json.loads(field_value)
                                        except:
                                            # 如果解析失败，尝试直接处理
                                            items_pattern = r'("[^"]*"|\'[^\']*\')'
                                            items = re.findall(items_pattern, field_value)
                                            
                                            # 去除首尾引号
                                            values = []
                                            for item in items:
                                                if item.startswith('"') and item.endswith('"'):
                                                    values.append(item[1:-1])
                                                elif item.startswith("'") and item.endswith("'"):
                                                    values.append(item[1:-1])
                                                else:
                                                    values.append(item)
                                            
                                            cleaned_json[field] = values
                        
                        # 如果成功提取了必要字段
                        if cleaned_json and ("answer" in cleaned_json or "process" in cleaned_json):
                            response = cleaned_json
                        else:
                            # 无法解析，创建一个空结果
                            response = {"answer": "", "process": []}
                    
                except Exception as e:
                    print(f"解析响应JSON时出错: {e}")
                    # 解析失败时，构造一个空结果字典
                    response = {"answer": "", "process": []}
            elif not isinstance(response, dict):
                # 确保response是字典
                response = {"answer": str(response), "process": []}
            
            # 检查response中必要的字段
            if "answer" not in response:
                response["answer"] = ""
            if "process" not in response:
                response["process"] = []
            
            # 确保reference是字典类型
            if isinstance(reference, str):
                try:
                    reference = json.loads(reference)
                except:
                    # 解析失败时，创建一个包含原始内容的字典
                    reference = {"answer": reference, "process": []}
            
            # 确保reference包含必要的字段
            if not isinstance(reference, dict):
                reference = {"answer": str(reference), "process": []}
            
            # 确保answer和process字段存在
            if "answer" not in reference:
                reference["answer"] = ""
            if "process" not in reference:
                reference["process"] = []
            
            # 处理字段名称不匹配的情况
            if "process" in response and "process" in reference:
                response["process"] = response["process"]
            elif "process" in response and "process" not in response:
                response["process"] = response["process"]
            
            # 检查答案正确性 (A-Acc)
            answer_accuracy = self._check_answer(
                response.get("answer", ""),
                reference.get("answer", "")
            )
            
            # 检查步骤正确性 (P-Acc)
            step_similarity = 0.0
            if "process" not in response or not response.get("process"):
                step_accuracy = 0.0
            else:
                if eval_model:
                    # 使用评估模型进行步骤评估
                    step_similarity = self._evaluate_steps_with_model(
                        eval_model,
                        response.get("process", []),
                        reference.get("process", [])
                    )
                    # 根据相似度阈值确定准确率
                    step_accuracy = 1.0 if step_similarity >= 0.35 else 0.0
                else:
                    # 使用step_checker进行评估
                    step_similarity = self.step_checker.check_steps(
                        response.get("process", []),
                        reference.get("process", [])
                    )
                    step_accuracy = 1.0 if step_similarity >= 0.35 else 0.0
            
            # 计算综合准确率 (AP-Acc)
            ap_accuracy = answer_accuracy and step_accuracy
            
            return {
                "answer_accuracy": answer_accuracy,
                "step_accuracy": step_accuracy,
                "ap_accuracy": ap_accuracy,
                "step_similarity": step_similarity  # 添加步骤相似度到结果中
            }
        except Exception as e:
            import traceback
            print(f"评估错误: {e}")
            print(traceback.format_exc())
            return {
                "answer_accuracy": False,
                "step_accuracy": False,
                "ap_accuracy": False,
                "error": str(e)
            }
        
    def _check_answer(self, response_answer, reference_answer):
        """检查答案是否正确，增强多种格式处理能力并支持语义相似度比较"""
        
        # 解析和清理参考答案格式
        if isinstance(reference_answer, dict) and "answer" in reference_answer:
            # 从字典中提取answer字段
            reference_answer = reference_answer["answer"]
        
        # 处理非标准格式的JSON列表 (如 "0:"、"1:" 键)
        if isinstance(reference_answer, dict):
            try:
                # 尝试转换成列表
                reference_list = []
                # 查找数字键并按顺序排列
                keys = sorted([int(k.rstrip(':')) for k in reference_answer.keys() if k.rstrip(':').isdigit()])
                for k in keys:
                    reference_list.append(reference_answer.get(str(k) + ":", reference_answer.get(str(k), "")))
                reference_answer = reference_list
            except Exception as e:
                print(f"解析非标准JSON列表时出错: {e}")
        
        # 处理字符串格式的列表 (如 "[1, 2, 3]")
        if isinstance(reference_answer, str) and reference_answer.startswith("[") and reference_answer.endswith("]"):
            try:
                reference_answer = json.loads(reference_answer)
            except:
                pass

        # 处理模型响应中的非标准格式
        if isinstance(response_answer, dict):
            try:
                # 尝试转换成列表
                response_list = []
                # 查找数字键并按顺序排列
                keys = sorted([int(k.rstrip(':')) for k in response_answer.keys() if k.rstrip(':').isdigit()])
                for k in keys:
                    response_list.append(response_answer.get(str(k) + ":", response_answer.get(str(k), "")))
                response_answer = response_list
            except Exception as e:
                print(f"解析非标准JSON响应列表时出错: {e}")
        
        # 标准化参考答案和响应
        # 如果参考答案是列表但只有一项，且响应是字符串，将参考答案转为字符串进行比较
        if isinstance(reference_answer, list) and len(reference_answer) == 1 and isinstance(response_answer, str):
            if isinstance(reference_answer[0], str):
                reference_answer = reference_answer[0]
        
        # 如果响应是列表但只有一项，且参考答案是字符串，将响应转为字符串进行比较
        if isinstance(response_answer, list) and len(response_answer) == 1 and isinstance(reference_answer, str):
            if isinstance(response_answer[0], str):
                response_answer = response_answer[0]
        
        # 如果是字符串，进行语义比较（优先）或精确比较
        if isinstance(reference_answer, str) and isinstance(response_answer, str):
            # 对字符串进行清理
            ref_clean = reference_answer.strip().lower()
            resp_clean = response_answer.strip().lower()
            
            # 精确比较
            if resp_clean == ref_clean:
                return True
                
            # 特殊情况：yes/no问题的同义词处理
            yes_synonyms = ["yes", "是", "正确", "对", "true", "correct"]
            no_synonyms = ["no", "否", "不正确", "错", "false", "incorrect"]
            
            if ref_clean in yes_synonyms and resp_clean in yes_synonyms:
                return True
            if ref_clean in no_synonyms and resp_clean in no_synonyms:
                return True
                
            # 尝试使用相似度比较
            if hasattr(self, 'step_checker') and self.step_checker:
                similarity = self.step_checker._calculate_step_similarity(resp_clean, ref_clean)
                return similarity > 0.8  # 高阈值确保答案相似性
                
            return False
        
        # 如果是列表，比较列表内容
        elif isinstance(reference_answer, list) and isinstance(response_answer, list):
            # 如果列表为空，直接判定不等
            if not reference_answer or not response_answer:
                return reference_answer == response_answer
                
            # 对于单项列表，尝试比较第一项
            if len(reference_answer) == 1 and len(response_answer) == 1:
                return self._check_answer(response_answer[0], reference_answer[0])
                
            # 如果列表长度不同但至少有一项，考虑首项匹配
            if len(reference_answer) != len(response_answer):
                # 简单选择题答案只对比第一项
                first_ref = reference_answer[0] if reference_answer else ""
                first_resp = response_answer[0] if response_answer else ""
                
                if isinstance(first_ref, str) and isinstance(first_resp, str):
                    # 清理字符串
                    first_ref_clean = first_ref.strip().lower()
                    first_resp_clean = first_resp.strip().lower()
                    
                    # 精确比较
                    if first_resp_clean == first_ref_clean:
                        return True
                        
                    # 特殊情况：yes/no问题的同义词处理
                    yes_synonyms = ["yes", "是", "正确", "对", "true", "correct"]
                    no_synonyms = ["no", "否", "不正确", "错", "false", "incorrect"]
                    
                    if first_ref_clean in yes_synonyms and first_resp_clean in yes_synonyms:
                        return True
                    if first_ref_clean in no_synonyms and first_resp_clean in no_synonyms:
                        return True
            
            # 如果列表长度相同，逐项比较
            else:
                all_match = True
                for ref_item, resp_item in zip(reference_answer, response_answer):
                    if isinstance(ref_item, str) and isinstance(resp_item, str):
                        ref_clean = ref_item.strip().lower()
                        resp_clean = resp_item.strip().lower()
                        
                        if ref_clean != resp_clean:
                            # 特殊情况：yes/no问题的同义词处理
                            yes_synonyms = ["yes", "是", "正确", "对", "true", "correct"]
                            no_synonyms = ["no", "否", "不正确", "错", "false", "incorrect"]
                            
                            if (ref_clean in yes_synonyms and resp_clean in yes_synonyms) or \
                               (ref_clean in no_synonyms and resp_clean in no_synonyms):
                                continue
                            else:
                                all_match = False
                                break
                    elif ref_item != resp_item:
                        all_match = False
                        break
                
                return all_match
        
        # 尝试转换格式后比较
        try:
            # 尝试将字符串转换为列表进行比较
            if isinstance(reference_answer, list) and isinstance(response_answer, str):
                response_list = json.loads(response_answer)
                return self._check_answer(response_list, reference_answer)
            
            # 尝试将列表转换为字符串进行比较
            elif isinstance(reference_answer, str) and isinstance(response_answer, list):
                reference_list = json.loads(reference_answer)
                return self._check_answer(response_answer, reference_list)
        except:
            pass
        
        # 其他情况，转为字符串比较
        return str(response_answer).strip().lower() == str(reference_answer).strip().lower()
        
    def _evaluate_steps_with_model(self, eval_model, response_process, reference_process):
        """使用评估模型评估推理步骤
        Args:
            eval_model: 评估模型
            response_process: 模型输出的推理步骤
            reference_process: 参考答案的推理步骤
        Returns:
            步骤准确率(0-1)
        """
        try:
            # 处理列表类型的process
            if isinstance(response_process, list):
                response_process = " ".join(response_process)
            if isinstance(reference_process, list):
                reference_process = " ".join(reference_process)
                
            # 构建评估提示
            prompt = f"""你是一个专业的语义相似度评估专家。请评估以下两段推理步骤之间的语义相似度，并给出0到1之间的分数。

参考答案:
{reference_process}

模型输出:
{response_process}

评估指南:
1. 重点关注语义相似性和逻辑结构，而非表面词汇重叠。两段文本可以使用完全不同的词汇表达相同的思想。
2. 寻找两段文本中共享的关键概念、推理逻辑和结论，不论其表述方式如何。
3. 对于科学、技术或学术领域的问题，应特别注意专业概念的一致性，即使表达方式不同。
4. 如果模型输出捕捉到了与参考答案相同的核心推理，即使表达方式不同，也应该获得高分。
5. 不要过分关注额外细节或解释的差异，只要核心推理过程和结论相似即可。
6. 评估不应受到文本长度差异的影响 - 一个简洁的答案可能与详细的答案表达相同的核心思想。
7. 要考虑领域特定的推理特点 - 不同学科(如数学、物理、生物、历史等)有不同的推理表达方式。
8. 如果两段文本表达的结论一致，即使推理过程表述不同，也应考虑给予较高分数。

相似度评分标准:
- 0.75以上: 两段文本表达基本相同的推理思路和结论，核心概念高度一致
- 0.5-0.75: 文本的核心推理和结论相似，但表达方式或侧重点有所不同
- 0.35-0.5: 文本有一定程度的概念重叠和类似结论，但推理路径有明显差异
- 0.2-0.35: 文本在某些关键点上有相似之处，但整体推理思路不同
- 0.2以下: 文本在推理方法和结论上有本质差异

请只返回一个0到1之间的数字作为相似度评分，不需要其他解释。
例如: 0.85"""

            print(f"Prompt sent to evaluation model:\n{prompt}")

            # 使用评估模型生成评分
            if isinstance(eval_model, dict) and "api_url" in eval_model:
                # 使用API
                import requests
                headers = {
                    "Authorization": f"Bearer {eval_model['api_key']}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": eval_model["model_engine"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0
                }
                response = requests.post(eval_model["api_url"], headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    print(f"Evaluation model returned result: {content}")
                    
                    # 解析返回结果中的数字
                    import re
                    # 更全面的数字匹配模式，匹配各种格式的0-1之间的数字
                    numbers = re.findall(r"(?<![a-zA-Z0-9\.])[01](?:\.\d+)?(?![a-zA-Z0-9\.])|(?<=分数[是为:：])\s*[01](?:\.\d+)?|(?<=similarity score[:\s]*)\s*[01](?:\.\d+)?|(?<=相似度[分值得分:：])\s*[01](?:\.\d+)?", content)
                    
                    if numbers:
                        # 如果找到多个数字，选择第一个有效的0-1之间的数字
                        for num in numbers:
                            try:
                                score = float(num.strip())
                                if 0 <= score <= 1:
                                    print(f"Extracted similarity score: {score}")
                                    # 记录评估结果
                                    return max(0.0, min(1.0, score))  # 确保分数在0-1之间
                            except ValueError:
                                continue
                        
                        print("Failed to extract valid similarity score from evaluation model response")
                    else:
                        # 如果没有找到数字，尝试从文本中解析相似度描述
                        if "高度相似" in content or "very similar" in content.lower() or "highly similar" in content.lower():
                            return 0.85
                        elif "基本相似" in content or "similar" in content.lower() or "substantial" in content.lower():
                            return 0.65
                        elif "部分相似" in content or "somewhat similar" in content.lower() or "moderate" in content.lower():
                            return 0.45
                        elif "略微相似" in content or "slightly similar" in content.lower() or "少量相似" in content:
                            return 0.30
                        else:
                            print("Failed to extract similarity score from evaluation model response")
                else:
                    print(f"API call failed, status code: {response.status_code}")
                    print(f"Error message: {response.text}")
            else:
                # 使用本地模型
                from transformers import pipeline
                pipe = pipeline("text-generation", model=eval_model)
                result = pipe(prompt, max_length=50, num_return_sequences=1)
                
                content = result[0]["generated_text"]
                print(f"Evaluation model returned result: {content}")
                
                # 解析返回结果中的数字
                import re
                # 更全面的数字匹配模式，匹配各种格式的0-1之间的数字
                numbers = re.findall(r"(?<![a-zA-Z0-9\.])[01](?:\.\d+)?(?![a-zA-Z0-9\.])|(?<=分数[是为:：])\s*[01](?:\.\d+)?|(?<=similarity score[:\s]*)\s*[01](?:\.\d+)?|(?<=相似度[分值得分:：])\s*[01](?:\.\d+)?", content)
                
                if numbers:
                    # 如果找到多个数字，选择第一个有效的0-1之间的数字
                    for num in numbers:
                        try:
                            score = float(num.strip())
                            if 0 <= score <= 1:
                                print(f"Extracted similarity score: {score}")
                                return max(0.0, min(1.0, score))  # 确保分数在0-1之间
                        except ValueError:
                            continue
                    
                    print("Failed to extract valid similarity score from evaluation model response")
                else:
                    # 如果没有找到数字，尝试从文本中解析相似度描述
                    if "高度相似" in content or "very similar" in content.lower() or "highly similar" in content.lower():
                        return 0.85
                    elif "基本相似" in content or "similar" in content.lower() or "substantial" in content.lower():
                        return 0.65
                    elif "部分相似" in content or "somewhat similar" in content.lower() or "moderate" in content.lower():
                        return 0.45
                    elif "略微相似" in content or "slightly similar" in content.lower() or "少量相似" in content:
                        return 0.30
                    else:
                        print("Failed to extract similarity score from evaluation model response")
            
            print("Evaluation model did not return a valid score, falling back to step_checker")
            # 如果评估模型失败，回退到step_checker
            similarity = self.step_checker.check_steps(response_process, reference_process)
            print(f"Similarity calculated by step_checker: {similarity}")
            return similarity
            
        except Exception as e:
            import traceback
            print(f"Error evaluating steps with model: {e}")
            print(traceback.format_exc())
            # 出错时回退到step_checker
            similarity = self.step_checker.check_steps(response_process, reference_process)
            print(f"Similarity calculated by step_checker: {similarity}")
            return similarity
        
    def _save_results(self, results, task_name, model=None, eval_model=None):
        """保存评估结果
        保存格式：
        tasks/
            任务名/
                reasoning/
                    answer_accuracy_results_时间.json  # 答案准确率结果
                    step_accuracy_results_时间.json    # 步骤准确率结果
                    ap_accuracy_results_时间.json      # 综合准确率结果
        Args:
            results: 评估结果列表
            task_name: 任务名称
            model: 被评估的模型信息
            eval_model: 用于评估的模型信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = os.path.join(TASKS_DIR, task_name, "reasoning")
        os.makedirs(task_dir, exist_ok=True)
        
        # 按指标分类保存结果
        metrics = {
            "answer_accuracy": [],
            "step_accuracy": [],
            "ap_accuracy": []
        }
        
        # 整理每个指标的结果
        for result in results:
            for metric in metrics:
                metrics[metric].append({
                    "qid": result["qid"],
                    "level": result["level"],
                    "category": result["category"],
                    "score": result[metric]
                })
        
        # 提取模型信息
        model_info = {}
        if model:
            if isinstance(model, dict):
                if "api_key" in model:
                    # API模型
                    model_info = {
                        "model_type": "API",
                        "model_engine": model.get("model_engine", ""),
                        "api_url": model.get("api_url", "")
                    }
                else:
                    # Huggingface模型
                    model_info = {
                        "model_type": "huggingface/local",
                        "model_path": str(getattr(model.get("model", {}), "name_or_path", ""))
                    }
        
        # 提取评估模型信息
        eval_model_info = {}
        if eval_model:
            if isinstance(eval_model, dict):
                if "api_key" in eval_model:
                    # API模型
                    eval_model_info = {
                        "eval_model_type": "API",
                        "eval_model_engine": eval_model.get("model_engine", ""),
                        "eval_api_url": eval_model.get("api_url", "")
                    }
                else:
                    # Huggingface模型
                    eval_model_info = {
                        "eval_model_type": "huggingface/local",
                        "eval_model_path": str(getattr(eval_model.get("model", {}), "name_or_path", ""))
                    }
        
        # 分别保存各个指标的结果
        for metric, metric_results in metrics.items():
            save_path = os.path.join(task_dir, f"{metric}_results_{timestamp}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                # 合并所有信息
                result_data = {
                    "task_name": task_name,
                    "metric": metric,
                    "timestamp": timestamp,
                    "results": metric_results
                }
                
                # 添加模型信息
                if model_info:
                    result_data.update(model_info)
                
                # 添加评估模型信息
                if eval_model_info:
                    result_data.update(eval_model_info)
                
                json.dump(result_data, f, ensure_ascii=False, indent=2)