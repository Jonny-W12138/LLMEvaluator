"""
Module for generating model responses for retrieval tasks
"""

import json
import time
import logging
import torch
from typing import Dict, List, Any, Optional, Union, Callable
from .answer_extractor import AnswerExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalResponseGenerator:
    def __init__(self, model_api=None, prompt_template=None):
        """
        Initialize the retrieval response generator
        
        Args:
            model_api: Model API interface (optional, can be set later)
            prompt_template: Optional prompt template
        """
        self.model_api = model_api
        self.default_template = """
You are a professional assistant who needs to answer questions based on given context.
Please read the following context carefully and then answer the question. 

Context:
{context}

Question:
{question}

Options:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

First, provide your answer as a single letter (A, B, C, or D).
Then, on a new line, briefly explain your reasoning (1-2 sentences).

Your response:
"""
        self.prompt_template = prompt_template if prompt_template else self.default_template
        self.answer_extractor = AnswerExtractor()
    
    def generate_prompt(self, item: Dict[str, Any]) -> str:
        """
        Generate prompt from data item
        
        Args:
            item: Data item containing question, options and context
            
        Returns:
            Formatted prompt string
        """
        # Handle variable field names (some datasets use "options" field)
        options = {}
        if "options" in item:
            options = item["options"]
        else:
            for letter in ["A", "B", "C", "D"]:
                key = f"choice_{letter}"
                if key in item:
                    options[letter] = item[key]
                else:
                    options[letter] = f"Option {letter} (missing)"
                    logger.warning(f"Missing option {letter} in item: {item.get('_id', 'unknown')}")
        
        try:
            return self.prompt_template.format(
                context=item.get("context", ""),
                question=item.get("question", ""),
                choice_A=options.get("A", item.get("choice_A", "")),
                choice_B=options.get("B", item.get("choice_B", "")),
                choice_C=options.get("C", item.get("choice_C", "")),
                choice_D=options.get("D", item.get("choice_D", ""))
            )
        except KeyError as e:
            logger.error(f"Failed to format prompt: {e}")
            raise ValueError(f"Missing required field in prompt template: {e}")
    
    def _generate_with_api(self, model_info: Dict[str, Any], prompt: str) -> str:
        """使用API生成响应"""
        try:
            # 记录API调用
            logger.info(f"Calling API with model: {model_info.get('model_engine', 'unknown')}")
            
            # 提取API配置
            api_url = model_info.get("api_url", "")
            api_key = model_info.get("api_key", "")
            model_engine = model_info.get("model_engine", "")
            
            # 1. 标准OpenAI兼容API
            if "openai.com" in api_url.lower() or "compatible" in api_url.lower():
                try:
                    import openai
                    # 设置客户端
                    client = openai.OpenAI(api_key=api_key, base_url=api_url)
                    
                    # 调用API
                    response = client.chat.completions.create(
                        model=model_engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=200
                    )
                    
                    # 提取响应文本
                    return response.choices[0].message.content.strip()
                except ImportError:
                    # 如果没有openai包，使用请求库
                    logger.warning("OpenAI package not found, using requests instead")
                    import requests
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": model_engine,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200
                    }
                    response = requests.post(api_url, headers=headers, json=data)
                    response.raise_for_status()
                    response_json = response.json()
                    
                    # 尝试提取响应内容
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        if "message" in response_json["choices"][0]:
                            return response_json["choices"][0]["message"]["content"].strip()
                        elif "text" in response_json["choices"][0]:
                            return response_json["choices"][0]["text"].strip()
                    
                    # 如果无法解析，返回整个响应
                    return f"无法解析的响应: {str(response_json)[:200]}"
            
            # 2. 百度文心或国内模型API
            elif "baidu" in api_url.lower() or "wenxin" in api_url.lower():
                import requests
                headers = {
                    "Content-Type": "application/json"
                }
                
                # 百度API通常需要不同的认证方式
                if "access_token" not in api_url:
                    # 如果URL中没有access_token，则添加到headers
                    headers["Authorization"] = f"Bearer {api_key}"
                
                data = {
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # 某些API需要指定模型名称
                if model_engine:
                    data["model"] = model_engine
                
                response = requests.post(api_url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                # 百度文心API的响应格式
                if "result" in response_json:
                    return response_json["result"].strip()
                elif "response" in response_json:
                    return response_json["response"].strip()
                elif "content" in response_json:
                    return response_json["content"].strip()
                elif "output" in response_json:
                    return response_json["output"].strip()
                elif "generated_text" in response_json:
                    return response_json["generated_text"].strip()
                
                # 无法解析的情况
                return f"无法解析的国内模型API响应: {str(response_json)[:200]}"
            
            # 3. 通用API处理 - 尝试多种常见格式
            else:
                import requests
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # 构建通用请求体
                data = {
                    "prompt": prompt,  # 有些API用prompt
                    "messages": [{"role": "user", "content": prompt}],  # 有些用messages
                    "model": model_engine,
                    "temperature": 0.1,
                    "max_tokens": 200
                }
                
                # 发送请求
                response = requests.post(api_url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                # 尝试多种常见响应格式
                possible_paths = [
                    # OpenAI格式
                    lambda x: x["choices"][0]["message"]["content"],
                    lambda x: x["choices"][0]["text"],
                    # Anthropic格式
                    lambda x: x["completion"],
                    lambda x: x["content"][0]["text"],
                    # 通义/智谱等
                    lambda x: x["output"]["text"],
                    lambda x: x["response"],
                    lambda x: x["result"],
                    lambda x: x["generated_text"],
                    # 通用路径
                    lambda x: x["output"],
                    lambda x: x["answer"],
                    lambda x: x["text"],
                    lambda x: x["content"],
                ]
                
                # 尝试所有可能的路径
                for path_func in possible_paths:
                    try:
                        result = path_func(response_json)
                        if result and isinstance(result, str):
                            return result.strip()
                    except (KeyError, IndexError, TypeError):
                        continue
                
                # 如果所有路径都失败，返回原始JSON的前200个字符
                logger.warning(f"Failed to parse API response: {str(response_json)[:200]}")
                return f"API响应无法解析: {str(response_json)[:200]}"
        
        except Exception as e:
            logger.error(f"API调用错误: {str(e)}")
            # 返回错误信息而不是引发异常，这样评估可以继续进行
            return f"API调用失败: {str(e)}"
    
    def generate_response(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate model response for a retrieval question
        
        Args:
            item: Data item containing question, options and context
            
        Returns:
            Dictionary with original question, model response and response time
        """
        prompt = self.generate_prompt(item)
        logger.info(f"Generated prompt for item: {item.get('_id', 'unknown')}")
        
        # Get item ID from either _id or item_id field
        item_id = item.get("_id", item.get("item_id", ""))
        
        # Get options in consistent format
        options = {}
        if "options" in item:
            options = item["options"]
        else:
            for letter in ["A", "B", "C", "D"]:
                options[letter] = item.get(f"choice_{letter}", "")
        
        try:
            # 使用模型生成回答
            start_time = time.time()
            
            if self.model_api:
                # 检查模型类型并调用适当的生成方法
                if isinstance(self.model_api, dict) and "api_key" in self.model_api:
                    # API模型
                    response = self._generate_with_api(self.model_api, prompt)
                elif hasattr(self.model_api, "generate"):
                    # 有generate方法的模型接口
                    response = self.model_api.generate(prompt)
                else:
                    # 未知模型类型
                    response = "模型接口类型不支持"
                    raise ValueError(f"Unsupported model type: {type(self.model_api)}")
            else:
                response = "未初始化模型API"
                raise ValueError("Model API not initialized")
                
            # 确保响应不为None
            if response is None:
                response = "模型未返回任何响应"
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Log successful response
            logger.info(f"Generated response for item {item_id} in {response_time:.2f}s")
            
            response_text = str(response)
            reasoning = self.answer_extractor.extract_reasoning(response_text)
            
            return {
                "item_id": item_id,
                "domain": item.get("domain", ""),
                "sub_domain": item.get("sub_domain", ""),
                "difficulty": item.get("difficulty", ""),
                "length": item.get("length", ""),
                "question": item.get("question", ""),
                "options": options,
                "ground_truth": item.get("answer", ""),
                "model_response": response_text,  # 确保转换为字符串
                "model_reasoning": reasoning,  # 添加模型理由字段
                "response_time": response_time
            }
        except Exception as e:
            logger.error(f"Error generating response for item {item_id}: {str(e)}")
            return {
                "item_id": item_id,
                "question": item.get("question", ""),
                "options": options,
                "ground_truth": item.get("answer", ""),
                "error": str(e),
                "response_time": 0
            }
    
    def batch_generate(self, items: List[Dict[str, Any]], callback: Callable = None) -> List[Dict[str, Any]]:
        """
        Generate model responses in batch
        
        Args:
            items: List of retrieval questions
            callback: Optional callback function to report progress
            
        Returns:
            List of response results
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            try:
                logger.info(f"Processing question {i+1}/{total}...")
                if callback:
                    callback(i/total, f"Processing question {i+1}/{total}...")
                
                result = self.generate_response(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                # Add error information
                results.append({
                    "item_id": item.get("_id", item.get("item_id", "")),
                    "error": str(e),
                    "question": item.get("question", ""),
                    "ground_truth": item.get("answer", "")
                })
            
            # Final progress update
            if callback and i == total-1:
                callback(1.0, "Completed processing all questions")
        
        return results 