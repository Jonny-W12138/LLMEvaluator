import json
import torch
from typing import Dict, List, Union, Optional

class ReasoningResponseGenerator:
    def __init__(self):
        self.prompt_template = """\
问题：{contexts}
{examples_text}
请根据问题进行推理并给出答案。
请按照以下JSON格式输出：
{{
    "answer": "最终答案",
    "process": "完整的推理过程，解释如何得出答案",
}}
JSON输出："""

    def generate(self, model: Union[Dict, 'OpenAI'], 
                input_text: str, 
                examples: Optional[List[str]] = None,
                max_tokens: int = 400,
                temperature: float = 0.0,
                top_p: float = 1.0) -> Dict:
        """生成推理响应
        Args:
            model: 使用的模型(支持HuggingFace模型或API模型)
            input_text: 输入文本
            examples: few-shot示例列表（字符串列表）
            max_tokens: 最大生成长度，默认400
            temperature: 温度参数，默认0.0表示最确定的输出
            top_p: 核采样参数，默认1.0表示考虑所有可能性
        Returns:
            JSON格式的推理过程和答案
        """
        # 构建few-shot示例
        examples_text = ""
        if examples:
            # 直接处理字符串列表
            for example in examples:
                examples_text += f"示例：\n{example}\n\n"
        
        # 构建完整提示
        prompt = self.prompt_template.format(
            contexts=input_text,
            examples_text=examples_text
        )
        
        # 根据模型类型调用不同的生成方法
        if isinstance(model, dict) and "api_key" in model:
            # API模式
            return self._generate_with_api(
                model, prompt, max_tokens, temperature, top_p
            )
        else:
            # HuggingFace模式
            return self._generate_with_model(
                model, prompt, max_tokens, temperature, top_p
            )
            
    def _generate_with_model(self, model: Dict, 
                           prompt: str,
                           max_tokens: int,
                           temperature: float,
                           top_p: float) -> Dict:
        """使用HuggingFace模型生成"""
        # 将输入转换为tensor
        inputs = model["tokenizer"](prompt, return_tensors="pt")
        
        # 如果有GPU则使用
        if torch.cuda.is_available():
            model["model"].cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = model["model"].generate(
                inputs["input_ids"],
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=model["tokenizer"].eos_token_id,
                do_sample=(temperature > 0)
            )
        
        # 解码输出
        response = model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        
        # 解析JSON响应
        return self._parse_response(response)
        
    def _generate_with_api(self, model_info, prompt, max_tokens, temperature, top_p):
        """使用API生成响应"""
        try:
            import openai
            
            # 设置API参数
            openai.api_key = model_info.get("api_key", "")
            openai.api_base = model_info.get("api_url", "https://api.openai.com/v1")
            
            # 使用新版API接口
            client = openai.OpenAI(
                api_key=model_info.get("api_key", ""),
                base_url=model_info.get("api_url", "https://api.openai.com/v1")
            )
            
            # 发送请求
            response = client.chat.completions.create(
                model=model_info.get("model_engine", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # 返回生成的文本
            return response.choices[0].message.content
        except Exception as e:
            # 返回错误信息
            return f"API错误: \n\n{str(e)}"
        
    def _parse_response(self, response: str) -> Dict:
        """解析响应确保符合JSON格式"""
        try:
            # 尝试提取JSON部分
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # 验证必要字段
                if "process" in result and isinstance(result["process"], list) and "answer" in result:
                    return result
        except:
            pass
        
        # 解析失败时返回空结果，确保是字典类型
        # 尝试从文本中提取答案
        try:
            # 查找可能的答案
            answer = ""
            if "答案是" in response or "密码是" in response:
                answer_start = max(response.find("答案是"), response.find("密码是"))
                if answer_start > -1:
                    answer_end = response.find("。", answer_start)
                    if answer_end > -1:
                        answer = response[answer_start+3:answer_end].strip()
        except:
            pass
        
        return {
            "process": [],
            "answer": answer if answer else ""
        } 