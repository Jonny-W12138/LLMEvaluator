from transformers import (
    AutoModelForCausalLM, 
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import openai
from typing import Union, Dict, Tuple

def load_model(model_path_or_api: Union[str, Dict]) -> Union[Dict[str, Union[PreTrainedModel, PreTrainedTokenizer]], 'OpenAI']:
    """加载模型
    Args:
        model_path_or_api: 模型路径或API配置
    Returns:
        加载的模型或API客户端
    """
    if isinstance(model_path_or_api, dict):
        # API模式
        openai.api_key = model_path_or_api["api_key"]
        return openai
    else:
        # HuggingFace模式
        try:
            # 1. 首先尝试加载分词器
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_api,
                trust_remote_code=True  # 允许执行远程代码
            )
            
            # 2. 尝试不同的模型类型
            try:
                # 首先尝试 CausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_or_api,
                    trust_remote_code=True
                )
            except ValueError:
                try:
                    # 如果失败，尝试通用模型类型
                    model = AutoModel.from_pretrained(
                        model_path_or_api,
                        trust_remote_code=True
                    )
                except Exception as e:
                    raise ValueError(f"无法加载模型 {model_path_or_api}: {str(e)}")
            
            return {
                "model": model,
                "tokenizer": tokenizer
            }
            
        except Exception as e:
            raise ValueError(f"加载模型失败: {str(e)}\n"
                           f"请确保模型路径正确且模型文件完整。")