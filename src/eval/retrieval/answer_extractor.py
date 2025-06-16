"""
Module for extracting answer options from model responses
"""

import re
import logging
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnswerExtractor:
    def __init__(self, custom_patterns=None):
        """
        Initialize answer extractor
        
        Args:
            custom_patterns: Optional list of custom regex patterns to use
        """
        # Default patterns for matching options A/B/C/D
        self.default_patterns = [
            # English patterns
            r'(?:answer|option|choice|select|choose|the answer is)[\s]*[:：]?[\s]*([ABCD])',
            r'([ABCD])[\s]*(?:is the answer|is correct|is the correct answer)',
            r'^[\s]*([ABCD])[\s]*$',
            r'I believe (?:the answer is|that) [\s]*([ABCD])',
            
            # Chinese patterns
            r'(?:选择|选项|答案|回答|答案是|我选择|正确答案是|应该选择)[\s]*[:：]?[\s]*([ABCD])',
            r'([ABCD])[\s]*(?:是正确的|是正确答案)',
            r'我认为答案是[\s]*([ABCD])'
        ]
        
        # Use custom patterns if provided, otherwise use defaults
        self.option_patterns = custom_patterns if custom_patterns else self.default_patterns
        logger.info(f"Initialized answer extractor with {len(self.option_patterns)} patterns")
    
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer option from model response
        
        Args:
            response: Model text response
            
        Returns:
            Extracted option (A/B/C/D) or None if not extractable
        """
        if not response:
            logger.warning("Empty response received")
            return None
            
        # Log the response for debugging
        logger.debug(f"Extracting answer from response: {response[:100]}...")
        
        # Try using patterns to match answer
        for i, pattern in enumerate(self.option_patterns):
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()  # Ensure uppercase
                logger.debug(f"Pattern {i+1} matched: {answer}")
                return answer
        
        # If no pattern matches, look for standalone A/B/C/D
        options = re.findall(r'\b([ABCD])\b', response, re.IGNORECASE)
        if options:
            # Return first occurrence
            answer = options[0].upper()
            logger.debug(f"Standalone option found: {answer}")
            return answer
        
        logger.warning(f"Could not extract answer from response: {response[:100]}...")
        return None
    
    def extract_reasoning(self, text: str) -> str:
        """
        Extract reasoning from model response
        
        Args:
            text: Model response text
            
        Returns:
            Extracted reasoning
        """
        # 首先尝试提取答案字母，理由通常紧随其后
        if not text:
            return ""
        
        # 提取第一个选项字母后的内容作为理由
        import re
        
        # 尝试多种模式匹配理由
        patterns = [
            # 答案字母后接换行再接理由
            r'(?:^|\s)[ABCD][\s\.\:\n]+(.+)$',
            # 答案是X，因为...
            r'(?:答案是|选择|选项)[ABCD][\s\,\，\.。\:\：]+(.+)$',
            # 我选择X，因为...
            r'(?:我选择|我的答案是)[ABCD][\s\,\，\.。\:\：]+(.+)$'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                reasoning = matches.group(1).strip()
                # 限制长度，避免过长
                if len(reasoning) > 200:
                    reasoning = reasoning[:197] + "..."
                return reasoning
            
        # 如果无法提取，返回整个响应（除了可能的选项字母）
        # 首先尝试移除开头的选项字母
        cleaned = re.sub(r'^[ABCD][\s\.\:\n]+', '', text.strip())
        if len(cleaned) > 200:
            cleaned = cleaned[:197] + "..."
        return cleaned
    
    def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete response data, adding extracted answer and correctness assessment
        
        Args:
            response_data: Response data dictionary
            
        Returns:
            Response data with added extraction and correctness info
        """
        model_response = response_data.get("model_response", "")
        ground_truth = response_data.get("ground_truth", "")
        item_id = response_data.get("item_id", "unknown")
        
        logger.info(f"Processing response for item: {item_id}")
        
        extracted_answer = self.extract_answer(model_response)
        
        # Create a copy to avoid modifying the original
        result = response_data.copy()
        result["extracted_answer"] = extracted_answer
        
        # Determine correctness (case-insensitive comparison)
        is_correct = False
        if extracted_answer and ground_truth:
            is_correct = extracted_answer.upper() == ground_truth.upper()
        
        result["is_correct"] = is_correct
        
        if extracted_answer:
            logger.info(f"Item {item_id}: Extracted '{extracted_answer}', correct: {is_correct}")
        else:
            logger.warning(f"Item {item_id}: Failed to extract answer")
        
        return result
    
    def batch_process(self, responses_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process response data in batch
        
        Args:
            responses_data: List of response data
            
        Returns:
            Processed response data list
        """
        logger.info(f"Batch processing {len(responses_data)} responses")
        return [self.process_response(response) for response in responses_data] 