from difflib import SequenceMatcher
import numpy as np
from typing import List, Tuple, Optional, Union
import re
import string

try:
    # 尝试导入语义相似度模块，如果可用
    from sentence_transformers import SentenceTransformer
    import torch
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

class StepChecker:
    def __init__(self, nlp=None):
        """
        初始化步骤检查器
        
        Args:
            nlp: 可选的spaCy模型实例
        """
        self.semantic_threshold = 0.35  # 调整相似度阈值到更合理的水平
        # 尝试加载spaCy模型，如果没有传入
        if nlp is None:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("zh_core_web_md")
                except:
                    try:
                        self.nlp = spacy.load("en_core_web_md")
                    except:
                        print("警告：无法加载spaCy模型，将使用字符级相似度")
                        self.nlp = None
            except ImportError:
                print("警告：spaCy未安装，将使用字符级相似度")
                self.nlp = None
        else:
            self.nlp = nlp
        
        # 常见停用词列表 (中英文)
        self.stopwords = set([
            "的", "了", "和", "是", "在", "我", "有", "这", "个", "它", "们", "也", "为", "与", "但", "很", "上", "下", "中", "就", "到", "对", "能", "被", "等", "会", "把", "那", "或", "由", "从", "向", "给", "让", "所",
            "the", "is", "are", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "it", "its", "this", "that", "as", "be", "been", "being", "was", "were", "has", "have", "had"
        ])
        
    def set_semantic_threshold(self, threshold: float):
        """设置语义相似度阈值"""
        self.semantic_threshold = max(0.0, min(1.0, threshold))

    def check_steps(self, response_process, reference_process):
        """
        检查推理过程的正确性
        Args:
            response_process: 模型输出的推理过程（字符串）
            reference_process: 参考答案的推理过程（字符串）
        Returns:
            推理过程准确率(0-1)
        """
        if not response_process or not reference_process:
            return 0.0
            
        # 如果输入是列表，将其合并为字符串
        if isinstance(response_process, list):
            response_process = " ".join(response_process)
        if isinstance(reference_process, list):
            reference_process = " ".join(reference_process)
            
        # 直接计算两个推理过程的相似度
        return self._calculate_step_similarity(response_process, reference_process)
    
    def _extract_keywords(self, text):
        """提取文本中的关键词（去除停用词和标点）"""
        # 清理文本：移除标点，转小写
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # 扩展停用词列表，包括更多常见词汇
        extended_stopwords = self.stopwords.union({
            # 英文常见助动词、介词和连词等
            "can", "could", "will", "would", "should", "may", "might", "must", 
            "about", "above", "across", "after", "against", "along", "among", "around",
            "because", "before", "behind", "below", "beneath", "beside", "between",
            "beyond", "during", "except", "inside", "outside", "through", "throughout",
            "therefore", "however", "although", "though", "since", "while", "where",
            
            # 中文常见虚词
            "这个", "那个", "因为", "所以", "如果", "虽然", "但是", "而且", "以及",
            "可以", "应该", "一个", "没有", "什么", "如何", "为什么", "怎么", "哪里"
        })
        
        # 分词（简单按空格和中文字符分词）
        words = []
        for word in re.findall(r'\w+', text):
            words.append(word)
        
        # 提取中文词组（考虑2-4个字符的词组）
        chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
        if len(chinese_chars) > 1:
            # 添加中文单字
            for char in chinese_chars:
                words.append(char)
            
            # 添加中文2-4字词组（滑动窗口）
            for i in range(len(chinese_chars) - 1):
                # 双字词
                if i < len(chinese_chars) - 1:
                    words.append(chinese_chars[i] + chinese_chars[i+1])
                # 三字词
                if i < len(chinese_chars) - 2:
                    words.append(chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2])
                # 四字词
                if i < len(chinese_chars) - 3:
                    words.append(chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2] + chinese_chars[i+3])
        
        # 过滤停用词，并保留长度>1的词
        keywords = [word for word in words if word not in extended_stopwords and len(word) > 1]
        
        # 提取可能的术语特征（数字+字母组合，通常是专业术语）
        number_letter_pattern = re.compile(r'[a-z]+[0-9]+[a-z]*|[a-z]*[0-9]+[a-z]+')
        for match in number_letter_pattern.finditer(text):
            term = match.group()
            if term not in keywords:
                keywords.append(term)
                
        return keywords
    
    def _calculate_step_similarity(self, text1: str, text2: str) -> float:
        """计算两个推理过程之间的相似度"""
        # 如果两个文本完全相同，直接返回1.0
        if text1 == text2:
            return 1.0
            
        # 处理中英文yes/no问题的相似度
        yes_synonyms = ["yes", "是", "正确", "对", "true", "correct"]
        no_synonyms = ["no", "否", "不正确", "错", "false", "incorrect"]
        
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # 如果两个文本都是yes或都是no的同义词，返回高相似度
        if (text1_lower in yes_synonyms and text2_lower in yes_synonyms) or \
           (text1_lower in no_synonyms and text2_lower in no_synonyms):
            return 0.95
        
        # 关键词匹配
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        # 如果有关键词
        if keywords1 and keywords2:
            # 计算关键词交集
            common_keywords = set(keywords1).intersection(set(keywords2))
            # 关键词匹配率
            keyword_match_rate = len(common_keywords) / max(len(keywords1), len(keywords2))
            
            # 增强关键词匹配的权重
            keyword_bonus = 0.0
            if keyword_match_rate > 0.5:
                keyword_bonus = 0.3  # 提高关键词匹配良好时的加分
            elif keyword_match_rate > 0.3:
                keyword_bonus = 0.2  # 关键词匹配一般时也给予一定加分
            elif keyword_match_rate > 0.1:
                keyword_bonus = 0.1  # 即使只有少量关键词匹配也给予小幅加分
        else:
            keyword_match_rate = 0.0
            keyword_bonus = 0.0
            
        # 使用语义相似度（如果可用）
        semantic_similarity = self._calculate_semantic_similarity(text1, text2)
        if semantic_similarity is not None:
            # 结合关键词匹配结果提高分数
            adjusted_similarity = semantic_similarity + keyword_bonus
            return min(1.0, adjusted_similarity)  # 确保不超过1.0
        
        # 回退到字符级相似度
        char_similarity = self._calculate_char_similarity(text1, text2)
        
        # 对于长文本，使用更宽松的评估标准
        if len(text1) > 50 and len(text2) > 50:
            # 降低字符相似度阈值，增加加分
            if char_similarity > 0.5:  # 降低阈值
                return min(1.0, char_similarity + keyword_bonus + 0.1)  # 增加额外加分
            return char_similarity + keyword_bonus
                
        return min(1.0, char_similarity + keyword_bonus)
        
    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的字符级相似度"""
        base_similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # 动态识别共同专业词汇而非使用固定列表
        # 提取两段文本中的关键词
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        # 找出共同的词汇
        common_words = set(keywords1).intersection(set(keywords2))
        
        # 计算共同词汇的权重
        term_bonus = 0.0
        if common_words:
            # 越长的词越可能是专业术语，给予更高权重
            for word in common_words:
                if len(word) >= 4:  # 长词更可能是专业术语
                    term_bonus += 0.015  # 每个长词加0.015分
                elif len(word) >= 2:
                    term_bonus += 0.01   # 短词加0.01分
            
            # 术语奖励上限为0.3
            term_bonus = min(0.3, term_bonus)
        
        # 加成后的总相似度
        enhanced_similarity = min(1.0, base_similarity + term_bonus)
        return enhanced_similarity
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        计算两个文本的语义相似度
        如果语义模型不可用，返回None
        """
        if not self.nlp:
            return None
            
        try:
            # 计算嵌入
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # 计算余弦相似度
            cosine_sim = doc1.similarity(doc2)
            
            return max(0.0, cosine_sim)  # 确保非负
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return None 