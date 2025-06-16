"""
Module for evaluating model retrieval capabilities
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .response_generate import RetrievalResponseGenerator
from .answer_extractor import AnswerExtractor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    def __init__(self, model_config, prompt_template=None, custom_patterns=None):
        """
        Initialize retrieval capability evaluator
        
        Args:
            model_config: Model configuration dictionary
            prompt_template: Optional prompt template
            custom_patterns: Optional custom extraction patterns
        """
        try:
            # 直接使用model_config作为模型API
            self.model_api = model_config
            self.model_config = model_config
            
            # Initialize components
            self.response_generator = RetrievalResponseGenerator(self.model_api, prompt_template)
            self.answer_extractor = AnswerExtractor(custom_patterns)
            
            logger.info("Retrieval evaluator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval evaluator: {str(e)}")
            raise RuntimeError(f"Failed to initialize retrieval evaluator: {str(e)}")
    
    def evaluate(self, dataset: List[Dict[str, Any]], output_dir: str = None, progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Evaluate model retrieval capability on a dataset
        
        Args:
            dataset: List of data items
            output_dir: Output directory for results
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            Evaluation results summary
        """
        start_time = time.time()
        logger.info(f"Starting retrieval evaluation on dataset with {len(dataset)} items")
        
        try:
            # Validate dataset
            if len(dataset) == 0:
                logger.error("Empty dataset provided")
                return {
                    "error": "Empty dataset provided",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Report progress
            if progress_callback:
                progress_callback(0.05, "Generating responses...")
            
            # Generate responses
            logger.info("Generating model responses")
            responses = self.response_generator.batch_generate(dataset, callback=progress_callback)
            
            # Extract answers from responses
            if progress_callback:
                progress_callback(0.7, "Extracting answers and evaluating...")
                
            logger.info("Extracting answers from responses")
            for i, response in enumerate(responses):
                if "error" not in response:
                    try:
                        # Extract the model's answer from its response
                        extracted = self.answer_extractor.extract_answer(response["model_response"])
                        response["extracted_answer"] = extracted
                        
                        # Check if the extracted answer matches the ground truth
                        is_correct = False
                        if extracted:
                            ground_truth = response["ground_truth"].strip().upper()
                            extracted = extracted.strip().upper()
                            is_correct = extracted == ground_truth
                        
                        response["is_correct"] = is_correct
                    except Exception as e:
                        logger.error(f"Error extracting answer: {str(e)}")
                        response["extracted_answer"] = None
                        response["is_correct"] = False
            
            # Calculate metrics
            if progress_callback:
                progress_callback(0.9, "Calculating metrics...")
                
            logger.info("Calculating evaluation metrics")
            metrics = self._calculate_metrics(responses)
            
            # Prepare result
            result = {
                "metrics": metrics,
                "responses": responses,
                "timestamp": datetime.now().isoformat(),
                "model_config": {k: v for k, v in self.model_config.items() if k != "api_key"} if isinstance(self.model_config, dict) else {"model_type": "custom"}
            }
            
            # Save result to file if output directory is provided
            if output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = os.path.join(output_dir, f"retrieval_results_{timestamp}.json")
                
                try:
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Results saved to {result_file}")
                except Exception as e:
                    logger.error(f"Error saving results: {str(e)}")
            
            # Record end time
            end_time = time.time()
            evaluation_time = end_time - start_time
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            
            # Final progress update
            if progress_callback:
                progress_callback(1.0, "Evaluation complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Traceback: {error_trace}")
            
            # Report error to callback
            if progress_callback:
                progress_callback(1.0, f"Evaluation failed: {str(e)}")
            
            # Return error information
            return {
                "error": str(e),
                "traceback": error_trace,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from responses
        
        Args:
            responses: List of model responses
            
        Returns:
            Dictionary of metrics
        """
        # Filter out responses with errors
        valid_responses = [r for r in responses if "error" not in r]
        total = len(valid_responses)
        
        if total == 0:
            logger.warning("No valid responses to calculate metrics")
            return {
                "total": 0,
                "answered": 0,
                "correct": 0,
                "accuracy": 0,
                "answer_rate": 0
            }
        
        # Count responses where answer was extracted
        answered = sum(1 for r in valid_responses if r.get("extracted_answer") is not None)
        
        # Count correct answers
        correct = sum(1 for r in valid_responses if r.get("is_correct", False))
        
        # Calculate metrics by difficulty
        by_difficulty = {}
        for level in ["easy", "medium", "hard"]:
            level_responses = [r for r in valid_responses if r.get("difficulty", "").lower() == level]
            level_total = len(level_responses)
            if level_total > 0:
                level_correct = sum(1 for r in level_responses if r.get("is_correct", False))
                by_difficulty[level] = {
                    "total": level_total,
                    "correct": level_correct,
                    "accuracy": level_correct / level_total
                }
        
        # Calculate metrics by context length
        by_length = {}
        for length in ["short", "medium", "long"]:
            length_responses = [r for r in valid_responses if r.get("length", "").lower() == length]
            length_total = len(length_responses)
            if length_total > 0:
                length_correct = sum(1 for r in length_responses if r.get("is_correct", False))
                by_length[length] = {
                    "total": length_total,
                    "correct": length_correct,
                    "accuracy": length_correct / length_total
                }
        
        # Calculate metrics by domain
        by_domain = {}
        domains = set(r.get("domain", "") for r in valid_responses if r.get("domain", ""))
        for domain in domains:
            domain_responses = [r for r in valid_responses if r.get("domain", "") == domain]
            domain_total = len(domain_responses)
            if domain_total > 0:
                domain_correct = sum(1 for r in domain_responses if r.get("is_correct", False))
                by_domain[domain] = {
                    "total": domain_total,
                    "correct": domain_correct,
                    "accuracy": domain_correct / domain_total
                }
        
        # Calculate average response time
        response_times = [r.get("response_time", 0) for r in valid_responses if "response_time" in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Collect error rate by type
        error_analysis = {}
        error_responses = [r for r in valid_responses if not r.get("is_correct", False)]
        extraction_failures = sum(1 for r in error_responses if r.get("extracted_answer") is None)
        wrong_extractions = sum(1 for r in error_responses if r.get("extracted_answer") is not None)
        
        if len(error_responses) > 0:
            error_analysis = {
                "total_errors": len(error_responses),
                "extraction_failures": extraction_failures,
                "extraction_failure_rate": extraction_failures / len(error_responses) if len(error_responses) > 0 else 0,
                "wrong_extractions": wrong_extractions,
                "wrong_extraction_rate": wrong_extractions / len(error_responses) if len(error_responses) > 0 else 0
            }
        
        return {
            "total": total,
            "answered": answered,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "answer_rate": answered / total if total > 0 else 0,
            "by_difficulty": by_difficulty,
            "by_length": by_length,
            "by_domain": by_domain,
            "avg_response_time": avg_response_time,
            "error_analysis": error_analysis
        } 