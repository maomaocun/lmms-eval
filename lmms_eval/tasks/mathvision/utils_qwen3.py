"""
Qwen3-VL MathVision Evaluation Implementation

This module provides Qwen3-VL optimized evaluation for the MathVision benchmark.

Key features:
1. Optimized for Qwen3-VL with official image resolution settings
2. Support for min_pixels/max_pixels parameters
3. Compatible with MathVision standard and reasoning evaluation

Reference:
- MathVision: https://mathvision-cuhk.github.io/
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
# compute_score imported inside functions to avoid ANTLR version issues at module load

# Import original eval_utils functions
try:
    from lmms_eval.tasks.mathvision.eval_utils import (
        find_math_answer,
        is_equal,
        is_number,
    )
except ImportError as e:
    eval_logger.warning(f"Error importing eval_utils: {e}")
    # Define fallback functions if import fails
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def find_math_answer(s):
        return s.strip()
    
    def is_equal(a, b):
        return a.strip() == b.strip()


# ==================== Standard Version Functions ====================

def mathvision_doc_to_visual_qwen3(doc: Dict) -> List:
    """
    Extract visual elements from document for Qwen3-VL.
    
    Args:
        doc: Document containing image data
        
    Returns:
        List of RGB images
    """
    decoded_image = doc.get("decoded_image")
    if decoded_image:
        return [decoded_image.convert("RGB")]
    return []


def mathvision_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Convert document to text prompt for Qwen3-VL.
    
    Args:
        doc: Document containing question and options
        lmms_eval_specific_kwargs: Optional kwargs for customization
        
    Returns:
        Formatted question text
    """
    question = doc.get("question", "")
    choices = doc.get("options", [])
    len_choices = len(choices)
    
    if len_choices > 0:
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    else:
        choices_str = ""
    
    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs.get("mc_prompt", "")
    
    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    
    return query_prompt


def mathvision_qwen3_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Process evaluation results for Qwen3-VL standard version.
    
    This function uses the same evaluation logic as the original MathVision
    but is compatible with Qwen3-VL output format.
    
    Args:
        doc: Ground truth document
        results: List of model predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    correct_list = []
    
    for pred in results:
        model_answer = pred.strip()
        gt_answer = str(doc.get("answer", ""))
        
        choices = doc.get("options", [])
        if len(choices) > 0:
            gt_answer_value = choices[ord(gt_answer) - ord("A")] if len(gt_answer) == 1 and gt_answer in "ABCDE" else ""
        else:
            gt_answer_value = ""
        
        # Parse multiple choice answers
        for c in "ABCDE":
            if model_answer.endswith(f" {c}.") or \
               model_answer.endswith(f" ({c}).") or \
               model_answer.startswith(f"{c}\n") or \
               model_answer.startswith(f"({c})\n") or \
               model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        
        if is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        
        if "oxed{" not in model_answer:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                flag = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]
        
        model_answer = (
            find_math_answer(model_answer)
            .replace("(a)", "a")
            .replace("(b)", "b")
            .replace("(c)", "c")
            .replace("(d)", "d")
            .replace("(e)", "e")
            .replace("{a}", "a")
            .replace("{b}", "b")
            .replace("{c}", "c")
            .replace("{d}", "d")
            .replace("{e}", "e")
            .rstrip(".")
            .lstrip(":")
            .strip()
        )
        
        correct = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
        correct_list.append(correct)
    
    return {
        "mathvision_qwen3_eval": {
            "response": results,
            "scores": correct_list,
        },
        "score": float(correct_list[0]) if correct_list else 0.0,
    }


def mathvision_aggregate_results_qwen3(results: List[Dict]) -> float:
    """
    Aggregate MathVision Qwen3 evaluation results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Accuracy as percentage
    """
    if not results:
        return 0.0
    
    total = len(results)
    correct = sum(1 for result in results if result.get("scores", [False])[0])
    accuracy = round(correct / total * 100, 2)
    return accuracy


# ==================== Reasoning Version Functions ====================

def mathvision_reason_doc_to_visual_qwen3(doc: Dict) -> List:
    """
    Extract visual elements from document for Qwen3-VL reasoning version.
    
    Args:
        doc: Document containing image data
        
    Returns:
        List of RGB images
    """
    decoded_image = doc.get("decoded_image")
    if decoded_image:
        return [decoded_image.convert("RGB")]
    return []


def mathvision_reason_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Convert document to text prompt for Qwen3-VL reasoning version.
    
    Args:
        doc: Document containing question and options
        lmms_eval_specific_kwargs: Optional kwargs for customization
        
    Returns:
        Formatted question text
    """
    question = doc.get("question", "")
    choices = doc.get("options", [])
    len_choices = len(choices)
    
    if len_choices > 0:
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    else:
        choices_str = ""
    
    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs.get("mc_prompt", "")
    
    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    
    return query_prompt


def mathvision_reason_doc_to_messages_qwen3(
    doc: Dict, 
    lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """
    Convert document to Qwen3-VL message format for reasoning tasks.
    
    This function creates messages optimized for Qwen3-VL with:
    - Official image resolution settings (min_pixels, max_pixels)
    - System prompt for reasoning tasks
    - Proper message structure
    
    Args:
        doc: Document containing question, options, and image
        lmms_eval_specific_kwargs: Optional kwargs including format, system_prompt, 
                                   min_pixels, max_pixels
        
    Returns:
        List of message dictionaries for Qwen3-VL
    """
    # Get config defaults if not provided
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    # Extract configuration
    system_prompt = lmms_eval_specific_kwargs.get(
        "system_prompt",
        "You are a helpful assistant. When the user asks a question, your response must include two parts: "
        "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. "
        "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
    )
    
    # Official image resolution settings (match Qwen3-VL official repo)
    min_pixels = lmms_eval_specific_kwargs.get("min_pixels", 1280 * 28 * 28)  # ~1M pixels
    max_pixels = lmms_eval_specific_kwargs.get("max_pixels", 5120 * 28 * 28)  # ~4M pixels
    
    # Get question and visuals
    question = mathvision_reason_doc_to_text_qwen3(doc, lmms_eval_specific_kwargs)
    visuals = mathvision_reason_doc_to_visual_qwen3(doc)
    
    # Build messages with system prompt
    system_messages = [{
        "role": "system", 
        "content": [{"type": "text", "text": system_prompt}]
    }]
    
    user_content = []
    
    # Add image with official resolution settings if available
    if visuals:
        user_content.append({
            "type": "image",
            "url": visuals[0],
            "min_pixels": min_pixels,
            "max_pixels": max_pixels
        })
    
    # Add question text
    user_content.append({
        "type": "text", 
        "text": question.strip()
    })
    
    messages = system_messages + [{"role": "user", "content": user_content}]
    return messages


def mathvision_reason_qwen3_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Process evaluation results for Qwen3-VL reasoning version.
    
    This function uses the reasoning evaluation logic with <think> and <answer> tags.
    
    Args:
        doc: Ground truth document
        results: List of model predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    question = mathvision_reason_doc_to_text_qwen3(doc, None)
    ground_truth = str(doc.get("answer", ""))
    extra_info = {"question": question}
    
    # Lazy import to avoid ANTLR version issues at module load
    from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
    
    acc_score = 0
    fmt_score = 0
    
    for pred in results:
        score_dict = compute_score(
            data_source="mathvista", 
            solution_str=pred.strip(), 
            ground_truth=ground_truth, 
            extra_info=extra_info
        )
        acc_score += score_dict.get("acc_score", 0)
        fmt_score += score_dict.get("format_reward_score", 0.0)
    
    n = len(results) if results else 1
    
    return {
        "acc_score": acc_score / n, 
        "format_score": fmt_score / n
    }
