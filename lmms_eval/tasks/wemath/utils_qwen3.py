"""
Qwen3-VL WeMath Evaluation Implementation

This module provides Qwen3-VL optimized evaluation for the WeMath benchmark.

Key features:
1. Optimized for Qwen3-VL with official image resolution settings
2. Support for min_pixels/max_pixels parameters
3. Compatible with WeMath reasoning evaluation metrics

Reference:
- WeMath: https://github.com/We-Math/We-Math
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

# compute_score imported inside functions to avoid ANTLR version issues at module load
from lmms_eval.tasks.wemath.wemath_utils import (
    calculate_metrics,
    compute_final_scores,
    process_steps_data,
    update_main_results_df,
)



def wemath_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """
    Convert document to text prompt for Qwen3-VL.
    
    Args:
        doc: Document containing question and options
        lmms_eval_specific_kwargs: Optional kwargs for customization
        
    Returns:
        Formatted question text
    """
    return doc.get("question", "") + "\n" + doc.get("option", "")


def wemath_doc_to_visual_qwen3(doc: Dict) -> List:
    """
    Extract visual elements from document for Qwen3-VL.
    
    Args:
        doc: Document containing image data
        
    Returns:
        List of RGB images
    """
    image_path = doc.get("image_path")
    if image_path:
        return [image_path.convert("RGB")]
    return []


def wemath_doc_to_messages_qwen3(
    doc: Dict, 
    lmms_eval_specific_kwargs: Optional[Dict] = None
) -> List[Dict]:
    """
    Convert document to Qwen3-VL message format with official settings.
    
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
    question = wemath_doc_to_text_qwen3(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual_qwen3(doc)
    
    # Build messages with system prompt
    system_messages = [{
        "role": "system", 
        "content": [{"type": "text", "text": system_prompt}]
    }]
    
    user_message = {"role": "user", "content": []}
    
    # Add image with official resolution settings if available
    if visuals:
        user_message["content"].append({
            "type": "image",
            "url": visuals[0],
            "min_pixels": min_pixels,
            "max_pixels": max_pixels
        })
    
    # Add question text
    user_message["content"].append({
        "type": "text", 
        "text": question.strip()
    })
    
    messages = system_messages + [user_message]
    return messages


def wemath_qwen3_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Process evaluation results for Qwen3-VL.
    
    This function computes accuracy and format scores for WeMath evaluation.
    Compatible with the original WeMath evaluation logic.
    
    Args:
        doc: Ground truth document
        results: List of model predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Lazy import to avoid ANTLR version issues at module load
    from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
    
    acc_score = 0
    format_score = 0
    question = wemath_doc_to_text_qwen3(doc, None)
    extra_info = {"question": question}
    
    for pred in results:
        score_dict = compute_score(
            data_source="wemath", 
            solution_str=pred.strip(), 
            ground_truth=doc.get("answer", ""), 
            extra_info=extra_info
        )
        acc_score += score_dict.get("acc_score", 0)
        format_score += score_dict.get("format_reward_score", 0.0)
    
    num_results = len(results) if results else 1
    
    data_dict = {
        "ID": doc.get("ID"),
        "split": doc.get("split"),
        "knowledge concept": doc.get("knowledge concept"),
        "question": doc.get("question"),
        "option": doc.get("option"),
        "answer": doc.get("answer"),
        "key": doc.get("key"),
        "question number": doc.get("question number"),
        "knowledge concept description": doc.get("knowledge concept description"),
        "acc_score": acc_score / num_results,
    }
    
    return {
        "wemath_loose": data_dict, 
        "wemath_strict": data_dict, 
        "acc_score": acc_score / num_results, 
        "format_score": format_score / num_results
    }


def wemath_aggregate_results(results: List[Dict], metric_name: str) -> str:
    """
    Aggregate WeMath results and compute final scores.
    
    This function processes results from 2-step and 3-step questions,
    computes various metrics, and returns the final score.
    
    Args:
        results: List of result dictionaries
        metric_name: Either "wemath_loose" or "wemath_strict"
        
    Returns:
        Formatted score string
    """
    if not results:
        return "0.00%"
    
    data = pd.DataFrame(results)
    data["joker"] = data["acc_score"] == 1.0
    
    # Split data by step type
    data_2steps = data[data["key"].str.contains("2steps", na=False)]
    data_3steps = data[data["key"].str.contains("3steps", na=False)]
    
    # Process step data
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    
    # Calculate metrics
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    
    if metric_name == "wemath_loose":
        return score_dict.get("Score (Loose)", "0.00%")
    elif metric_name == "wemath_strict":
        return score_dict.get("Score (Strict)", "0.00%")
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")


def wemath_aggregate_results_loose(results: List[Dict]) -> str:
    """Aggregate results using loose scoring."""
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict(results: List[Dict]) -> str:
    """Aggregate results using strict scoring."""
    return wemath_aggregate_results(results, "wemath_strict")
