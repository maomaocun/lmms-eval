"""
Qwen3-VL MathVerse Evaluation Implementation

Provides Qwen3-VL optimized evaluation for the MathVerse benchmark.

Key features:
1. Optimized for Qwen3-VL with official image resolution settings
2. Support for min_pixels/max_pixels parameters
3. Compatible with MathVerse standard and reasoning evaluation
4. Rule-based evaluation first, LLM judge fallback (Qwen3 aligned)
"""

import os
from typing import Any, Dict, List, Optional

from lmms_eval.tasks.mathverse.reasoning.utils import (
    SYSTEM_PROMPT as MATHVERSE_REASON_SYSTEM_PROMPT,
    mathverse_doc_to_text as mathverse_reason_doc_to_text,
    mathverse_doc_to_visual as mathverse_reason_doc_to_visual,
    mathverse_process_results as mathverse_reason_qwen3_process_results,
)
from lmms_eval.tasks.mathverse.utils import (
    mathverse_aggregate_results_eval as mathverse_qwen3_aggregate_results_eval,
    mathverse_aggregate_results_submission as mathverse_qwen3_aggregate_results_submission,
    mathverse_doc_to_text,
    mathverse_doc_to_visual,
    mathverse_process_results as _mathverse_process_results_orig,
)




def _has_judge_api() -> bool:
    """Check if OpenAI-compatible judge API is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return api_key and api_key not in ("YOUR_API_KEY", "your-api-key", "")


def _rule_match_mathverse(prediction: str, answer: str) -> float:
    """
    Rule-based matching for MathVerse.
    Tries exact match, then math_verify equivalence.
    """
    from lmms_eval.tasks._task_utils.reasoning_utils import relax_exact_match, simple_parse

    pred = simple_parse(prediction)
    gt = simple_parse(answer)

    # 1. Exact / relaxed match (includes MCQ parsing)
    score = relax_exact_match(pred, gt)
    if score == 1.0:
        return 1.0

    # 2. math_verify fallback for mathematical expressions
    try:
        from math_verify import parse, verify

        gold = parse(gt)
        p = parse(pred)
        if verify(gold, p):
            return 1.0
    except Exception:
        pass

    return 0.0


def _extract_doc_info(doc_or_sample):
    """Extract ground-truth info from raw doc or judge sample."""
    # Judge mode wrapper from standalone.py
    if doc_or_sample and "__sample_context__" in doc_or_sample:
        sample = doc_or_sample["__sample_context__"]
        gt_doc = sample.get("doc", sample)
        prediction = sample.get("filtered_resps", [""])
        if isinstance(prediction, list) and prediction:
            prediction = prediction[0]
        else:
            prediction = str(prediction)
    else:
        gt_doc = doc_or_sample
        prediction = None

    return gt_doc, prediction


def mathverse_qwen3_process_results(doc, results):
    """
    MathVerse Qwen3 result processing.

    Generation mode: rule-based only, defers LLM judge to standalone phase.
    Judge mode (when __sample_context__ present): allows LLM judge fallback.
    """
    gt_doc, pred_from_sample = _extract_doc_info(doc)
    # Prediction priority: sample wrapper > results list
    if pred_from_sample is not None:
        prediction = str(pred_from_sample).strip()
    else:
        prediction = results[0].strip() if results else ""

    question = gt_doc.get("question_for_eval", "")
    answer = gt_doc.get("answer") if "answer" in gt_doc else None
    if answer is None:
        answer = gt_doc.get("target")

    judge_result = 0
    needs_llm_judge = False
    if answer is not None:
        # Stage 1: rule-based matching
        judge_result = int(_rule_match_mathverse(prediction, answer))

        # Stage 2: defer LLM judge fallback to standalone judge (both generation and judge mode)
        # This avoids double-judging when lmms-eval judge --judge-mode auto is used.
        if judge_result == 0:
            needs_llm_judge = True

    result = {
        "sample_index": gt_doc.get("sample_index", gt_doc.get("doc_id", 0)),
        "problem_index": gt_doc.get("problem_index", 0),
        "problem_version": gt_doc.get("problem_version", ""),
        "question": gt_doc.get("question", ""),
        "answer": answer,
        "prediction": prediction,
        "question_type": gt_doc.get("question_type", ""),
        "metadata": gt_doc.get("metadata", {}),
        "query_wo": gt_doc.get("query_wo", ""),
        "query_cot": gt_doc.get("query_cot", ""),
        "question_for_eval": question,
        "true_false": judge_result == 1,
    }

    metrics = {"gpt_eval_score": judge_result, "submission": result}
    if needs_llm_judge:
        metrics["needs_llm_judge"] = True
        metrics["formatted_question"] = question
        metrics["answer"] = answer
    return metrics


# ==================== Standard Version Functions ====================

def mathverse_doc_to_visual_qwen3(doc: Dict) -> List:
    """Extract visual elements from document for Qwen3-VL standard version."""
    return mathverse_doc_to_visual(doc)


def mathverse_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Convert document to text prompt for Qwen3-VL standard version."""
    return mathverse_doc_to_text(doc, lmms_eval_specific_kwargs)


# ==================== Reasoning Version Functions ====================

def mathverse_reason_doc_to_visual_qwen3(doc: Dict) -> List:
    """Extract visual elements from document for Qwen3-VL reasoning version."""
    return mathverse_reason_doc_to_visual(doc)


def mathverse_reason_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Convert document to text prompt for Qwen3-VL reasoning version."""
    return mathverse_reason_doc_to_text(doc, lmms_eval_specific_kwargs)


def mathverse_reason_doc_to_messages_qwen3(
    doc: Dict,
    lmms_eval_specific_kwargs: Optional[Dict] = None,
) -> List[Dict]:
    """Convert document to Qwen3-VL message format for reasoning tasks."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    system_prompt = lmms_eval_specific_kwargs.get(
        "system_prompt",
        MATHVERSE_REASON_SYSTEM_PROMPT,
    )
    min_pixels = lmms_eval_specific_kwargs.get("min_pixels", 1280 * 28 * 28)
    max_pixels = lmms_eval_specific_kwargs.get("max_pixels", 5120 * 28 * 28)

    question = mathverse_reason_doc_to_text_qwen3(doc, lmms_eval_specific_kwargs)
    visuals = mathverse_reason_doc_to_visual_qwen3(doc)

    system_messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    user_content = []
    if visuals:
        user_content.append(
            {
                "type": "image",
                "url": visuals[0],
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
    user_content.append({"type": "text", "text": question.strip()})

    return system_messages + [{"role": "user", "content": user_content}]
