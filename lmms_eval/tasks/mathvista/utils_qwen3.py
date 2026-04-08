"""
Qwen3-VL MathVista Evaluation Implementation

Provides Qwen3-VL optimized evaluation for the MathVista benchmark.

Key features:
1. Optimized for Qwen3-VL with official image resolution settings
2. Support for min_pixels/max_pixels parameters
3. Compatible with MathVista standard and reasoning evaluation
4. Rule-based evaluation first, LLM judge fallback (Qwen3 aligned)
"""

import os
from typing import Any, Dict, List, Optional

from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator
from lmms_eval.tasks.mathvista.reasoning.utils import (
    SYSTEM_PROMPT as MATHVISTA_REASON_SYSTEM_PROMPT,
    mathvista_doc_to_text as mathvista_reason_doc_to_text,
    mathvista_doc_to_visual as mathvista_reason_doc_to_visual,
    mathvista_process_results as _mathvista_reason_process_results_orig,
)
from lmms_eval.tasks.mathvista.utils import (
    mathvista_aggregate_results as mathvista_qwen3_aggregate_results,
    mathvista_doc_to_text,
    mathvista_doc_to_visual,
    mathvista_process_results as _mathvista_process_results_orig,
)

# Initialize evaluator for potential LLM judge fallback
_mathvista_evaluator = MathVistaEvaluator()


def _has_judge_api() -> bool:
    """Check if OpenAI-compatible judge API is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return api_key and api_key not in ("YOUR_API_KEY", "your-api-key", "")


def _rule_match_mathvista(prediction: str, answer: str, choices: Optional[List[str]] = None) -> float:
    """
    Rule-based matching for MathVista.
    Tries exact match, MCQ option match, then math_verify equivalence.
    """
    from lmms_eval.tasks._task_utils.reasoning_utils import relax_exact_match, simple_parse

    pred = simple_parse(prediction)
    gt = simple_parse(answer)

    # 1. Exact / relaxed match (includes MCQ parsing)
    score = relax_exact_match(pred, gt)
    if score == 1.0:
        return 1.0

    # 2. If choices available, try matching option content
    if choices:
        pred_upper = pred.upper().strip()
        options = [chr(ord("A") + i) for i in range(len(choices))]
        if pred_upper in options:
            idx = options.index(pred_upper)
            if choices[idx] == gt:
                return 1.0

    # 3. math_verify fallback for mathematical expressions
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


def mathvista_qwen3_process_results(doc, results):
    """
    MathVista Qwen3 result processing.

    Unified logic for both generation mode and standalone judge mode:
    1. Rule-based matching first (exact / option / math_verify)
    2. LLM judge fallback ONLY if API key is configured and rule fails
    """
    gt_doc, pred_from_sample = _extract_doc_info(doc)

    if pred_from_sample is not None:
        prediction = str(pred_from_sample).strip()
    else:
        prediction = results[0].strip() if results else ""

    # Extract ground truth fields (handle both raw doc and judge sample)
    answer = gt_doc.get("answer")
    if answer is None:
        answer = gt_doc.get("target")

    choices = gt_doc.get("choices", [])
    question_type = gt_doc.get("question_type", "")
    answer_type = gt_doc.get("answer_type", "")
    precision = gt_doc.get("precision", 0)
    query = gt_doc.get("query", "")
    pid = gt_doc.get("pid", gt_doc.get("id", gt_doc.get("doc_id", 0)))
    metadata = gt_doc.get("metadata", {})
    unit = gt_doc.get("unit", "")
    caption = gt_doc.get("caption", "")
    ocr = gt_doc.get("ocr", "")

    true_false = False
    extraction = prediction

    if answer is not None:
        # Stage 1: rule-based matching
        judge_result = _rule_match_mathvista(prediction, answer, choices)

        # Stage 2: use MathVistaEvaluator's extraction/normalization if rule fails and API available
        if judge_result == 0.0 and _has_judge_api():
            try:
                problem = {
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "query": query,
                    "choices": choices,
                    "answer": answer,
                    "precision": precision,
                }
                extraction = _mathvista_evaluator.extract_answer(
                    prediction, problem, quick_extract=metadata.get("quick_extract", False)
                )
                prediction_norm = _mathvista_evaluator.normalize_extracted_answer(
                    extraction, choices, question_type, answer_type, precision
                )
                true_false = _mathvista_evaluator.safe_equal(prediction_norm, answer)
            except Exception:
                true_false = False
        else:
            true_false = judge_result == 1.0

    result = {
        "question_id": pid,
        "query": query,
        "choices": choices,
        "answer": answer,
        "extraction": extraction,
        "prediction": prediction,
        "true_false": true_false,
        "score": 1.0 if true_false else 0.0,
        "question_type": question_type,
        "answer_type": answer_type,
        "precision": precision,
        "metadata": metadata,
    }

    return {"llm_as_judge_eval": result, "submission": result}


# ==================== Standard Version Functions ====================

def mathvista_doc_to_visual_qwen3(doc: Dict) -> List:
    """Extract visual elements from document for Qwen3-VL standard version."""
    return mathvista_doc_to_visual(doc)


def mathvista_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Convert document to text prompt for Qwen3-VL standard version."""
    return mathvista_doc_to_text(doc, lmms_eval_specific_kwargs)


# ==================== Reasoning Version Functions ====================

def mathvista_reason_doc_to_visual_qwen3(doc: Dict) -> List:
    """Extract visual elements from document for Qwen3-VL reasoning version."""
    return mathvista_reason_doc_to_visual(doc)


def mathvista_reason_doc_to_text_qwen3(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    """Convert document to text prompt for Qwen3-VL reasoning version."""
    return mathvista_reason_doc_to_text(doc, lmms_eval_specific_kwargs)


def mathvista_reason_doc_to_messages_qwen3(
    doc: Dict,
    lmms_eval_specific_kwargs: Optional[Dict] = None,
) -> List[Dict]:
    """Convert document to Qwen3-VL message format for reasoning tasks."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    system_prompt = lmms_eval_specific_kwargs.get(
        "system_prompt",
        MATHVISTA_REASON_SYSTEM_PROMPT,
    )
    min_pixels = lmms_eval_specific_kwargs.get("min_pixels", 1280 * 28 * 28)
    max_pixels = lmms_eval_specific_kwargs.get("max_pixels", 5120 * 28 * 28)

    question = mathvista_reason_doc_to_text_qwen3(doc, lmms_eval_specific_kwargs)
    visuals = mathvista_reason_doc_to_visual_qwen3(doc)

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
