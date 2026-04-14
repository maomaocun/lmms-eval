from typing import Dict, Any, List, Optional
from PIL import Image


def get_judge_prompt(doc: Dict[str, Any], prediction: str, target: Optional[str] = None) -> str:
    """Return the chemistry-specific judge prompt for standalone judging."""
    question = doc_to_text(doc)
    ground_truth = target if target is not None else doc_to_target(doc)
    return f"""You are a professional evaluation assistant. Please carefully compare whether the model's predicted answer matches the standard answer.

Evaluation criteria:
1. For chemical formulas/E-SMILES: Consider correct if structures are identical
2. For numerical answers: Consider correct if values are the same (allow minor differences in decimal places)
3. For text answers: Consider correct if semantics are the same
4. For Yes/No questions: Consider correct if the answer direction is consistent

Please only answer "correct" or "incorrect", do not explain the reasons.

Question: {question}

Standard Answer: {ground_truth}
Model Prediction: {prediction}

Please judge whether the model prediction is correct? Only answer "correct" or "incorrect":"""


def doc_to_visual(doc):
    image = doc.get("image")
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return []


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    content = doc.get("question", "")
    return f"{pre_prompt}{content}{post_prompt}"


def doc_to_target(doc):
    return doc.get("answer", "")


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    prediction = results[0] if isinstance(results, list) else results
    target = doc_to_target(doc)
    # Decoupled from LLM judge: return a placeholder so that standalone JudgeRunner
    # can trigger LLM fallback in auto mode and apply the custom chemistry prompt.
    return {
        "llm_judge_score": 0,
        "needs_llm_judge": True,
        "question": doc_to_text(doc),
        "raw_output": prediction,
        "ground_truth": target,
    }


def aggregation(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0
