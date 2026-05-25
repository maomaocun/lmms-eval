"""Official SciVQR reasoning-quality evaluation adapter."""

from __future__ import annotations

import ast
import json
import os
import re
import time
import uuid
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from lmms_eval.tasks.scivqr import utils as base


JUDGE_MODEL = os.getenv("SCIVQR_REASONING_JUDGE_MODEL", "gpt-4o")
JUDGE_MAX_TOKENS = 5120
JUDGE_TEMPERATURE = 0.7
FORCE_REPROCESS_FROM_SAMPLE = True
OFFICIAL_BATCH_ENDPOINT = "/v1/chat/completions"
OFFICIAL_BATCH_COMPLETION_WINDOW = "24h"
OFFICIAL_REASONING_PROMPT_PATTERN = re.compile(r"\\boxed\{\}.*?\n(.*?)(?:\nChoices:|\Z)", re.DOTALL)

SYSTEM_PROMPT = """
You are a reasoning evaluator designed to assess the alignment, coherence, and quality of reasoning steps in text responses. Your task is to evaluate reasoning steps between the * ground truth * and the * LLM response * using the following metrics:

1. ** Faithfulness (1 - 10) :**
    - Definition : Measures how well the reasoning steps in the LLM response align with the source reasoning steps .
    - Scoring Guidelines :
        - 9 - 10: All or almost all steps match or closely reflect the ground truth reasoning.
        - 7 - 8: Most steps are aligned , with minor deviations.
        - 5 - 6: Some steps align , but several are missing or significantly altered.
        - 3 - 4: Few steps align correctly ; most are off or missing .
        - 1 - 2: The majority of steps are not aligned with the source .

2. ** Informativeness ( Info - Step ) (1 - 10) :**
    - Definition : Measures how well the reasoning steps extract all relevant information from the source .
    - Scoring Guidelines :
        - 9 - 10: Almost all critical information steps are present and accurate .
        - 7 - 8: Most important points are included , with minor omissions .
        - 5 - 6: Some key information is missing or underdeveloped .
        - 3 - 4: Limited inclusion of critical content .
        - 1 - 2: Very poor extraction of relevant information .

3. ** Repetition and Redundancy (1 - 10) :**
    - Definition : Identifies repeated or unnecessarily paraphrased reasoning steps within the hypothesis or redundant reasoning steps that do not add value.
    - Scoring Guidelines :
        - 9 -10: No or minimal unnecessary repetition and redundancy.
        - 7 -8: Minor repetition or redundancy that doesn ' t impede clarity .
        - 5 -6: Noticeable repetition or redundancy that doesn ' t add value .
        - 3 -4: Frequent repetition or redundancy that disrupts coherence .
        - 1 -2: Excessive repetition or redundancy reducing the quality of reasoning .

4. ** Hallucination (1 - 10) :**
    - Definition : Detect irrelevant or invented reasoning steps not aligned with the source .
    - Scoring Guidelines :
        - 9 - 10: No hallucinations ; all reasoning is grounded in the source .
        - 7 - 8: One or two minor hallucinations .
        - 5 - 6: Several steps contain invented or irrelevant details .
        - 3 - 4: Many hallucinations , but some grounding remains .
        - 1 - 2: Mostly hallucinated reasoning .

5. ** Missing Step (1 -10) :**
    - Definition : Identify if any necessary reasoning steps are missing .
    - Scoring Guidelines :
        - 9 - 10: No critical steps missing .
        - 7 - 8: Minor missing steps that don ' t significantly affect
        the conclusion .
        - 5 - 6: Some important steps absent , affecting the outcome .
        - 3 - 4: Several crucial missing steps .
        - 1 - 2: Major gaps ; the reasoning chain is incomplete .

** Additional Instructions for Consistency :**
    - Always follow the above scoring guidelines strictly .
    - Before scoring , re - read both the ground truth and the LLM response carefully .
    - Compare the reasoning steps directly to determine where they align or diverge .
    - Use the provided scoring benchmarks ( anchor examples , if any ) as a reference to maintain consistency across evaluations .
    - Avoid subjective interpretation and adhere to the given thresholds .
    - Once scores for all metrics are determined , compute the Overall Score as the average of all metric scores .
    - Provide the final output as a Python dictionary with the structure only don ' t add a anything extra , beacuase your out will be used in code pipeline . So single change in you output will crash whole system . :
        # Example output : { 'Faithfulness ': 8.0 , 'Informativeness': 8.5 , 'Repetition&Redundancy': 9.0 , 'Hallucination': 9.5 , 'Missing': 8.5 , 'Overall': 8.65}
"""


def scivqr_process_docs_reasoning(dataset):
    return base.scivqr_process_docs_reasoning(dataset)


def scivqr_doc_to_visual(doc):
    return base.scivqr_doc_to_visual(doc)


def scivqr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    choices = doc.get("choices") or []
    if choices and len(choices) > 0:
        options = [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = "\n".join(f"{option}. {choice}" for option, choice in zip(options, choices))
        return f"{question}\n{choices_str}"
    return question


def scivqr_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    user_content = []
    for visual in scivqr_doc_to_visual(doc):
        user_content.append({"type": "image", "url": visual})
    user_content.append({"type": "text", "text": scivqr_doc_to_text(doc, lmms_eval_specific_kwargs).strip()})
    return [{"role": "user", "content": user_content}]


def scivqr_doc_to_target(doc, model_specific_target_kwargs=None):
    return doc.get("solution") or doc.get("answer", "")


def scivqr_reasoning_process_results(doc, results):
    if "__sample_context__" in doc:
        sample = doc["__sample_context__"]
        nested = sample.get("scivqr_reasoning") or {}
        choices = nested.get("choices")
        if choices is None:
            choices = sample.get("choices") or []
        doc = {
            "pid": nested.get("pid", sample.get("question_id", sample.get("doc_id"))),
            "question": nested.get("question") or sample.get("question") or sample.get("prompt") or sample.get("input", ""),
            "solution": nested.get("gt_reason") or sample.get("gt_reason") or sample.get("solution") or sample.get("target", ""),
            "answer": nested.get("answer", sample.get("answer", "")),
            "choices": choices,
            "subject": nested.get("subject", sample.get("subject", "unknown")),
            "question_type": nested.get("question_type", "unknown"),
        }
    response = results[0] if results else ""
    record = {
        "question_id": str(doc.get("pid", "")),
        "pid": doc.get("pid"),
        "question": doc.get("question", ""),
        "gt_reason": doc.get("solution", ""),
        "response": response,
        "answer": doc.get("answer", ""),
        "choices": list(doc.get("choices") or []),
        "subject": doc.get("subject", "unknown"),
        "question_type": doc.get("question_type", "unknown"),
        "score": None,
        "parsed_score": None,
    }
    return {
        "scivqr_reasoning": record,
        "needs_llm_judge": True,
    }


def get_judge_messages(doc, prediction, target=None):
    gt_reason = doc.get("solution") or doc.get("gt_reason") or target or ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"* ground truth *: {gt_reason}\n* LLM response *: {prediction}"},
    ]


def parse_judge_response(response: str) -> Dict[str, Any]:
    raw = (response or "").strip()
    if not raw:
        return {}
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    try:
        return json.loads(raw)
    except Exception:
        pass
    return {"raw": raw}


def _normalize_question_key(question: Any) -> str:
    text = str(question or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def extract_official_reasoning_question(prompt: str) -> Optional[str]:
    """Extract the question text exactly like official evaluate_reasoning.py."""
    match = OFFICIAL_REASONING_PROMPT_PATTERN.search(prompt or "")
    if not match:
        return None
    return match.group(1)


def extract_reasoning_question_fallback(sample: Dict[str, Any]) -> str:
    """Best-effort extraction for lmms-eval/official JSONL rows without \\boxed{}."""
    text = str(sample.get("question") or sample.get("prompt") or sample.get("input") or "")
    official = extract_official_reasoning_question(text)
    if official is not None:
        return official.strip()
    text = re.split(r"\n[A-E]\.\s+", text, maxsplit=1)[0]
    text = text.split("\nChoices:")[0]
    for suffix in [
        "\nAnswer with the option's letter from the given choices directly.",
        "\nAnswer the question directly.",
    ]:
        text = text.replace(suffix, "")
    return text.strip()


def build_official_reasoning_items(
    samples: Iterable[Dict[str, Any]],
    dataset_docs: Iterable[Dict[str, Any]],
    *,
    strict_official_prompt: bool = False,
) -> List[Dict[str, Any]]:
    """Build the official reasoning batch input rows.

    The official script indexes ground-truth reasoning by exact question text
    and keeps only rows whose prompt can be matched to a non-empty solution.
    ``strict_official_prompt=True`` preserves that regex-only behavior.  The
    default additionally supports lmms-eval's JSONL prompts by falling back to
    pid/question matching while keeping the same output schema.
    """
    docs = list(dataset_docs or [])
    gt_by_question = {
        doc.get("question"): doc.get("solution")
        for doc in docs
        if doc.get("question")
    }
    gt_by_normalized_question = {
        _normalize_question_key(doc.get("question")): doc.get("solution")
        for doc in docs
        if doc.get("question")
    }
    gt_by_subject_question = {
        (str(doc.get("subject")), _normalize_question_key(doc.get("question"))): doc.get("solution")
        for doc in docs
        if doc.get("subject") and doc.get("question")
    }
    gt_by_pid = {
        str(doc.get("pid")): doc.get("solution")
        for doc in docs
        if doc.get("pid") is not None
    }

    data = []
    for sample in samples:
        prompt = str(sample.get("prompt") or sample.get("input") or sample.get("question") or "")
        question = extract_official_reasoning_question(prompt)
        gt_reason = gt_by_question.get(question) if question is not None else None

        if not strict_official_prompt and not gt_reason:
            question = extract_reasoning_question_fallback(sample)
            subject = sample.get("subject")
            matched_subject_question = False
            if subject and question:
                subject_key = (str(subject), _normalize_question_key(question))
                if subject_key in gt_by_subject_question:
                    gt_reason = gt_by_subject_question[subject_key]
                    matched_subject_question = True

            if not matched_subject_question and subject:
                gt_reason = None
            elif not matched_subject_question:
                matched_pid = False
                for key in ["pid", "question_id", "doc_id"]:
                    value = sample.get(key)
                    if value is not None and str(value) in gt_by_pid:
                        gt_reason = gt_by_pid[str(value)]
                        matched_pid = True
                        break
                if not matched_pid:
                    gt_reason = gt_by_normalized_question.get(_normalize_question_key(question))

        if not question or not gt_reason:
            continue
        data.append(
            {
                "question_id": sample.get("question_id", sample.get("pid", sample.get("doc_id"))),
                "question": question,
                "gt_reason": gt_reason,
                "response": sample.get("response", sample.get("filtered_resps", "")),
            }
        )
    return data


def _official_chunk_bounds(total_items: int, split_id: int, num_chunk: int) -> Tuple[int, int]:
    if num_chunk <= 0:
        raise ValueError("num_chunk must be positive")
    chunk_size = ceil(total_items / num_chunk) if total_items else 0
    start = split_id * chunk_size
    end = min(start + chunk_size, total_items)
    return start, end


def build_official_reasoning_batch_request(item: Dict[str, Any], custom_id: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"* ground truth *: {item['gt_reason']}\n* LLM response *: {item['response']}"},
    ]
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": OFFICIAL_BATCH_ENDPOINT,
        "body": {
            "model": JUDGE_MODEL,
            "temperature": JUDGE_TEMPERATURE,
            "max_tokens": JUDGE_MAX_TOKENS,
            "messages": messages,
        },
    }


def write_official_reasoning_batch_requests(
    data: List[Dict[str, Any]],
    output_path,
    *,
    split_id: int,
    num_chunk: int,
    uuid_fn: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Write official ``requests_chunk*.jsonl`` and return id mapping metadata."""
    start, end = _official_chunk_bounds(len(data), split_id, num_chunk)
    split_data = data[start:end]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uuid_fn = uuid_fn or uuid.uuid4

    id_mapping = {}
    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(split_data):
            custom_id = str(uuid_fn())
            id_mapping[custom_id] = idx
            req = build_official_reasoning_batch_request(item, custom_id)
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")

    return {
        "data": data,
        "split_data": split_data,
        "id_mapping": id_mapping,
        "start": start,
        "end": end,
    }


def write_official_reasoning_results_from_ndjson(
    original_data: List[Dict[str, Any]],
    id_mapping: Dict[str, int],
    result_ndjson_path,
    output_json_path,
    *,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    """Process OpenAI Batch ndjson exactly like official evaluate_reasoning.py."""
    chunk_data = [dict(item) for item in original_data[start:end]]
    hit_idx = []
    with open(result_ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            response = result["response"]["body"]
            if custom_id in id_mapping:
                idx = id_mapping[custom_id]
                try:
                    chunk_data[idx]["score"] = response["choices"][0]["message"]["content"].strip()
                    hit_idx.append(idx)
                except KeyError:
                    pass

    official_results = [chunk_data[i] for i in range(len(chunk_data)) if i in hit_idx]
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(official_results, f, indent=2, ensure_ascii=False)
    return official_results


def submit_official_reasoning_batch(client: Any, requests_jsonl_path) -> str:
    """Submit the official SciVQR reasoning OpenAI Batch job."""
    with open(requests_jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=OFFICIAL_BATCH_ENDPOINT,
        completion_window=OFFICIAL_BATCH_COMPLETION_WINDOW,
    )
    return batch.id


def wait_for_official_reasoning_batch(
    client: Any,
    batch_id: str,
    *,
    interval: int = 10,
    sleep_fn: Callable[[int], None] = time.sleep,
) -> Any:
    """Poll an official OpenAI Batch job until it reaches a terminal status."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            return batch
        sleep_fn(interval)


def download_official_reasoning_batch_results(client: Any, batch: Any, output_ndjson_path) -> None:
    """Download official Batch output ndjson from ``batch.output_file_id``."""
    if batch.status != "completed":
        raise RuntimeError(f"Batch not successful: {batch.status}")

    output_ndjson_path = Path(output_ndjson_path)
    output_ndjson_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(batch.output_file_id)
    with open(output_ndjson_path, "wb") as f:
        for chunk in content.iter_bytes():
            f.write(chunk)


def update_metrics_from_judge(doc, results, metrics, parsed, raw_response, model):
    updated = dict(metrics)
    updated["llm_judge_raw"] = raw_response
    updated["llm_judge_model"] = model
    updated["llm_judge_success"] = bool(parsed)
    if isinstance(parsed, dict) and "Overall" in parsed:
        updated["llm_judge_score"] = parsed["Overall"]

    if "scivqr_reasoning" in updated and isinstance(updated["scivqr_reasoning"], dict):
        record = dict(updated["scivqr_reasoning"])
        record["score"] = raw_response
        record["parsed_score"] = parsed
        record["judge_model"] = model
        updated["scivqr_reasoning"] = record
    return updated


def save_judged_results(results: List[Dict[str, Any]], output_path) -> None:
    """Write SciVQR reasoning judged outputs like official Evaluation-Chunk JSON."""
    official_results = []
    for sample in results:
        metrics = sample.get("metrics", {}) if isinstance(sample, dict) else {}
        record = metrics.get("scivqr_reasoning", {}) if isinstance(metrics, dict) else {}
        official_results.append(
            {
                "question_id": record.get("question_id", str(sample.get("question_id", sample.get("doc_id", "")))),
                "question": record.get("question", sample.get("question", sample.get("prompt", ""))),
                "gt_reason": record.get("gt_reason", sample.get("gt_reason", sample.get("solution", ""))),
                "response": record.get("response", sample.get("response", "")),
                "score": record.get("score"),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(official_results, f, indent=2, ensure_ascii=False)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scivqr_reasoning_aggregate_scores(results: List[Dict[str, Any]]):
    score_keys = ["Faithfulness ", "Faithfulness", "Informativeness", "Repetition&Redundancy", "Hallucination", "Missing", "Overall"]
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    judged = 0

    for result in results:
        parsed = result.get("parsed_score")
        if not isinstance(parsed, dict):
            continue
        judged += 1
        for key in score_keys:
            value = _as_float(parsed.get(key))
            if value is None:
                continue
            canonical = key.strip()
            totals[canonical] = totals.get(canonical, 0.0) + value
            counts[canonical] = counts.get(canonical, 0) + 1

    if judged == 0:
        return {"mode": "generation_only", "total_samples": len(results), "judged_samples": 0}

    summary = {key: totals[key] / counts[key] for key in sorted(totals) if counts.get(key)}
    summary["mode"] = "judged"
    summary["total_samples"] = len(results)
    summary["judged_samples"] = judged
    return summary
