"""SciVQR official evaluation adapters for lmms-eval.

The scoring code below intentionally mirrors the official SciVQR repository:
``src/SciVQR/code/evaluate_multichoice.py``,
``src/SciVQR/code/evaluate_open.py`` and ``src/SciVQR/code/utils.py``.
Only the surrounding data-shaping code is adapted for lmms-eval.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import defaultdict
from math import *  # noqa: F403 - official SciVQR uses eval() on latex2sympy output
from typing import Any, Dict, Iterable, List, Optional

try:
    from latex2sympy2_extended import latex2sympy
except ImportError:  # pragma: no cover - upstream SciVQR originally used this package name
    from latex2sympy2 import latex2sympy
from sympy import Integer, Rational


SUBJECTS = ["math", "physics", "chemistry", "biology", "geography", "astronomy"]
OPEN_JUDGE_MODEL = os.getenv("SCIVQR_OPEN_JUDGE_MODEL", "Qwen2.5-72B-Instruct")
OPEN_JUDGE_MAX_TOKENS = None
JUDGE_MODEL = OPEN_JUDGE_MODEL
JUDGE_MAX_TOKENS = OPEN_JUDGE_MAX_TOKENS
JUDGE_TEMPERATURE = None
JUDGE_NUM_RETRIES = 2
JUDGE_RETRY_DELAY = 2
JUDGE_TIMEOUT = 10
FORCE_REPROCESS_FROM_SAMPLE = True
_FRAMEWORK_OUTPUT_KEYS = {
    "doc",
    "doc_id",
    "target",
    "filtered_resps",
    "resps",
    "token_counts",
    "doc_hash",
    "input",
    "input_media",
    "arguments",
    "metrics",
    "judge_mode",
    "__sample_context__",
    "__scivqr_subject_from_path",
}


def _decode_base64_image(b64_str: str):
    img_bytes = base64.b64decode(b64_str)
    from PIL import Image

    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def scivqr_process_docs_mcq(dataset):
    return dataset


def scivqr_process_docs_open(dataset):
    return dataset.filter(lambda x: x.get("question_type") == "open")


def scivqr_process_docs_reasoning(dataset):
    return dataset.filter(lambda x: bool(x.get("solution", "")))


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    splits = string.split("\\text{ ")
    return splits[0]


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if "sqrt" in string:
        string = _fix_sqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def find_math_answer(s: str) -> str:
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")
    try:
        pattern = re.compile("oxed{(.*)}", flags=re.S)
        ans = pattern.findall(s)[-1]
    except Exception:
        ans = s
    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]
    ans = ans.split("=")[-1]
    ans = ans.split("\\approx")[-1]
    ans = ans.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    ans = ans.replace("+\\infty", "\\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    ans = ans.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")
    return _strip_string(ans)


def eval_tuple(s):
    sl = s[1:-1].split(",")
    try:
        if s[0] == "(" and s[-1] == ")" and len(sl) > 1:
            s = ",".join([str(round(eval(str(latex2sympy(sub))), 2)) if "infty" not in sub and sub not in ["a", "-a"] else sub for sub in sl])
            return f"({s})"
        elif s[0] == "[" and s[-1] == "]" and len(sl) > 1:
            s = ",".join([str(round(eval(str(latex2sympy(sub))), 2)) if "infty" not in sub and sub not in ["a", "-a"] else sub for sub in sl])
            return f"[{s}]"
    except Exception:
        return s
    return s


def is_equal(asw: str, gt_asw: str) -> bool:
    asw = asw.lower()
    gt_asw = gt_asw.lower()
    if asw.replace(" ", "") == "" or gt_asw.replace(" ", "") == "":
        return False
    if gt_asw.strip() == asw.strip():
        return True
    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)
    if gt_asw == asw:
        return True
    try:
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
            return True
        else:
            return False
    except Exception:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _letter_match_official(segment: str, c: str) -> bool:
    return (
        segment.startswith(f"{c}")
        or f"\\{c}" in segment
        or f"/{c}" in segment
        or f"({c})" in segment
        or f"*{c}" in segment
        or f":{c}" in segment
        or f"box{{{c}}}" in segment
        or f"ircled{c}" in segment
        or f"ircle{{{c}}}" in segment
        or f"\\u{c}" in segment
    )


def scivqr_extract_model_answer(response: str, choices: Iterable[Any]) -> str:
    """Official SciVQR model-answer extraction.

    This function deliberately preserves the official script's control flow,
    including capitalization handling and final ``find_math_answer`` cleanup.
    """

    choices = list(choices or [])
    model_answer = str(response).strip()
    for c in "ABCDE":
        if (
            model_answer.endswith(f" {c}.")
            or model_answer.endswith(f" ({c}).")
            or model_answer.startswith(f"{c}\n")
            or model_answer.startswith(f"({c})\n")
            or model_answer.startswith(f"({c}) {c}\n")
            or model_answer.endswith(f"\\{c}")
            or model_answer.endswith(f":{c}")
        ):
            model_answer = c
    if is_number(model_answer.split("is ")[-1].rstrip(".")):
        model_answer = model_answer.split("is ")[-1].rstrip(".")
    if "oxed{" not in model_answer:
        if len(choices) > 0:
            for flag in ["the final answer is", "the answer is", "the correct answer is", "the answer should be"]:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    if ":\n\n" in model_answer:
                        model_answer = model_answer.split(":\n\n")[1].split(". ")[0]
                        for c in "ABCDE":
                            if _letter_match_official(model_answer, c):
                                model_answer = c
                    elif ":\n" in model_answer:
                        if _letter_match_official(model_answer, c):
                            model_answer = c
                    else:
                        model_answer = model_answer.split("\n")[0].split(". ")[0]

                flag = flag.replace("the", "The")
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    if ":\n\n" in model_answer and len(choices) > 0:
                        model_answer = model_answer.split(":\n\n")[1].split(". ")[0]
                        for c in "ABCDE":
                            if _letter_match_official(model_answer, c) and len(choices) > 0:
                                model_answer = c
                    elif ":\n" in model_answer and len(choices) > 0:
                        if _letter_match_official(model_answer, c) and len(choices) > 0:
                            model_answer = c
                    else:
                        model_answer = model_answer.split("\n")[0].split(". ")[0]
        else:
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
    return model_answer


def _official_gt_letter(choices: Iterable[Any], answer_value: str) -> str:
    choices = list(choices or [])
    if len(choices) > 0:
        sequential_characters = [chr(ord("A") + i) for i in range(len(choices))]
        try:
            return sequential_characters[choices.index(answer_value)]
        except Exception:
            return ""
    return ""


def scivqr_score_official(
    answer_value: str,
    choices: Iterable[Any],
    response: str,
    model_answer: Optional[str] = None,
    regen_answer: bool = True,
) -> Dict[str, Any]:
    choices = list(choices or [])
    gt_answer = _official_gt_letter(choices, answer_value)
    if model_answer is None or regen_answer:
        model_answer = scivqr_extract_model_answer(response, choices)
    if len(choices) > 0:
        correct = is_equal(gt_answer, model_answer) or is_equal(answer_value, model_answer)
        try:
            if type(latex2sympy(model_answer)) == Integer or type(latex2sympy(model_answer)) == Rational:
                if model_answer in gt_answer or model_answer in answer_value:
                    correct = True
        except Exception:
            pass
    else:
        correct = is_equal(gt_answer, model_answer) or is_equal(answer_value, model_answer)
    return {"model_answer": model_answer, "correct": bool(correct)}


def scivqr_doc_to_visual(doc):
    return [_decode_base64_image(doc["decoded_image"])]


def scivqr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    choices = doc.get("choices") or []
    if choices and len(choices) > 0:
        options = [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = "\n".join(f"{option}. {choice}" for option, choice in zip(options, choices))
        return f"{question}\n{choices_str}\nAnswer with the option's letter from the given choices directly."
    return f"{question}\nAnswer the question directly."


def scivqr_doc_to_text_open(doc, lmms_eval_specific_kwargs=None):
    return f"{doc['question']}\nAnswer the question directly."


def scivqr_doc_to_target(doc, model_specific_target_kwargs=None):
    choices = doc.get("choices") or []
    answer = doc["answer"]
    if len(choices) > 0:
        try:
            idx = list(choices).index(answer)
            return chr(ord("A") + idx)
        except ValueError:
            return answer
    return answer


def _record_from_doc(doc: Dict[str, Any], response: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    return {
        "question_id": str(doc.get("pid", doc.get("question_id", ""))),
        "pid": doc.get("pid"),
        "prompt": prompt if prompt is not None else scivqr_doc_to_text(doc),
        "response": response,
        "choices": list(doc.get("choices") or []),
        "answer": doc.get("answer", ""),
        "subject": doc.get("subject", "unknown"),
        "question_type": doc.get("question_type", "unknown"),
        "metadata": {},
    }


def _extract_choices_from_prompt(prompt: str) -> List[str]:
    choices = []
    for line in (prompt or "").splitlines():
        match = re.match(r"^([A-Z])\.\s*(.*)$", line.strip())
        if match:
            choices.append(match.group(2))
    return choices


def _doc_from_sample_context(sample: Dict[str, Any]) -> Dict[str, Any]:
    nested = sample.get("scivqr_acc") or sample.get("scivqr_open") or sample.get("scivqr_reasoning") or {}
    prompt = sample.get("input") or sample.get("prompt") or nested.get("prompt") or nested.get("question") or ""
    choices = nested.get("choices")
    if choices is None:
        choices = sample.get("choices")
    if choices is None:
        choices = _extract_choices_from_prompt(prompt)
    target = sample.get("target", nested.get("answer", sample.get("answer", "")))
    answer = target
    if isinstance(target, str) and len(target) == 1 and target.isalpha() and choices:
        idx = ord(target.upper()) - ord("A")
        if 0 <= idx < len(choices):
            answer = choices[idx]
    question = nested.get("question") or sample.get("question")
    if not question:
        question = re.split(r"\n[A-Z]\.\s+", prompt, maxsplit=1)[0].strip()
        question = question.split("\nChoices:")[0].strip()
        question = question.replace("\nAnswer the question directly.", "").strip()
        question = question.replace("\nAnswer with the option's letter from the given choices directly.", "").strip()
    return {
        "pid": nested.get("pid", sample.get("question_id", sample.get("doc_id"))),
        "question": question,
        "decoded_image": "",
        "choices": choices,
        "answer": answer,
        "solution": nested.get("gt_reason", sample.get("gt_reason", sample.get("solution", ""))),
        "question_type": nested.get("question_type", "multi-choice" if choices else "open"),
        "subject": nested.get("subject", sample.get("subject", "unknown")),
        "model_answer": nested.get("model_answer", sample.get("model_answer")),
    }


def scivqr_mcq_process_results(doc, results):
    if "__sample_context__" in doc:
        doc = _doc_from_sample_context(doc["__sample_context__"])
    response = results[0] if results else ""
    existing_model_answer = doc.get("model_answer")
    scored = scivqr_score_official(
        doc["answer"],
        doc.get("choices") or [],
        response,
        model_answer=existing_model_answer,
        regen_answer=existing_model_answer is None,
    )
    record = _record_from_doc(doc, response)
    record["model_answer"] = scored["model_answer"]
    record["correct"] = scored["correct"]
    return {
        "scivqr_acc": record,
        "scivqr_subject_acc": record,
    }


def scivqr_open_process_results(doc, results):
    if "__sample_context__" in doc:
        doc = _doc_from_sample_context(doc["__sample_context__"])
    response = results[0] if results else ""
    record = _record_from_doc(doc, response, prompt=scivqr_doc_to_text_open(doc))
    record["correct"] = None
    record["judge_response"] = None
    record["judge_model"] = None
    return {
        "scivqr_open": record,
        "needs_llm_judge": True,
    }


def get_judge_prompt(doc, prediction, target=None):
    answer = doc.get("answer", target if target is not None else "")
    return (
        "You are given a response from a model and the correct answer. "
        + "Your task is to determine if the model's response is correct. "
        + "You should only return 'true' if the response matches the answer. "
        + "If the answer is a floating-point number greater than 1, when it is represented in scientific notation, "
        + "a difference of up to 0.1 is allowed. Otherwise, return 'false'.\n"
        + f"Response: {prediction}\n"
        + f"Correct Answer: {answer}\n"
        + "Is the response correct? (true/false)"
    )


def get_judge_messages(doc, prediction, target=None):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_judge_prompt(doc, prediction, target)},
            ],
        }
    ]


def parse_judge_response(response: str) -> Optional[bool]:
    response = (response or "").lower()
    if "true" in response or "correct" in response:
        return True
    if "false" in response or "incorrect" in response:
        return False
    return None


def update_metrics_from_judge(doc, results, metrics, parsed, raw_response, model):
    updated = dict(metrics)
    correct = parsed
    updated["llm_judge_raw"] = raw_response
    updated["llm_judge_model"] = model
    updated["llm_judge_success"] = correct is not None
    if correct is not None:
        updated["llm_judge_score"] = int(bool(correct))
        updated["correct"] = bool(correct)
    else:
        updated["llm_judge_failed"] = True

    if "scivqr_open" in updated and isinstance(updated["scivqr_open"], dict):
        record = dict(updated["scivqr_open"])
        record["correct"] = bool(correct) if correct is not None else None
        record["judge_response"] = raw_response
        record["judge_model"] = model
        updated["scivqr_open"] = record
    return updated


def _original_official_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    output = {}
    for key, value in sample.items():
        if key in _FRAMEWORK_OUTPUT_KEYS:
            continue
        if key == "subject" and sample.get("__scivqr_subject_from_path"):
            continue
        output[key] = value
    return output


def _scivqr_mcq_official_output(sample: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    output = _original_official_sample(sample)
    if "question_id" not in output and record.get("question_id") is not None:
        output["question_id"] = str(record.get("question_id"))
    for key in ["prompt", "response", "choices", "answer"]:
        if key not in output and key in record:
            output[key] = record[key]
    output["model_answer"] = record.get("model_answer")
    output["correct"] = record.get("correct")
    return output


def _scivqr_open_official_output(sample: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    model_id = os.getenv("SCIVQR_TESTED_MODEL", "InternVL3-8B-Instruct")
    return {
        "question_id": str(sample.get("question_id", record.get("question_id", ""))),
        "prompt": sample.get("prompt", record.get("prompt", "")),
        "response": sample.get("response", record.get("response", "")),
        "choices": sample.get("choices", record.get("choices", [])),
        "answer": sample.get("answer", record.get("answer", "")),
        "model_id": model_id,
        "metadata": sample.get("metadata", {}),
        "correct": record.get("correct"),
    }


def save_judged_results(results: List[Dict[str, Any]], output_path) -> None:
    """Write SciVQR judged outputs in the official JSONL schemas."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in results:
            metrics = sample.get("metrics", {}) if isinstance(sample, dict) else {}
            if "scivqr_open" in metrics and isinstance(metrics["scivqr_open"], dict):
                output = _scivqr_open_official_output(sample, metrics["scivqr_open"])
            elif "scivqr_acc" in metrics and isinstance(metrics["scivqr_acc"], dict):
                output = _scivqr_mcq_official_output(sample, metrics["scivqr_acc"])
            else:
                output = _original_official_sample(sample)
            f.write(json.dumps(output, ensure_ascii=False, default=str) + "\n")


def _accuracy_from_records(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("correct")) / len(records)


def scivqr_aggregate_accuracy(results):
    return _accuracy_from_records(results)


def scivqr_aggregate_subject_accuracy(results):
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    for result in results:
        subject = result.get("subject", "unknown")
        subject_total[subject] += 1
        if result.get("correct"):
            subject_correct[subject] += 1
    return {
        subject: (subject_correct[subject] / subject_total[subject] if subject_total[subject] else 0.0)
        for subject in SUBJECTS
        if subject_total[subject] > 0
    }


def scivqr_aggregate_open_results(results):
    judged = [r for r in results if r.get("correct") is not None]
    if not judged:
        return {"mode": "generation_only", "total_samples": len(results), "accuracy": None}
    return {
        "mode": "judged",
        "accuracy": _accuracy_from_records(judged),
        "total_correct": sum(1 for r in judged if r.get("correct")),
        "total_samples": len(judged),
    }


def scivqr_standalone_aggregate_accuracy(extracted_data: List[Dict[str, Any]]):
    if not extracted_data:
        return {"accuracy": 0.0, "total_samples": 0}
    judged = [r for r in extracted_data if r.get("correct") is not None]
    if not judged:
        return {"mode": "generation_only", "total_samples": len(extracted_data), "accuracy": None}
    return {
        "accuracy": _accuracy_from_records(judged),
        "subject_accuracy": scivqr_aggregate_subject_accuracy(judged),
        "total_correct": sum(1 for r in judged if r.get("correct")),
        "total_samples": len(judged),
    }
