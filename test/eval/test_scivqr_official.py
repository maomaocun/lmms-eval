from types import SimpleNamespace
from pathlib import Path
import json
import os
import tempfile
import unittest

from lmms_eval.cli.judge_cmd import _get_output_path, _resolve_input_files, _scivqr_reasoning_batch_paths, _strip_openai_chat_completions_url, _write_scivqr_mcq_metrics_json
from lmms_eval.llm_judge.protocol import Response, ServerConfig
from lmms_eval.llm_judge.providers.openai import OpenAIProvider
from lmms_eval.llm_judge.standalone import JudgeRunner
from lmms_eval.tasks.scivqr import utils as scivqr_utils
from lmms_eval.tasks.scivqr.reasoning import utils as scivqr_reasoning_utils


class _FakeProvider:
    def __init__(self, content):
        self.content = content
        self.config = ServerConfig(model_name="fake-judge", max_tokens=16)
        self.last_request = None

    def evaluate(self, request):
        self.last_request = request
        return Response(content=self.content, model_used="fake-judge")


class _RecordingOpenAIProvider(OpenAIProvider):
    def __init__(self):
        super().__init__(ServerConfig(model_name="fake-judge"))
        self.api_key = "dummy"
        self.api_urls = ["http://localhost:8001/v1"]
        self.use_client = False
        self.payload = None

    def _make_request(self, payload, timeout, url=None):
        self.payload = payload
        return {"choices": [{"message": {"content": "true"}}], "model": payload["model"]}


class TestSciVQROfficialAdapters(unittest.TestCase):
    def test_scivqr_official_answer_extraction_edges(self):
        self.assertEqual(
            scivqr_utils.scivqr_score_official("blue", ["red", "blue"], "The final answer is: B"),
            {"model_answer": "b", "correct": True},
        )
        self.assertEqual(
            scivqr_utils.scivqr_score_official("blue", ["red", "blue"], "B. blue"),
            {"model_answer": "b.blue", "correct": False},
        )
        self.assertEqual(
            scivqr_utils.scivqr_score_official("blue", ["red", "blue"], "ignored", model_answer="B", regen_answer=False),
            {"model_answer": "B", "correct": True},
        )

    def test_scivqr_open_judge_parser_preserves_official_order(self):
        self.assertIs(scivqr_utils.parse_judge_response("true"), True)
        self.assertIs(scivqr_utils.parse_judge_response("false"), False)
        self.assertIs(scivqr_utils.parse_judge_response("incorrect"), True)

    def test_scivqr_accuracy_uses_official_unrounded_float(self):
        records = [{"correct": True}] + [{"correct": False} for _ in range(5)]
        self.assertEqual(scivqr_utils.scivqr_aggregate_accuracy(records), 1 / 6)
        self.assertEqual(scivqr_utils.scivqr_standalone_aggregate_accuracy(records)["accuracy"], 1 / 6)

    def test_scivqr_mcq_process_docs_is_officially_unfiltered(self):
        dataset = object()
        self.assertIs(scivqr_utils.scivqr_process_docs_mcq(dataset), dataset)

    def test_scivqr_reasoning_parser_and_aggregation(self):
        raw = "{'Faithfulness ': 8.0, 'Informativeness': 7, 'Repetition&Redundancy': 9, 'Hallucination': 8, 'Missing': 6, 'Overall': 7.6}"
        parsed = scivqr_reasoning_utils.parse_judge_response(raw)
        summary = scivqr_reasoning_utils.scivqr_reasoning_aggregate_scores([{"parsed_score": parsed}])
        self.assertEqual(summary["Faithfulness"], 8.0)
        self.assertEqual(summary["Overall"], 7.6)
        self.assertEqual(summary["judged_samples"], 1)

    def test_standalone_judge_uses_scivqr_custom_parser(self):
        doc = {
            "pid": 1,
            "question": "Question?",
            "answer": "answer",
            "choices": None,
            "subject": "math",
            "question_type": "open",
        }
        runner = JudgeRunner()
        runner._judge_provider = _FakeProvider("true")
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_open_process_results))
        runner._current_task_name = "scivqr_open"

        metrics = scivqr_utils.scivqr_open_process_results(doc, ["answer"])
        judged = runner._apply_llm_judge(doc, ["answer"], metrics, target=doc["answer"])

        self.assertEqual(judged["llm_judge_score"], 1)
        self.assertIs(judged["scivqr_open"]["correct"], True)
        self.assertIsNone(runner._judge_provider.last_request.config.temperature)
        self.assertIsNone(runner._judge_provider.last_request.config.max_tokens)
        self.assertEqual(runner._judge_provider.last_request.config.num_retries, 2)
        self.assertEqual(runner._judge_provider.last_request.config.retry_delay, 2)
        self.assertEqual(
            runner._judge_provider.last_request.messages[0]["content"],
            [{"type": "text", "text": scivqr_utils.get_judge_prompt(doc, "answer", doc["answer"])}],
        )

    def test_standalone_judge_accepts_official_scivqr_result_format(self):
        sample = {
            "question_id": "official-1",
            "prompt": "What color is the sky?\nA. red\nB. blue\nAnswer with the option's letter from the given choices directly.",
            "response": "The final answer is: B",
            "choices": ["red", "blue"],
            "answer": "blue",
            "metadata": {},
        }
        task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_mcq_process_results))
        runner = JudgeRunner()
        runner._current_task = task
        runner._current_task_name = "scivqr_mcq"

        judged = runner._judge_sample(sample, task, scivqr_utils.scivqr_mcq_process_results)

        record = judged["metrics"]["scivqr_acc"]
        self.assertIs(record["correct"], True)
        self.assertEqual(record["question_id"], "official-1")
        self.assertEqual(record["response"], sample["response"])
        self.assertEqual(record["choices"], sample["choices"])
        self.assertEqual(record["answer"], "blue")

    def test_standalone_reasoning_official_format_uses_dataset_solution(self):
        raw = "{'Faithfulness ': 9, 'Informativeness': 8, 'Repetition&Redundancy': 9, 'Hallucination': 8, 'Missing': 7, 'Overall': 8.2}"
        sample = {
            "question_id": "42",
            "prompt": "Dataset question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
            "response": "model reasoning",
            "answer": "two",
        }
        dataset_doc = {
            "pid": 42,
            "question": "Dataset question?",
            "solution": "official ground-truth reasoning",
            "answer": "two",
            "choices": ["one", "two"],
            "subject": "math",
            "question_type": "multi-choice",
        }
        task = SimpleNamespace(
            config=SimpleNamespace(process_results=scivqr_reasoning_utils.scivqr_reasoning_process_results),
            eval_docs_no_media=[dataset_doc],
        )
        runner = JudgeRunner()
        runner._judge_provider = _FakeProvider(raw)
        runner._current_task = task
        runner._current_task_name = "scivqr_reasoning"

        judged = runner._judge_sample(sample, task, scivqr_reasoning_utils.scivqr_reasoning_process_results)

        self.assertEqual(judged["metrics"]["scivqr_reasoning"]["gt_reason"], "official ground-truth reasoning")
        self.assertEqual(
            runner._judge_provider.last_request.messages,
            scivqr_reasoning_utils.get_judge_messages(dataset_doc, "model reasoning", dataset_doc["solution"]),
        )

    def test_standalone_reasoning_uses_subject_question_before_local_question_id(self):
        raw = "{'Faithfulness ': 9, 'Informativeness': 8, 'Repetition&Redundancy': 9, 'Hallucination': 8, 'Missing': 7, 'Overall': 8.2}"
        sample = {
            "question_id": "1",
            "subject": "astronomy",
            "__scivqr_subject_from_path": True,
            "prompt": "Sky question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
            "response": "astronomy reasoning",
            "answer": "two",
        }
        astronomy_doc = {
            "pid": 99,
            "question": "Sky question?",
            "solution": "astronomy ground-truth reasoning",
            "answer": "two",
            "choices": ["one", "two"],
            "subject": "astronomy",
            "question_type": "multi-choice",
        }
        task = SimpleNamespace(
            config=SimpleNamespace(process_results=scivqr_reasoning_utils.scivqr_reasoning_process_results),
            eval_docs_no_media=[
                {
                    "pid": 1,
                    "question": "Physics question?",
                    "solution": "wrong physics reasoning",
                    "answer": "two",
                    "choices": ["one", "two"],
                    "subject": "physics",
                    "question_type": "multi-choice",
                },
                astronomy_doc,
            ],
        )
        runner = JudgeRunner()
        runner._judge_provider = _FakeProvider(raw)
        runner._current_task = task
        runner._current_task_name = "scivqr_reasoning"

        judged = runner._judge_sample(sample, task, scivqr_reasoning_utils.scivqr_reasoning_process_results)

        self.assertEqual(judged["metrics"]["scivqr_reasoning"]["gt_reason"], "astronomy ground-truth reasoning")
        self.assertEqual(
            runner._judge_provider.last_request.messages,
            scivqr_reasoning_utils.get_judge_messages(astronomy_doc, "astronomy reasoning", astronomy_doc["solution"]),
        )

    def test_judge_cli_resolves_official_scivqr_subject_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for subject in ["math", "physics", "astronomy"]:
                (root / f"{subject}_results.jsonl").write_text("{}", encoding="utf-8")

            resolved = _resolve_input_files(str(root), ["scivqr_mcq"])

        self.assertEqual([item[0] for item in resolved], ["scivqr_mcq", "scivqr_mcq", "scivqr_mcq"])
        self.assertEqual([item[1].name for item in resolved], ["math_results.jsonl", "physics_results.jsonl", "astronomy_results.jsonl"])

    def test_judge_cli_resolves_official_scivqr_wildcard_with_explicit_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for subject in ["math", "physics"]:
                (root / f"{subject}_results.jsonl").write_text("{}", encoding="utf-8")

            resolved = _resolve_input_files(str(root / "*_results.jsonl"), ["scivqr_open"])

        self.assertEqual([item[0] for item in resolved], ["scivqr_open", "scivqr_open"])
        self.assertEqual([item[1].name for item in resolved], ["math_results.jsonl", "physics_results.jsonl"])

    def test_judge_cli_uses_official_scivqr_output_paths(self):
        old_model_id = os.environ.get("SCIVQR_TESTED_MODEL")
        old_split_id = os.environ.get("SCIVQR_SPLIT_ID")
        os.environ["SCIVQR_TESTED_MODEL"] = "OfficialModel"
        os.environ["SCIVQR_SPLIT_ID"] = "3"
        try:
            self.assertEqual(
                _get_output_path(Path("math_results.jsonl"), None, "/tmp/out", "scivqr_open"),
                Path("/tmp/out/OfficialModel/math_results.jsonl"),
            )
            self.assertEqual(
                _get_output_path(Path("math_results.jsonl"), None, "/tmp/out", "scivqr_reasoning"),
                Path("/tmp/out/Evaluation-Chunk3.json"),
            )
        finally:
            if old_model_id is None:
                os.environ.pop("SCIVQR_TESTED_MODEL", None)
            else:
                os.environ["SCIVQR_TESTED_MODEL"] = old_model_id
            if old_split_id is None:
                os.environ.pop("SCIVQR_SPLIT_ID", None)
            else:
                os.environ["SCIVQR_SPLIT_ID"] = old_split_id

    def test_judge_cli_writes_official_scivqr_mcq_metrics_json(self):
        summary = {"subject_accuracy": {"math": 0.25, "physics": 0.5}}
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_scivqr_mcq_metrics_json(tmpdir, None, summary)
            saved = json.loads((Path(tmpdir) / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(saved, {"math": 0.25, "physics": 0.5})

    def test_judge_cli_uses_official_scivqr_reasoning_batch_paths(self):
        requests_jsonl, result_ndjson, result_json = _scivqr_reasoning_batch_paths("/tmp/scivqr", "o1", 4)
        self.assertEqual(requests_jsonl, Path("/tmp/scivqr/uploads/o1/requests_chunk4.jsonl"))
        self.assertEqual(result_ndjson, Path("/tmp/scivqr/results/o1_results/output_chunk4.ndjson"))
        self.assertEqual(result_json, Path("/tmp/scivqr/results/o1_results/Evaluation-Chunk4.json"))
        self.assertEqual(
            _strip_openai_chat_completions_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1",
        )

    def test_scivqr_mcq_save_uses_official_jsonl_shape(self):
        sample = {
            "question_id": "1",
            "prompt": "Question?\nA. red\nB. blue",
            "response": "B",
            "choices": ["red", "blue"],
            "answer": "blue",
            "subject": "math",
            "__scivqr_subject_from_path": True,
            "metrics": {
                "scivqr_acc": {
                    "question_id": "1",
                    "prompt": "Question?\nA. red\nB. blue",
                    "response": "B",
                    "choices": ["red", "blue"],
                    "answer": "blue",
                    "model_answer": "b",
                    "correct": True,
                }
            },
        }
        runner = JudgeRunner()
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_mcq_process_results))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "math_results.jsonl"
            runner.save_results([sample], output)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertNotIn("metrics", saved)
        self.assertNotIn("subject", saved)
        self.assertNotIn("metadata", saved)
        self.assertEqual(saved["model_answer"], "b")
        self.assertIs(saved["correct"], True)

    def test_scivqr_mcq_save_preserves_original_metadata(self):
        sample = {
            "question_id": "1",
            "prompt": "Question?\nA. red\nB. blue",
            "response": "B",
            "choices": ["red", "blue"],
            "answer": "blue",
            "metadata": {"source": "official"},
            "metrics": {
                "scivqr_acc": {
                    "question_id": "1",
                    "prompt": "Question?\nA. red\nB. blue",
                    "response": "B",
                    "choices": ["red", "blue"],
                    "answer": "blue",
                    "model_answer": "b",
                    "correct": True,
                }
            },
        }
        runner = JudgeRunner()
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_mcq_process_results))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "math_results.jsonl"
            runner.save_results([sample], output)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(saved["metadata"], {"source": "official"})

    def test_scivqr_open_save_uses_official_jsonl_shape(self):
        sample = {
            "question_id": "2",
            "prompt": "Open question?",
            "response": "42",
            "choices": [],
            "answer": "42",
            "model_id": "NonOfficialInputModel",
            "metadata": {},
            "metrics": {
                "scivqr_open": {
                    "question_id": "2",
                    "prompt": "Open question?",
                    "response": "42",
                    "choices": [],
                    "answer": "42",
                    "correct": True,
                }
            },
        }
        runner = JudgeRunner()
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_open_process_results))
        old_model_id = os.environ.get("SCIVQR_TESTED_MODEL")
        os.environ["SCIVQR_TESTED_MODEL"] = "OfficialModel"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "math_results.jsonl"
                runner.save_results([sample], output)
                saved = json.loads(output.read_text(encoding="utf-8"))
        finally:
            if old_model_id is None:
                os.environ.pop("SCIVQR_TESTED_MODEL", None)
            else:
                os.environ["SCIVQR_TESTED_MODEL"] = old_model_id

        self.assertEqual(
            list(saved.keys()),
            ["question_id", "prompt", "response", "choices", "answer", "model_id", "metadata", "correct"],
        )
        self.assertEqual(saved["model_id"], "OfficialModel")
        self.assertIs(saved["correct"], True)

    def test_scivqr_open_save_defaults_to_official_tested_model(self):
        sample = {
            "question_id": "2",
            "prompt": "Open question?",
            "response": "42",
            "choices": [],
            "answer": "42",
            "model_id": "NonOfficialInputModel",
            "metadata": {},
            "metrics": {
                "scivqr_open": {
                    "question_id": "2",
                    "prompt": "Open question?",
                    "response": "42",
                    "choices": [],
                    "answer": "42",
                    "correct": True,
                }
            },
        }
        runner = JudgeRunner()
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_utils.scivqr_open_process_results))
        old_model_id = os.environ.get("SCIVQR_TESTED_MODEL")
        if old_model_id is not None:
            os.environ.pop("SCIVQR_TESTED_MODEL", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "math_results.jsonl"
                runner.save_results([sample], output)
                saved = json.loads(output.read_text(encoding="utf-8"))
        finally:
            if old_model_id is not None:
                os.environ["SCIVQR_TESTED_MODEL"] = old_model_id

        self.assertEqual(saved["model_id"], "InternVL3-8B-Instruct")

    def test_scivqr_reasoning_save_uses_official_json_shape(self):
        sample = {
            "question_id": "3",
            "metrics": {
                "scivqr_reasoning": {
                    "question_id": "3",
                    "question": "Reasoning question?",
                    "gt_reason": "ground truth reasoning",
                    "response": "model reasoning",
                    "score": "{'Overall': 8.0}",
                }
            },
        }
        runner = JudgeRunner()
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_reasoning_utils.scivqr_reasoning_process_results))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "Evaluation-Chunk0.json"
            runner.save_results([sample], output)
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            saved,
            [
                {
                    "question_id": "3",
                    "question": "Reasoning question?",
                    "gt_reason": "ground truth reasoning",
                    "response": "model reasoning",
                    "score": "{'Overall': 8.0}",
                }
            ],
        )

    def test_openai_provider_omits_officially_unspecified_fields(self):
        provider = _RecordingOpenAIProvider()
        provider.evaluate(
            SimpleNamespace(
                messages=scivqr_utils.get_judge_messages({"answer": "42"}, "42"),
                images=None,
                config=ServerConfig(model_name="Qwen2.5-72B-Instruct", temperature=None, max_tokens=None),
            )
        )
        self.assertNotIn("temperature", provider.payload)
        self.assertNotIn("max_tokens", provider.payload)
        self.assertEqual(provider.payload["model"], "Qwen2.5-72B-Instruct")

    def test_standalone_judge_uses_scivqr_reasoning_payload(self):
        raw = "{'Faithfulness ': 8.0, 'Informativeness': 7, 'Repetition&Redundancy': 9, 'Hallucination': 8, 'Missing': 6, 'Overall': 7.6}"
        doc = {
            "pid": 1,
            "question": "Question?",
            "solution": "ground truth reasoning",
            "answer": "answer",
            "choices": [],
            "subject": "math",
            "question_type": "open",
        }
        runner = JudgeRunner()
        runner._judge_provider = _FakeProvider(raw)
        runner._current_task = SimpleNamespace(config=SimpleNamespace(process_results=scivqr_reasoning_utils.scivqr_reasoning_process_results))
        runner._current_task_name = "scivqr_reasoning"

        metrics = scivqr_reasoning_utils.scivqr_reasoning_process_results(doc, ["model reasoning"])
        judged = runner._apply_llm_judge(doc, ["model reasoning"], metrics, target=doc["solution"])

        self.assertEqual(judged["llm_judge_score"], 7.6)
        self.assertEqual(runner._judge_provider.last_request.config.model_name, "gpt-4o")
        self.assertEqual(runner._judge_provider.last_request.config.temperature, 0.7)
        self.assertEqual(runner._judge_provider.last_request.config.max_tokens, 5120)
        self.assertEqual(
            runner._judge_provider.last_request.messages,
            scivqr_reasoning_utils.get_judge_messages(doc, "model reasoning", doc["solution"]),
        )

    def test_scivqr_reasoning_builds_official_batch_requests(self):
        samples = [
            {
                "question_id": "q1",
                "prompt": "Intro \\boxed{}\nWhat is x?\nChoices:\nA. one\nB. two",
                "response": "model reasoning",
            }
        ]
        dataset_docs = [{"pid": "q1", "question": "What is x?", "solution": "ground truth reasoning"}]
        data = scivqr_reasoning_utils.build_official_reasoning_items(
            samples,
            dataset_docs,
            strict_official_prompt=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = Path(tmpdir) / "uploads" / "gpt-4o" / "requests_chunk0.jsonl"
            meta = scivqr_reasoning_utils.write_official_reasoning_batch_requests(
                data,
                request_path,
                split_id=0,
                num_chunk=1,
                uuid_fn=lambda: "fixed-uuid",
            )
            saved = json.loads(request_path.read_text(encoding="utf-8").strip())

        self.assertEqual(meta["id_mapping"], {"fixed-uuid": 0})
        self.assertEqual(saved["custom_id"], "fixed-uuid")
        self.assertEqual(saved["method"], "POST")
        self.assertEqual(saved["url"], "/v1/chat/completions")
        self.assertEqual(saved["body"]["model"], "gpt-4o")
        self.assertEqual(saved["body"]["temperature"], 0.7)
        self.assertEqual(saved["body"]["max_tokens"], 5120)
        self.assertEqual(
            saved["body"]["messages"],
            scivqr_reasoning_utils.get_judge_messages(
                {"solution": "ground truth reasoning"},
                "model reasoning",
                "ground truth reasoning",
            ),
        )

    def test_scivqr_reasoning_batch_items_support_lmms_eval_prompt_fallback(self):
        samples = [
            {
                "question_id": "42",
                "prompt": "Dataset question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
                "response": "model reasoning",
            }
        ]
        dataset_docs = [
            {
                "pid": 42,
                "question": "Dataset question?",
                "solution": "official ground-truth reasoning",
            }
        ]

        self.assertEqual(
            scivqr_reasoning_utils.build_official_reasoning_items(
                samples,
                dataset_docs,
                strict_official_prompt=True,
            ),
            [],
        )
        self.assertEqual(
            scivqr_reasoning_utils.build_official_reasoning_items(samples, dataset_docs),
            [
                {
                    "question_id": "42",
                    "question": "Dataset question?",
                    "gt_reason": "official ground-truth reasoning",
                    "response": "model reasoning",
                }
            ],
        )

    def test_scivqr_reasoning_batch_fallback_does_not_override_empty_pid_solution(self):
        samples = [
            {
                "question_id": "1",
                "prompt": "Repeated question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
                "response": "model reasoning",
            },
            {
                "prompt": "Repeated question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
                "response": "model reasoning without pid",
            },
        ]
        dataset_docs = [
            {"pid": 1, "question": "Repeated question?", "solution": ""},
            {"pid": 2, "question": "Repeated question?", "solution": "solved duplicate"},
        ]

        self.assertEqual(
            scivqr_reasoning_utils.build_official_reasoning_items(samples, dataset_docs),
            [
                {
                    "question_id": None,
                    "question": "Repeated question?",
                    "gt_reason": "solved duplicate",
                    "response": "model reasoning without pid",
                }
            ],
        )

    def test_scivqr_reasoning_batch_fallback_prefers_subject_question_over_local_id(self):
        samples = [
            {
                "question_id": "1",
                "subject": "astronomy",
                "prompt": "Sky question?\nA. one\nB. two\nAnswer with the option's letter from the given choices directly.",
                "response": "astronomy reasoning",
            }
        ]
        dataset_docs = [
            {"pid": 1, "subject": "physics", "question": "Physics question?", "solution": "physics solution"},
            {"pid": 99, "subject": "astronomy", "question": "Sky question?", "solution": "astronomy solution"},
        ]

        self.assertEqual(
            scivqr_reasoning_utils.build_official_reasoning_items(samples, dataset_docs),
            [
                {
                    "question_id": "1",
                    "question": "Sky question?",
                    "gt_reason": "astronomy solution",
                    "response": "astronomy reasoning",
                }
            ],
        )

    def test_scivqr_reasoning_processes_official_batch_ndjson(self):
        data = [
            {"question_id": "1", "question": "Q1", "gt_reason": "G1", "response": "R1"},
            {"question_id": "2", "question": "Q2", "gt_reason": "G2", "response": "R2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            ndjson_path = Path(tmpdir) / "output_chunk0.ndjson"
            ndjson_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "custom_id": "u2",
                                "response": {
                                    "body": {
                                        "choices": [
                                            {"message": {"content": " {'Overall': 7.0} "}},
                                        ]
                                    }
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "custom_id": "u1",
                                "response": {
                                    "body": {
                                        "choices": [
                                            {"message": {"content": "{'Overall': 8.0}"}},
                                        ]
                                    }
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = Path(tmpdir) / "Evaluation-Chunk0.json"
            saved = scivqr_reasoning_utils.write_official_reasoning_results_from_ndjson(
                data,
                {"u1": 0, "u2": 1},
                ndjson_path,
                output_path,
                start=0,
                end=2,
            )
            on_disk = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved, on_disk)
        self.assertEqual([item["question_id"] for item in saved], ["1", "2"])
        self.assertEqual(saved[0]["score"], "{'Overall': 8.0}")
        self.assertEqual(saved[1]["score"], "{'Overall': 7.0}")

    def test_scivqr_reasoning_official_batch_api_sequence(self):
        class _Obj:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class _Content:
            def iter_bytes(self):
                yield b'{"custom_id": "u1"}\n'

        class _Files:
            def __init__(self):
                self.created = None

            def create(self, file, purpose):
                self.created = {"purpose": purpose, "content": file.read()}
                return _Obj(id="file-1")

            def content(self, file_id):
                self.content_file_id = file_id
                return _Content()

        class _Batches:
            def __init__(self):
                self.created = None
                self.polls = [
                    _Obj(status="running"),
                    _Obj(status="completed", output_file_id="out-1"),
                ]

            def create(self, input_file_id, endpoint, completion_window):
                self.created = {
                    "input_file_id": input_file_id,
                    "endpoint": endpoint,
                    "completion_window": completion_window,
                }
                return _Obj(id="batch-1")

            def retrieve(self, batch_id):
                self.batch_id = batch_id
                return self.polls.pop(0)

        client = _Obj(files=_Files(), batches=_Batches())
        with tempfile.TemporaryDirectory() as tmpdir:
            request_path = Path(tmpdir) / "requests_chunk0.jsonl"
            request_path.write_text('{"body": {}}\n', encoding="utf-8")
            batch_id = scivqr_reasoning_utils.submit_official_reasoning_batch(client, request_path)
            sleeps = []
            batch = scivqr_reasoning_utils.wait_for_official_reasoning_batch(
                client,
                batch_id,
                interval=3,
                sleep_fn=sleeps.append,
            )
            output_path = Path(tmpdir) / "output_chunk0.ndjson"
            scivqr_reasoning_utils.download_official_reasoning_batch_results(client, batch, output_path)
            output_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(batch_id, "batch-1")
        self.assertEqual(client.files.created["purpose"], "batch")
        self.assertEqual(client.files.created["content"], b'{"body": {}}\n')
        self.assertEqual(
            client.batches.created,
            {
                "input_file_id": "file-1",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        self.assertEqual(sleeps, [3])
        self.assertEqual(client.batches.batch_id, "batch-1")
        self.assertEqual(client.files.content_file_id, "out-1")
        self.assertEqual(output_text, '{"custom_id": "u1"}\n')


if __name__ == "__main__":
    unittest.main()
