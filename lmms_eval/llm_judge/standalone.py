"""Standalone judge runner for JSONL files.

This module provides the core logic for judging model outputs from JSONL files
without regeneration, completely separating the generation and judging phases.
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from tqdm import tqdm

# Delay imports to avoid loading heavy dependencies
ProviderFactory = None
ServerConfig = None

def _get_provider_factory():
    global ProviderFactory
    if ProviderFactory is None:
        from lmms_eval.llm_judge.factory import ProviderFactory as PF
        ProviderFactory = PF
    return ProviderFactory

def _get_server_config():
    global ServerConfig
    if ServerConfig is None:
        from lmms_eval.llm_judge.protocol import ServerConfig as SC
        ServerConfig = SC
    return ServerConfig

# Delay import of TaskManager to avoid heavy imports
TaskManager = None

def _get_task_manager():
    global TaskManager
    if TaskManager is None:
        from lmms_eval.tasks import TaskManager as TM
        TaskManager = TM
    return TaskManager()


class JudgeRunner:
    """Runner for standalone judging from JSONL files.
    
    This class handles:
    1. Loading JSONL files with model outputs
    2. Loading the appropriate task's process_results function
    3. Applying rule-based or LLM-based judging
    4. Saving results with updated metrics
    
    Example:
        >>> runner = JudgeRunner(judge_mode="auto", judge_model="gpt-4o-mini")
        >>> results = runner.judge_file(Path("results.jsonl"), "mathvision_reason_testmini")
        >>> runner.save_results(results, Path("judged.jsonl"))
    """
    
    def __init__(
        self,
        judge_mode: str = "auto",
        judge_model: str = "gpt-4o-mini",
        judge_api_key: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        parallel: int = 1,
    ):
        """Initialize the judge runner.
        
        Args:
            judge_mode: Judging mode - "rule", "llm", or "auto" (rule first, LLM fallback)
            judge_model: Name of the LLM to use for judging
            judge_api_key: API key for the judge model
            judge_base_url: Base URL for the judge API
            parallel: Number of parallel workers (currently only for LLM judge)
        """
        self.judge_mode = judge_mode
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key or os.getenv("JUDGE_API_KEY")
        self.judge_base_url = judge_base_url or os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
        self.parallel = parallel
        self._task_manager = None
        self._judge_provider = None
        self._current_task_name = None
        logger.info(
            f"JudgeRunner initialized: mode={self.judge_mode}, model={self.judge_model}, "
            f"base_url={self.judge_base_url}, parallel={self.parallel}"
        )
        
    def judge_file(self, input_path: Path, task_name: str) -> List[Dict[str, Any]]:
        """Judge all samples in a JSONL file.
        
        Args:
            input_path: Path to JSONL file with model outputs
            task_name: Task name for loading process_results function
            
        Returns:
            List of judged samples with updated metrics
            
        Raises:
            ValueError: If task cannot be loaded
            FileNotFoundError: If input file doesn't exist
        """
        self._current_task_name = task_name
        
        # Load task and get process_results function
        logger.debug(f"Loading task: {task_name}")
        task = self._load_task(task_name)
        process_results_fn = task.config.process_results
        
        if process_results_fn is None:
            raise ValueError(f"Task {task_name} has no process_results function")
        
        logger.info(
            f"Task config [{task_name}]: dataset_path={getattr(task.config, 'dataset_path', 'N/A')}, "
            f"fewshot={getattr(task.config, 'num_fewshot', 'N/A')}, "
            f"metric_list={[m.get('metric') for m in getattr(task.config, 'metric_list', [])]}"
        )
        
        # Load samples
        logger.debug(f"Loading samples from {input_path}")
        samples = self._load_jsonl(input_path)
        logger.info(f"Loaded {len(samples)} samples")
        
        # Judge each sample concurrently
        logger.info(f"Starting concurrent judging with max_workers={self.parallel}")
        judged_samples = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            futures = {
                executor.submit(self._judge_sample, sample, task, process_results_fn): i
                for i, sample in enumerate(samples)
            }
            for future in tqdm(
                as_completed(futures),
                desc=f"Judging {task_name}",
                total=len(samples),
                miniters=10,
            ):
                idx = futures[future]
                judged_samples[idx] = future.result()

        logger.info(f"Finished judging {len(judged_samples)} samples")
        return judged_samples
    
    def _judge_sample(
        self,
        sample: Dict[str, Any],
        task: Any,
        process_results_fn: Callable,
    ) -> Dict[str, Any]:
        """Judge a single sample.
        
        Args:
            sample: Sample dict from JSONL (contains doc, filtered_resps, etc.)
            task: Task object with config
            process_results_fn: The task's process_results function
            
        Returns:
            Sample dict with added/updated metrics
        """
        doc = sample.get("doc", {})
        filtered_resps = sample.get("filtered_resps", [])
        target = sample.get("target", None)
        doc_id = sample.get("doc_id", "unknown")
        
        # Convert to list if single response
        if isinstance(filtered_resps, str):
            filtered_resps = [filtered_resps]
        
        # Initialize result
        result_sample = sample.copy()
        
        try:
            # Apply rule-based judging first (if not llm-only mode)
            if self.judge_mode in ("rule", "auto"):
                try:
                    metrics = process_results_fn(doc, filtered_resps)
                    result_sample["metrics"] = metrics
                    result_sample["judge_mode"] = "rule"
                except Exception as rule_err:
                    if self.judge_mode == "auto":
                        logger.debug(f"Sample {doc_id}: rule-based failed ({rule_err}), trying LLM judge")
                        metrics = {"rule_error": str(rule_err)}
                        result_sample["judge_mode"] = "rule_error"
                    else:
                        raise rule_err
                
                # Check if we need LLM fallback
                if self.judge_mode == "auto" and self._needs_llm_judge(metrics):
                    logger.debug(f"Sample {doc_id}: rule-based score low, trying LLM judge")
                    llm_metrics = self._apply_llm_judge(doc, filtered_resps, metrics, target=target, sample=sample)
                    result_sample["metrics"] = llm_metrics
                    result_sample["judge_mode"] = "llm_fallback"

            elif self.judge_mode == "llm":
                # LLM-only mode
                metrics = self._apply_llm_judge(doc, filtered_resps, {}, target=target, sample=sample)
                result_sample["metrics"] = metrics
                result_sample["judge_mode"] = "llm"
                
        except Exception as e:
            logger.warning(f"Error judging sample {doc_id}: {e}")
            result_sample["metrics"] = {"error": str(e), "judge_failed": True}
            result_sample["judge_mode"] = "error"
        
        return result_sample
    
    def _apply_llm_judge(
        self,
        doc: Dict[str, Any],
        results: List[str],
        fallback_metrics: Dict[str, Any],
        target: Optional[str] = None,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply LLM judge when rule-based fails or for LLM-only mode.
        
        Args:
            doc: Document dict with question, answer, etc.
            results: List of model responses
            fallback_metrics: Existing metrics from rule-based judging
            
        Returns:
            Updated metrics with LLM judge results
        """
        # Initialize judge provider if needed
        if self._judge_provider is None:
            self._init_judge_provider()
        
        # Extract question and answer
        question = self._extract_question(doc, sample)
        answer = str(doc.get("answer", ""))
        if not answer and target is not None:
            answer = str(target)
        prediction = results[0] if results else ""
        
        if not answer:
            logger.warning("No ground truth answer found in doc, skipping LLM judge")
            return {**fallback_metrics, "llm_judge_skipped": True}
        
        try:
            # Call LLM judge
            judge_result = self._judge_provider.evaluate_binary(
                question=question,
                answer=answer,
                prediction=prediction,
                output_format="0/1",
            )
            
            # Merge with fallback metrics
            metrics = fallback_metrics.copy()
            metrics["llm_judge_score"] = int(judge_result.get("result", 0))
            metrics["llm_judge_raw"] = judge_result.get("raw_response", "")
            metrics["llm_judge_model"] = judge_result.get("model", self.judge_model)
            metrics["llm_judge_success"] = judge_result.get("success", False)
            
            return metrics
            
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return {
                **fallback_metrics,
                "llm_judge_error": str(e),
                "llm_judge_failed": True,
            }
    
    def _init_judge_provider(self) -> None:
        """Initialize the LLM judge provider.
        
        Supports OpenAI, Azure, and local vLLM/SGLang servers via OpenAI-compatible API.
        For local vLLM, set JUDGE_BASE_URL to your vLLM endpoint (e.g., http://localhost:8000/v1).
        """
        # Delayed imports
        PF = _get_provider_factory()
        SC = _get_server_config()
        
        # For local vLLM/SGLang, use a dummy key if not provided
        # OpenAI client requires a key, but local servers often don't validate it
        api_key = self.judge_api_key or "dummy-key-for-local-vllm"
        
        # Detect if using local vLLM/SGLang
        is_local = any(x in self.judge_base_url for x in ["localhost", "127.0.0.1", ":8000", ":30000"])
        if is_local:
            logger.info(f"Detected local LLM server at {self.judge_base_url}")
        
        # Set env vars for the provider
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_URL"] = self.judge_base_url
        
        # Allow overriding API type for local servers
        api_type = os.getenv("JUDGE_API_TYPE", "openai")
        
        config = SC(
            model_name=self.judge_model,
            temperature=0.0,
            max_tokens=1024,
            max_concurrent=self.parallel,
        )
        logger.info(
            f"Judge ServerConfig: model={config.model_name}, temperature={config.temperature}, "
            f"max_tokens={config.max_tokens}, max_concurrent={config.max_concurrent}"
        )
        
        self._judge_provider = PF.create_provider(api_type=api_type, config=config)
        
        if is_local:
            logger.info(f"Initialized local LLM judge provider at {self.judge_base_url}")
        else:
            logger.info(f"Initialized cloud LLM judge provider with model: {self.judge_model}")
    
    def _extract_question(self, doc: Dict[str, Any], sample: Optional[Dict[str, Any]] = None) -> str:
        """Extract question text from doc.
        
        Tries multiple common field names. Falls back to sample input if doc is empty.
        """
        # Try common field names
        for key in ["question", "problem", "query", "prompt", "text"]:
            if key in doc and doc[key]:
                return str(doc[key])
        
        # Try to construct from available fields
        if "query_wo" in doc:  # MathVerse specific
            return str(doc["query_wo"])
        
        # Fallback to sample fields if doc is empty (common for offline JSONL judging)
        if sample is not None:
            for key in ["input", "prompt"]:
                if key in sample and sample[key]:
                    return str(sample[key])
        
        # Last resort: return string representation
        return str(doc)[:500]  # Limit length
    
    def _needs_llm_judge(self, metrics: Dict[str, Any]) -> bool:
        """Check if rule-based judge failed and LLM judge is needed.
        
        Args:
            metrics: Metrics dict from rule-based judging
            
        Returns:
            True if LLM judge should be applied
        """
        # Check common accuracy metrics
        for key in ["acc_score", "accuracy", "correct", "score", "exact_match"]:
            if key in metrics:
                val = metrics[key]
                # Consider 0 or False as needing fallback
                if val == 0 or val is False:
                    return True
                # For float values, check if very low
                if isinstance(val, float) and val < 0.1:
                    return True
        return False
    
    def _load_task(self, task_name: str) -> Any:
        """Load task by name.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task object
            
        Raises:
            ValueError: If task cannot be loaded
        """
        try:
            task_manager = _get_task_manager()
            task_dict = task_manager.load_task_or_group(task_name)
            tasks = list(task_dict.values())
            if not tasks:
                raise ValueError(f"Task {task_name} not found")
            return tasks[0]
        except Exception as e:
            raise ValueError(f"Failed to load task {task_name}: {e}")
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file.
        
        Args:
            path: Path to JSONL file
            
        Returns:
            List of sample dicts
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        
        return samples
    
    def compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate summary from judged results.
        
        Args:
            results: List of judged samples with metrics
            
        Returns:
            Dict mapping metric names to aggregated values (mean for numeric).
        """
        if not results:
            return {}
        
        metric_values: Dict[str, List[float]] = {}
        for sample in results:
            metrics = sample.get("metrics", {})
            for key, val in metrics.items():
                if isinstance(val, bool):
                    metric_values.setdefault(key, []).append(float(val))
                elif isinstance(val, (int, float)):
                    metric_values.setdefault(key, []).append(float(val))
                # Skip non-numeric metrics
        
        summary = {}
        for key, values in metric_values.items():
            if values:
                summary[key] = round(sum(values) / len(values), 4)
        return summary
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Save judged results to JSONL.
        
        Args:
            results: List of judged samples
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in results:
                # Handle non-serializable objects
                cleaned = self._clean_for_json(sample)
                f.write(json.dumps(cleaned, ensure_ascii=False, default=str) + "\n")
        
        logger.debug(f"Saved {len(results)} results to {output_path}")
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization.
        
        Recursively converts non-serializable objects to strings.
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
