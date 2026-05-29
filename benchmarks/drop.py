import asyncio
import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class DROPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        retry=retry_if_exception(lambda e: not isinstance(e, asyncio.CancelledError)),
        reraise=True
    )
    async def _generate_output(self, graph, input_text):
        # 300s timeout; timeouts also trigger retry (up to 3 times)
        return await asyncio.wait_for(graph(input_text), timeout=300)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")

        import time
        start_time = time.time()

        try:
            output, cost = await self._generate_output(graph, input_text)
            f1_scores = []

            for answer in answers:
                if answer.strip() != "":
                    output_parts = output.split("|")
                    for output_part in output_parts:
                        f1_score, _ = self.calculate_score(answer, output_part)
                        f1_scores.append(f1_score)

            uni_score = max(f1_scores)

            if uni_score < 0.3:
                self.log_mismatch(input_text, expected_output, output, output)

            return input_text, output, expected_output, uni_score, cost

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"TimeoutError after {elapsed:.1f}s. Input length: {len(input_text)} chars.")
            return None

        except asyncio.CancelledError:
            elapsed = time.time() - start_time
            logger.warning(f"CancelledError after {elapsed:.1f}s. Task was cancelled.")
            return None

        except Exception as e:
            elapsed = time.time() - start_time
            error_type = type(e).__name__
            error_msg = str(e)[:200]  # truncate error message
            logger.warning(f"[{error_type}] after {elapsed:.1f}s: {error_msg}")
            return None

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
