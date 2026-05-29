import re
import string
from collections import Counter
from typing import Callable, List, Tuple

import asyncio

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class HotpotQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> str:
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

    async def _generate_output_with_retry(self, graph, input_text):
        """Retry up to 2 times with 300s timeout; return None to skip sample on failure."""
        max_retries = 2
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                return await asyncio.wait_for(graph(input_text), timeout=300)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout after 300s (attempt {retry_count + 1}/{max_retries})")
                last_error = asyncio.TimeoutError("Timeout after 300s")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2)
            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(f"API error (attempt {retry_count}/{max_retries}): {type(e).__name__}")
                if retry_count < max_retries:
                    await asyncio.sleep(2)

        # Max retries reached: skip sample
        logger.warning(f"Max retries ({max_retries}) reached. Skipping sample. Last error: {last_error}")
        return None

    async def evaluate_problem(self, problem: dict, graph: Callable):
        """Evaluate a single problem; return None to skip if API fails."""
        input_text = problem["question"]
        expected_output = problem["answer"]
        paragraphs = [item[1] for item in problem["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {input_text}\n\nAnswer:"

        result = await self._generate_output_with_retry(graph, inputs)

        # None means the sample should be skipped
        if result is None:
            return None

        output, cost = result
        score, extracted_output = self.calculate_score(expected_output, output)

        if (
            score < 0.3
        ):  # We set the threshold for collecting incorrect questions to 0.3, as F1 Score cannot be simply judged using 0-1
            self.log_mismatch(input_text, expected_output, output, extracted_output)

        return input_text, context_str, output, expected_output, score, cost

    def get_result_columns(self) -> List[str]:
        return ["question", "context", "prediction", "expected_output", "score", "cost"]
