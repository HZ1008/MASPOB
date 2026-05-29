import asyncio
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text.

        Uses a multi-step approach for fair comparison across different baselines:
        1. First look for explicit answer patterns (#### X, The answer is X, etc.)
        2. If not found, fall back to the last number in the text
        """
        text = str(text)

        # Step 1: Look for explicit answer patterns
        # These patterns are commonly used in CoT and structured outputs
        answer_patterns = [
            r'####\s*([\d,]+(?:\.\d+)?)',  # GSM8K standard format
            r'(?:the answer is|answer is)[:\s]*\$?([\d,]+(?:\.\d+)?)',
            r'(?:therefore|thus|so)[,\s]+(?:the )?answer is[:\s]*\$?([\d,]+(?:\.\d+)?)',
            r'\*\*([\d,]+(?:\.\d+)?)\*\*\s*(?:$|\.)',  # **number** at end
            r'(?:total|result|answer)[:\s]*\$?([\d,]+(?:\.\d+)?)\s*(?:$|\.)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    continue

        # Step 2: Fall back to last number (original behavior)
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", text)
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        # Generate output with a timeout of 300 seconds
        return await asyncio.wait_for(graph(input_text), timeout=300)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, float, float, float]:
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            output, cost = await self._generate_output(graph, input_text)
            predicted_number = self.extract_number(output)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except asyncio.TimeoutError:
            # Timeout: skip sample, exclude from score
            logger.warning(f"Timeout after 300s. Skipping sample.")
            return None

        except Exception as e:
            # API error after retries exhausted: skip sample, exclude from score
            logger.warning(f"API error after 2 retries. Skipping sample. Error: {e}")
            return None

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
