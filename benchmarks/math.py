import asyncio
import inspect
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger

# Timeout for symbolic comparison (seconds)
SYMBOLIC_TIMEOUT = 5


class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        """Compare two expressions symbolically with timeout protection."""
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        def _do_comparison(a_str, b_str):
            """Perform the actual comparison - runs in a thread with timeout."""
            a_parsed = _parse(a_str)
            b_parsed = _parse(b_str)

            try:
                if simplify(a_parsed - b_parsed) == 0:
                    return True
            except:
                pass

            try:
                if isclose(N(a_parsed), N(b_parsed), abs_tol=1e-3):
                    return True
            except:
                pass
            return False

        # Run comparison with timeout to prevent hanging on complex expressions
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_comparison, a, b)
            try:
                return future.result(timeout=SYMBOLIC_TIMEOUT)
            except FuturesTimeoutError:
                # Comparison timed out - treat as not equal
                return False
            except Exception:
                return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        # Generate output with a timeout of 300 seconds
        return await asyncio.wait_for(graph(input_text), timeout=300)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )

            return input_text, output, expected_output, uni_score, cost

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
