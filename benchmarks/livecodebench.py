import asyncio
import json
import os
import multiprocessing
import threading
import time
import base64
import zlib
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger
import sys
sys.path.append("..")
sys.path.append("benchmarks")
# ensure lcb_runner is installed
from scripts.utils.lcb_runner import run_test

# Key functions copied from LiveCodeBench official evaluation
#sys.set_int_max_str_digits(50000)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def check_correctness(sample, generation, timeout, debug=True):
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
        if debug:
            logger.warning(
                f"Global timeout: {sample.get('question_id', 'unknown')}"
            )

    return result[0], metadata_list[0]

def evaluate_generations_by_problem(args):
    problem_generations, sample, debug, timeout = args
    res = []
    metadata = []
    for generation in problem_generations:
        curr_res = [-2]
        try:
            curr_res, curr_metadata = check_correctness(
                sample, generation, timeout=timeout, debug=debug
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        except Exception as e:
            curr_metadata = {
                "error": repr(e),
                "error_code": -5,
                "error_message": "TestRunnerError",
            }
        finally:
            res.append(curr_res)
            metadata.append(curr_metadata)
    return res, metadata

class LiveCodeBench(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str, timeout: int = 6):
        super().__init__(name, file_path, log_path)
        self.timeout = timeout
        self.num_process_evaluate = min(16, os.cpu_count() or 4)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        result = []
        exception_occurred = []
        stop_event = threading.Event()

        def target():
            try:
                return_value = func(*args)
                result.append(return_value)
            except Exception as e:
                exception_occurred.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")
        if exception_occurred:
            raise exception_occurred[0]
        return result[0] if result else None
    def parse_code(self, prediction):
        prediction = prediction.split("```python")[-1]
        prediction = prediction.split("```")[0]
        return prediction
    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        """Load data from JSONL and convert to LiveCodeBench evaluation format."""
        raw_data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                raw_data.append(json.loads(line))
        
        # Convert to evaluation format
        processed_data = []
        for item in raw_data:
            try:
                # Handle private test cases (only use private test cases for evaluation)
                try:
                    private_tests = json.loads(item["private_test_cases"])
                except:
                    private_tests = json.loads(
                        pickle.loads(
                            zlib.decompress(
                                base64.b64decode(item["private_test_cases"].encode("utf-8"))
                            )
                        )
                    )
                
                # Build evaluation sample
                processed_item = {
                    "question": item["question_content"],
                    "input_output": json.dumps({
                        "inputs": [t["input"] for t in private_tests],
                        "outputs": [t["output"] for t in private_tests],
                        "fn_name": json.loads(item["metadata"]).get("func_name", None) if item["metadata"] else None
                    }),
                    "question_id": item['question_id'],
                    "canonical_solution": item.get("starter_code", ""),
                    "metadata": {
                        "difficulty": item.get("difficulty", "unknown"),
                        "platform": item.get("platform", "unknown"),
                        "original_data": item  # Keep original data
                    }
                }
                processed_data.append(processed_item)   
            
            except Exception as e:
                logger.error(f"Error processing data: {str(e)}")
                continue
        
        if specific_indices is not None:
            return [processed_data[i] for i in specific_indices if i < len(processed_data)]
        return processed_data

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, agent: Callable, prompt: str, entry_point:str, question_id: str = "") -> Tuple[str, float]:
        # entry_point = ""  # specify function name
        return await asyncio.wait_for(agent(prompt, entry_point, question_id), timeout=120)

    async def evaluate_problem(self, problem: dict, agent: Callable, save_path: str = None) -> Tuple[str, str, str, float, Dict, float]:
        question = problem["question"]
        question_id = problem["question_id"]
        
        try:
            logger.info(
                f"Start evaluating LiveCodeBench problem: {question_id}"
            )
            
            # Generate code
            entry_point = problem["metadata"].get("func_name", "wrapped_function") if problem["metadata"] else "wrapped_function"
            question_id = problem["question_id"]
            logger.info(f"entry_point: {entry_point}")
            prediction, cost = await self._generate_output(
                agent, question, entry_point, question_id
            )
            logger.info(
                f"Finished code generation, task: {question_id}, cost: {cost}"
            )
            prediction = self.parse_code(prediction)
            # Use LiveCodeBench evaluation logic
            sample = {
                "question": question,
                "input_output": problem["input_output"],
                "question_id": question_id
            }
            # logger.info(f"Start evaluating sample {sample['input_output']}")
            
            # Evaluate in a multiprocessing environment
            args = ([prediction], sample, False, self.timeout)
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                results, metadata = await loop.run_in_executor(
                    executor, evaluate_generations_by_problem, args
                )
            
            # Parse results
            logger.info(f"Test results: {results}")
            test_results = results[0]  # Take all test case results of the first (and only) generation
            test_metadata = metadata[0]
            passed = all(r == 1 for r in test_results)
            score = 1.0 if passed else 0.0
            
            # Build evaluation details
            evaluation_details = {
                "question_id": question_id,
                "test_results": test_results,
                "metadata": test_metadata,
                "execution_success": passed,
                "difficulty": problem.get("metadata", {}).get("difficulty", "unknown"),
                "platform": problem.get("metadata", {}).get("platform", "unknown")
            }
            
            # Build expected output for logging
            expected_output = {
                "question_id": question_id,
                "difficulty": evaluation_details["difficulty"],
                "platform": evaluation_details["platform"],
                "canonical_solution": problem.get("canonical_solution", "")
            }

            # Log failures
            if not passed:
                self.log_mismatch(
                    problem=question,
                    expected_output=json.dumps(expected_output),
                    prediction=prediction,
                    extracted_output=prediction,
                    extract_answer_code="N/A"
                )
                logger.warning(f"Task failed: {question_id}, score: {score}")
            else:
                logger.info(f"Task succeeded: {question_id}, score: {score}")

            result = (question, prediction, json.dumps(expected_output), score, evaluation_details, cost)
            
            # Save results
            if save_path:
                async with aiofiles.open(save_path, mode="a", encoding="utf-8") as file:
                    await file.write(json.dumps(result) + "\n")
            
            return result

        except asyncio.TimeoutError:
            logger.error(f"Code generation timeout: {question_id}")
            evaluation_details = {"question_id": question_id, "error": "Timeout"}
            return (question, "Timeout", "", 0.0, evaluation_details, 0.0)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Evaluation error: {question_id}, error: {e}")
            evaluation_details = {"question_id": question_id, "error": str(e)}
            return (
                question,
                f"Evaluation error: {str(e)}",
                "",
                0.0,
                evaluation_details,
                0.0,
            )

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return 0.0, ""

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "evaluation_details", "cost"]

    async def run_baseline_with_load_data(self, agent: Callable, past_data_path: str = None, max_concurrent_tasks: int = 10):
        all_data = await self.load_data()
        
        if not past_data_path:
            past_data_path = os.path.join(self.log_path, f"{self.name}_results.jsonl")
        
        # Load past results
        past_results = {}
        if os.path.exists(past_data_path):
            async with aiofiles.open(past_data_path, mode="r", encoding="utf-8") as file:
                async for line in file:
                    try:
                        result = json.loads(line)
                        # Use question text as key
                        past_results[result[0]] = result
                    except:
                        continue

        # Filter new problems
        new_data = [p for p in all_data if p["question"] not in past_results]
        
        if not new_data:
            logger.info("All problems have been evaluated")
            return None, None, None

        logger.info(
            f"{len(new_data)} new problems to evaluate out of {len(all_data)} total"
        )

        # Evaluate new problems
        new_results = await self.evaluate_all_problems(
            new_data, agent, save_path=past_data_path, max_concurrent_tasks=max_concurrent_tasks
        )
        
        # Merge results
        all_results = list(past_results.values()) + new_results
        
        # Save final results
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(all_results, columns)
        
        logger.info(f"{self.name} dataset average score: {average_score:.5f}")
        logger.info(f"Total cost: {total_cost:.5f}")
        logger.info(f"Average cost: {average_cost:.5f}")
        return average_score, average_cost, total_cost
