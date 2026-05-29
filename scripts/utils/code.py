import re
import json
from enum import Enum
from typing import Any, List, Tuple, Union


class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"
    LIVE_CODE_BENCH = "LiveCodeBench"


def extract_test_cases_from_jsonl(entry_point: str, dataset: Union[CodeDataset, str] = CodeDataset.HUMAN_EVAL):
    dataset_value = dataset.value if isinstance(dataset, CodeDataset) else dataset

    file_map = {
        CodeDataset.HUMAN_EVAL.value: "data/datasets/humaneval_public_test.jsonl",
        CodeDataset.MBPP.value: "data/datasets/mbpp_public_test.jsonl",
        CodeDataset.LIVE_CODE_BENCH.value: "data/datasets/livecodebench_public_test.jsonl",
    }
    hardcoded_cases_map = {
        CodeDataset.HUMAN_EVAL.value: {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        },
        CodeDataset.MBPP.value: {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        },
        CodeDataset.LIVE_CODE_BENCH.value: {},
    }

    file_path = file_map.get(dataset_value)
    hardcoded_cases = hardcoded_cases_map.get(dataset_value, {})

    # Prefer hardcoded test cases if available
    if entry_point in hardcoded_cases:
        return hardcoded_cases[entry_point]

    key = "question_id" if dataset_value == CodeDataset.LIVE_CODE_BENCH.value else "entry_point"
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get(key) == entry_point:
                return data.get("test")

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function
