SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

FORMAT_PROMPT = """
Given the question and the best answer, format the final answer to be concise,
accurate, and directly addressing the question.
Ensure the answer is a clear, brief statement without additional explanation or reasoning.
If the answer is a name, profession, or short phrase, provide only that information without forming a complete sentence.

For example:
- If the answer is a person's name, just provide the name.
- If the answer is a profession, state only the profession.
- If the answer is a short phrase, give only that phrase.
Do not include any prefixes like "The answer is" or "The profession is".
Just provide the answer itself.
"""

FORMAT_ANSWER_PROMPT = """
Given the question and the best answer below, extract and format the final answer.
The answer should be concise - typically just a name, number, phrase, or short statement.
Do not include explanations, reasoning, or prefixes like "The answer is".
Put your formatted answer in the "answer" field.

{input}
"""