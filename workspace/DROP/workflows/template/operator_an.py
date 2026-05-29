from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class SolveOp(BaseModel):
    """Output schema for the DROP Solve operator."""
    thought: str = Field(default="", description="Step-by-step reasoning process")
    answer: str = Field(default="", description="The final answer (number or short phrase)")


class FormatOp(BaseModel):
    """Output schema for the Format operator."""
    solution: str = Field(default="", description="The formatted final answer")