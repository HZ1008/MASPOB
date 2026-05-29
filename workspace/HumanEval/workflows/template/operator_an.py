from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class ReflectionTestOp(BaseModel):
    reflection_and_solution: str = Field(
        default="", description="Corrective solution for code execution errors or test case failures"
    )