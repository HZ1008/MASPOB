from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class RefineAnswerOp(BaseModel):
    """Output schema for the RefineAnswer operator."""
    response: str = Field(default="", description="The refined and formatted solution with LaTeX notation")


class GenerateSolutionOp(BaseModel):
    """Output schema for the GenerateSolution operator."""
    response: str = Field(default="", description="The step-by-step solution with LaTeX notation")


class DetailedSolutionOp(BaseModel):
    """Output schema for the DetailedSolution operator."""
    response: str = Field(default="", description="The comprehensive detailed solution with explanations")
