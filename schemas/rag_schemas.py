from pydantic import BaseModel, Field, HttpUrl
from typing import List

class RunRequest(BaseModel):
    documents: HttpUrl = Field(..., description="Fully qualified URL to the target PDF document.")
    questions: List[str] = Field(..., description="List of questions to extract answers for.")

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="Ordered list of answers corresponding to the questions input.")