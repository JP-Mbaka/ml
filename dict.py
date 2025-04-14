"""
@author: Mbaka JohnPaul
"""

from pydantic import BaseModel

class Predict(BaseModel):
    department: int
    year: int
    gender: int
    difficultyEng: int
    difficultyComp: int
    difficultyStat: int
    qualityEng: int
    qualityComp: int
    qualityStat: int