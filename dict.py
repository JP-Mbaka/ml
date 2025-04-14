"""
@author: Mbaka JohnPaul
"""

from pydantic import BaseModel

class Predict(BaseModel):
    performance: float