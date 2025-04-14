"""
@author: Mbaka JohnPaul
"""

from pydantic import BaseModel

class PAM(BaseModel):
    rate: float
    kw:  float
    # max:  float
    # min:  float