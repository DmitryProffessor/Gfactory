from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    target: str
    kpi_name: str
    kpi_baseline: Optional[float] = None
    kpi_target: Optional[float] = None
    constraints: Dict[str, Any] = {}
    top_k_hypotheses: int = 5

class Hypothesis(BaseModel):
    rank: int
    statement: str
    justification: str
    mechanism: str
    novelty: str
    risk: str
    expected_kpi_impact: str
    experimental_roadmap: Optional[str] = None

class HypothesisResponse(BaseModel):
    hypotheses: List[Hypothesis]
