from langchain_community.vectorstores import FAISS
from src.schemas import QueryRequest
from typing import List, Set
from langchain_core.documents import Document as LangchainDocument

def build_subqueries(request: QueryRequest) -> List[str]:
    queries = []
    # Target + KPI
    target_text = f"{request.target}, KPI: {request.kpi_name}"
    if request.kpi_baseline is not None:
        target_text += f", baseline: {request.kpi_baseline}"
    if request.kpi_target is not None:
        target_text += f", target: {request.kpi_target}"
    queries.append(target_text)

    # Constraints
    for key, val in request.constraints.items():
        if val:
            if isinstance(val, list):
                val_str = ", ".join(str(v) for v in val)
            else:
                val_str = str(val)
            queries.append(f"{key}: {val_str}")
    # Optional: add material/system tags if we can extract them
    # (could parse from target/constraints or use a tag extraction step)
    return queries

def retrieve_and_merge(index: FAISS, request: QueryRequest, top_k: int = 30) -> List[LangchainDocument]:
    all_docs = []
    seen = set()
    for q in build_subqueries(request):
        docs = index.similarity_search(q, k=top_k)
        for d in docs:
            # Use a combination of source_id and first 100 chars as a simple key
            key = (d.metadata.get("source_id", ""), d.page_content[:100])
            if key not in seen:
                seen.add(key)
                all_docs.append(d)
    # Optionally cap total candidates to avoid huge reranking cost
    if len(all_docs) > top_k * 3:
        all_docs = all_docs[:top_k * 3]
    return all_docs
