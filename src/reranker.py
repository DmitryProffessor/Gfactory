from ragatouille import RAGPretrainedModel
from typing import List, Optional
from langchain_core.documents import Document as LangchainDocument
import yaml

def get_reranker_config():
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config["reranker"]

class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        config = get_reranker_config()
        self.model_name = model_name or config["model_name"]
        self.model = RAGPretrainedModel.from_pretrained(self.model_name)

    def rerank(self, query: str, documents: List[LangchainDocument], top_k: int) -> List[LangchainDocument]:
        doc_texts = [doc.page_content for doc in documents]
        results = self.model.rerank(query, doc_texts, k=top_k)
        # Map back to original documents
        reranked = []
        for item in results:
            # Find the doc with matching content
            for doc in documents:
                if doc.page_content == item["content"]:
                    reranked.append(doc)
                    break
        return reranked
