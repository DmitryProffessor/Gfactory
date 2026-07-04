from src.loader import MultiSourceLoader
from src.chunker import chunk_documents
from src.indexer import build_index, load_embedding_model
from src.retriever import retrieve_and_merge
from src.reranker import Reranker
from src.generator import HypothesisGeneratorLLM
from src.schemas import QueryRequest, HypothesisResponse
from langchain_community.vectorstores import FAISS
import os
import yaml

class HypothesisGenerator:
    def __init__(self, index_path: str = None, knowledge_root: str = None):
        self.embedding_model = load_embedding_model()
        self.reranker = Reranker()
        self.llm = HypothesisGeneratorLLM()

        if index_path and os.path.exists(index_path):
            self.index = FAISS.load_local(index_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            if knowledge_root is None:
                raise ValueError("Either index_path or knowledge_root must be provided.")
            loader = MultiSourceLoader(knowledge_root)
            raw_docs = loader.load()
            # Load embedding config to get tokenizer name
            with open("config/default.yaml", 'r') as f:
                config = yaml.safe_load(f)
            tokenizer_name = config["embedding"]["model_name"]
            chunked = chunk_documents(raw_docs, tokenizer_name)
            self.index = build_index(chunked, self.embedding_model)
            if index_path:
                self.index.save_local(index_path)

    def generate(self, request: QueryRequest) -> HypothesisResponse:
        # Retrieve
        candidates = retrieve_and_merge(self.index, request)
        # Rerank
        final_docs = self.reranker.rerank(request.target, candidates, top_k=request.top_k_hypotheses * 2)  # get extra, but we'll use only top_k in generation
        final_docs = final_docs[:request.top_k_hypotheses]
        # Generate
        return self.llm.generate(request, final_docs)
