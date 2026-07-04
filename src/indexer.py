from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from typing import List
import yaml
import torch

def get_embedding_config():
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config["embedding"]

def build_index(docs: List[LangchainDocument], embedding_model) -> FAISS:
    return FAISS.from_documents(docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)

def load_embedding_model():
    config = get_embedding_config()
    return HuggingFaceEmbeddings(
        model_name=config["model_name"],
        model_kwargs={"device": config.get("device", "cuda" if torch.cuda.is_available() else "cpu")},
        encode_kwargs={"normalize_embeddings": config.get("normalize_embeddings", True)},
    )
