from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_core.documents import Document as LangchainDocument
import yaml

def get_chunker_config():
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config["chunking"]

def chunk_documents(
    docs: List[LangchainDocument],
    tokenizer_name: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None,
) -> List[LangchainDocument]:
    config = get_chunker_config()
    chunk_size = chunk_size or config["chunk_size"]
    overlap_ratio = config["chunk_overlap_ratio"]
    chunk_overlap = chunk_overlap or int(chunk_size * overlap_ratio)
    separators = separators or config["separators"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=separators,
    )

    chunked = []
    for doc in docs:
        # If the doc already represents a table row, keep it as one chunk
        if doc.metadata.get("row_index") is not None:
            chunked.append(doc)
        else:
            chunked.extend(text_splitter.split_documents([doc]))
    return chunked
