import os
import json
from typing import List
from langchain_core.documents import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import pandas as pd

class MultiSourceLoader:
    def __init__(self, root_dir: str, sidecar_suffix: str = ".meta.json"):
        self.root_dir = root_dir
        self.sidecar_suffix = sidecar_suffix

    def load(self) -> List[LangchainDocument]:
        documents = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for f in filenames:
                filepath = os.path.join(dirpath, f)
                meta_path = filepath + self.sidecar_suffix
                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)

                ext = os.path.splitext(f)[1].lower()
                if ext == '.pdf':
                    loader = PyPDFLoader(filepath)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update(metadata)
                    documents.extend(docs)
                elif ext in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(filepath)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update(metadata)
                    documents.extend(docs)
                elif ext == '.csv':
                    df = pd.read_csv(filepath)
                    for idx, row in df.iterrows():
                        text = "; ".join([f"{col}: {val}" for col, val in row.items()])
                        doc = LangchainDocument(page_content=text, metadata=metadata.copy())
                        doc.metadata['row_index'] = idx
                        documents.append(doc)
                elif ext == '.xlsx':
                    df = pd.read_excel(filepath)
                    for idx, row in df.iterrows():
                        text = "; ".join([f"{col}: {val}" for col, val in row.items()])
                        doc = LangchainDocument(page_content=text, metadata=metadata.copy())
                        doc.metadata['row_index'] = idx
                        documents.append(doc)
                else:
                    # Fallback to plain text
                    loader = TextLoader(filepath)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update(metadata)
                    documents.extend(docs)
        return documents
