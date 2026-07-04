import argparse
import os
from src.loader import MultiSourceLoader
from src.chunker import chunk_documents
from src.indexer import build_index, load_embedding_model
from byaldi import RAGMultiModalModel
import os

def build():
    # Загружаем мультимодальную модель ColPali
    model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    
    # Папка с PDF
    input_path = "data/raw"
    
    # Создаем индекс (модель превратит страницы PDF в векторы-картинки)
    model.index(
        input_path=input_path,
        index_name="metallurgy_index",
        store_collection_with_index=True,
        overwrite=True
    )
    print("Индексация завершена! Схема теперь в базе.")

if __name__ == "__main__":
    build()
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Root directory of knowledge base")
    parser.add_argument("--index_path", required=True, help="Path to save the FAISS index")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    loader = MultiSourceLoader(args.data_dir)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} raw documents")

    embedding_model = load_embedding_model()
    tokenizer_name = config["embedding"]["model_name"]
    chunked = chunk_documents(raw_docs, tokenizer_name)
    print(f"Chunked into {len(chunked)} documents")

    index = build_index(chunked, embedding_model)
    index.save_local(args.index_path)
    print(f"Index saved to {args.index_path}")

if __name__ == "__main__":
    main()
