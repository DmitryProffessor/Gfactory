# Gfactory
Инструмент для генерации гипотез на основе базы знаний

Эта система преобразует целевое свойство или технологическую проблему в ранжированный список проверяемых, научно обоснованных гипотез, извлекая соответствующие знания из разнородных источников (научные статьи, патенты, внутренние отчеты, экспериментальные данные, справочные таблицы). Он основан на архитектуре RAG и расширяет ее за счет поиска по нескольким запросам, повторного ранжирования по Кольберу и структурированного запроса для генерации гипотез.


## Features

- **Multi‑source ingestion** – Загружайте PDF–файлы, документы Word, таблицы CSV / Excel и обычный текст с богатыми метаданными.
- **Token‑aware chunking** –Использует "RecursiveCharacterTextSplitter" из токенизаторов Hugging Face для соблюдения ограничений модели.
- **Domain‑optimized embeddings** – "BAAI/bge–base-en-v1.5" для технических/научных текстов.
- **Multi‑faceted retrieval** – Отдельные подзапросы для целей/ключевых показателей эффективности, ограничений и тегов материалов/процессов – объединены и дедуплицированы.
- **ColBERT reranking** – Позволяет отобрать наиболее релевантные документы перед созданием.
- **Structured hypothesis output** – Каждая гипотеза включает обоснование, механизм, новизну, риск, ожидаемое влияние на ключевые показатели эффективности и необязательный экспериментальный план действий.
- **Modular & configurable** –  Все компоненты (модели, разбивка на блоки, поиск) настраиваются с помощью YAML.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📂 **Multi‑source Ingestion** | Loads PDFs, Word, CSV, Excel, and text files with rich metadata (source type, date, material system, property/process tags). |
| ✂️ **Token‑aware Chunking** | Uses `RecursiveCharacterTextSplitter` with Hugging Face tokenizers to respect model `max_seq_length`. Keeps tabular rows intact. |
| 🧠 **Domain Embeddings** | Leverages `BAAI/bge-base-en-v1.5` for technical/scientific text (can be swapped). |
| 🔍 **Multi‑query Retrieval** | Generates separate sub‑queries for the target/KPI, each constraint, and extracted tags – merges and deduplicates results. |
| 🎯 **ColBERT Reranking** | Refines the candidate set using ColBERTv2 to keep only the most relevant documents. |
| 📝 **Structured Hypothesis Output** | Each hypothesis includes *statement*, *justification*, *mechanism*, *novelty*, *risk*, *expected KPI impact*, and *optional experimental roadmap*. |
| ⚙️ **Fully Configurable** | All models, chunking parameters, retrieval top‑k, and generation settings are controlled via `config/default.yaml`. |

### 📂 Project Structure

Gfactory/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── schemas.py          # Pydantic models
│   ├── loader.py           # Multi‑source loader
│   ├── chunker.py          # Token‑aware chunking
│   ├── indexer.py          # FAISS index building
│   ├── retriever.py        # Multi‑query retrieval
│   ├── reranker.py         # ColBERT reranking
│   ├── generator.py        # LLM prompt & parsing
│   └── pipeline.py         # End‑to‑end HypothesisGenerator
├── scripts/
│   ├── build_index.py
│   └── run_query.py
├── notebooks/
│   └── demo.ipynb
└── data/                   # (user‑provided)

## Installation

```bash
git clone https://github.com/DmitryProffessor/Gfactory.git
cd gfactory
pip install -e .
