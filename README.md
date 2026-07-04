# Gfactory
Инструмент для генерации гипотез на основе базы знаний

Эта система преобразует целевое свойство или технологическую проблему в ранжированный список проверяемых, научно обоснованных гипотез, извлекая соответствующие знания из разнородных источников (научные статьи, патенты, внутренние отчеты, экспериментальные данные, справочные таблицы). Он основан на архитектуре RAG и расширяет ее за счет поиска по нескольким запросам, повторного ранжирования по Кольберу и структурированного запроса для генерации гипотез.

# Технологический стек Gfactory

**Gfactory** построен на стеке современных технологий, каждая из которых решает свою задачу в пайплайне генерации гипотез. Ниже представлены ключевые компоненты.

---

## 🧠 Базовые фреймворки машинного обучения

### [PyTorch](https://pytorch.org/)
Основа для всех моделей глубокого обучения. Обеспечивает автоматическое дифференцирование и вычисления на GPU, критически важные для работы больших языковых моделей (LLM) и моделей эмбеддингов.

### [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
Центральная библиотека для работы с предобученными моделями. Предоставляет унифицированный API для загрузки, использования и тонкой настройки тысяч моделей, включая:
- **Zephyr-7B** (LLM)
- **BAAI/bge-base-en-v1.5** (эмбеддинги)

В проекте она используется через `AutoTokenizer` и `AutoModelForCausalLM`.

### [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
Библиотека, упрощающая запуск PyTorch-кода на различных аппаратных конфигурациях (GPU, TPU) без изменения самого кода. Обеспечивает кроссплатформенность Gfactory.

### [Sentence-Transformers](https://www.sbert.net/)
Специализированная библиотека для создания эмбеддингов предложений, текстов и изображений. Используется для работы с моделью `BAAI/bge-base-en-v1.5` и генерации плотных векторных представлений документов.

---

## 🗄️ Индексация и поиск (Retrieval)

### [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
Библиотека от Meta для эффективного поиска схожести и кластеризации плотных векторов. Gfactory использует её для создания индекса, который позволяет за миллисекунды находить документы, семантически близкие к запросу, среди десятков тысяч чанков.

В проекте применяется точный поиск по косинусному расстоянию (`DistanceStrategy.COSINE`).

### ColBERT & RAGatouille
Связка для **реранкинга** (повторной сортировки) результатов.

- **[ColBERT](https://github.com/stanford-futuredata/ColBERT)** (Contextualized Late Interaction over BERT) — модель, которая использует механизм "позднего взаимодействия" для оценки релевантности документа запросу на уровне отдельных токенов. Это дает более высокое качество, чем простое сравнение эмбеддингов всего документа.

- **[RAGatouille](https://github.com/bclavie/RAGatouille)** — библиотека-обертка, которая делает работу с ColBERT максимально простой. Она скрывает сложности индексации и запросов, позволяя добавить мощный реранкинг в пайплайн в несколько строк кода.

---

## 🧬 Модели

### BAAI/bge-base-en-v1.5 (Embedding Model)
Модель от Пекинской академии искусственного интеллекта (BAAI), созданная специально для задач поиска (Retrieval).

- Преобразует текст в плотный 768-мерный вектор.
- Использует архитектуру BERT-base с 12 слоями.
- Выбрана за высокое качество семантического представления технических и научных текстов.

### HuggingFaceH4/zephyr-7b-beta (Reader LLM)
«Мозг» Gfactory — языковая модель, которая генерирует финальные гипотезы.

- Имеет 7 миллиардов параметров и основана на архитектуре Transformer.
- Является доработанной (fine-tuned) версией модели `Mistral-7B-v0.1`.
- Обучена на смеси публичных и синтетических данных, что делает её эффективной для следования инструкциям.

---

## ⚙️ Инфраструктура и оптимизация

### [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
Библиотека, реализующая **4-битную квантизацию** (QLoRA). Эта техника сжимает веса модели Zephyr-7B с 16/32 бит до 4 бит, что **снижает потребление видеопамяти примерно на 75%** без значительной потери качества. Это позволяет запускать 7-миллиардную модель на потребительских GPU с 8–12 ГБ памяти.

### [LangChain](https://www.langchain.com/)
Основной фреймворк для построения всего RAG-пайплайна:

- **Загрузка документов** – абстракции `DocumentLoader` для работы с PDF, Word, CSV и др.
- **Разбивка текста** – `RecursiveCharacterTextSplitter` для создания чанков с учётом токенизатора.
- **Векторные хранилища** – интеграция с FAISS для создания индекса.
- **Управление пайплайном** – связывает все этапы в единый, управляемый процесс.

---

## 🛠️ Вспомогательные библиотеки

- **[Pydantic](https://docs.pydantic.dev/)** – определение строгих схем данных (`QueryRequest`, `Hypothesis`), обеспечивающих валидацию входных и выходных данных.
- **[PyYAML](https://pyyaml.org/)** – хранение всех настроек модели, чанкинга и поиска в удобном файле `config/default.yaml`, делая систему гибкой без изменения кода.
- **[tqdm](https://github.com/tqdm/tqdm)** – отображение прогресс-баров для длительных операций, таких как загрузка данных или индексация.

---

## Итог

Эта комбинация технологий создаёт мощную, оптимизированную и гибкую систему, способную обрабатывать сложные научно-технические запросы и генерировать обоснованные гипотезы.


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

## Installation

```bash
git clone https://github.com/DmitryProffessor/Gfactory.git
cd gfactory
pip install -e .
