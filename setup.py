from setuptools import setup, find_packages

setup(
    name="gfactory",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "langchain-text-splitters",
        "sentence-transformers",
        "faiss-gpu",
        "pypdf",
        "pdfplumber",
        "python-docx",
        "pandas",
        "openpyxl",
        "pyyaml",
        "python-dotenv",
        "ragatouille",
        "pydantic",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "build-index=scripts.build_index:main",
            "run-query=scripts.run_query:main",
        ],
    },
    author="DmitryProffessor",
    description="Gfactory",
    python_requires=">=3.10",
)
