from transformers import Pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import re
from typing import List, Optional
from src.schemas import QueryRequest, HypothesisResponse, Hypothesis
from langchain_core.documents import Document as LangchainDocument
import yaml

def get_reader_config():
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config["reader"]

class HypothesisGeneratorLLM:
    def __init__(self):
        self.config = get_reader_config()
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        model_name = self.config["model_name"]
        quant_cfg = self.config["quantization"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = Pipeline(
            model=model,
            tokenizer=self.tokenizer,
            task="text-generation",
            **self.config["generation"],
        )

    def generate(self, request: QueryRequest, context_docs: List[LangchainDocument]) -> HypothesisResponse:
        context = "\n".join([f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(context_docs)])
        constraints_str = ", ".join([f"{k}: {v}" for k, v in request.constraints.items() if v])

        system_prompt = """You are a materials science and process engineering expert. Based on the provided context (scientific papers, patents, internal reports, experimental data), generate a ranked list of testable hypotheses to achieve the user's target. For each hypothesis, provide:
- statement: concise hypothesis
- justification: evidence from context
- mechanism: proposed physical/chemical mechanism
- novelty: how novel is this approach?
- risk: potential risks or failure modes
- expected_kpi_impact: quantitative improvement expected
- experimental_roadmap: suggested next experiments (optional)

Respond with a valid JSON object containing a list of hypotheses. Do not include any other text."""

        user_prompt = f"""Target: {request.target}
KPI: {request.kpi_name} (baseline: {request.kpi_baseline}, target: {request.kpi_target})
Constraints: {constraints_str}
Context:
{context}
Generate {request.top_k_hypotheses} hypotheses."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.pipeline(final_prompt)[0]["generated_text"]

        # Parse JSON
        try:
            data = json.loads(response)
            if "hypotheses" in data:
                return HypothesisResponse(**data)
            else:
                return HypothesisResponse(hypotheses=[Hypothesis(**h) for h in data])
        except json.JSONDecodeError:
            # Try to extract JSON block
            match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return HypothesisResponse(**data)
                except:
                    pass
            # Fallback: create a single hypothesis with raw answer
            return HypothesisResponse(
                hypotheses=[Hypothesis(
                    rank=1,
                    statement="Error parsing LLM output",
                    justification=response,
                    mechanism="",
                    novelty="",
                    risk="",
                    expected_kpi_impact="",
                )]
            )
