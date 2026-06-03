#from __future__ import annotations
from pydantic import BaseModel
from transformers import pipeline
import torch
from chain.runable import Runable
from typing import Any
from schemas import AskResponse
import anthropic
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "claude":  "claude-haiku-4-5-20251001",
}

_model_cache: dict = {}

def get_or_load_model(model_key: str):
    if model_key == "claude":
        return None
    if model_key not in _model_cache:
        if model_key not in MODELS:
            raise ValueError(f"Okänd modell '{model_key}'. Tillgängliga: {list(MODELS.keys())}")
        _model_cache[model_key] = pipeline(
            "text-generation",
            model=MODELS[model_key],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return _model_cache[model_key]



class LLMRunnerInput(BaseModel):
    full_prompt: str
    original_question: str
    model_key: str = "smollm2"

class LLMRunnerOutput(BaseModel):
    raw_text: str
    original_question: str
    model_key: str = "smollm2"

class PromptBuilderInput(BaseModel):
    question: str
    context_stats: dict
    model_key: str = "smollm2"

class DataSelector(Runable[PromptBuilderInput, LLMRunnerInput]):
    name: str = "data_selector"
    
    def invoke(self, data: PromptBuilderInput) -> LLMRunnerInput:
        head_data = data.context_stats.get('head', [])
        system = "You are a data filter. Pick the most relevant data rows for the question."
        prompt = f"{system}\n\nData: {head_data}\n\nQuestion: {data.question}\nRelevant data:"
        
        result= LLMRunnerInput(full_prompt=prompt, original_question= data.question, model_key=data.model_key)
        print(f"DEBUG: DataSelector returnerar: {type(result)}")
        return result

class AnalysisStep(Runable[LLMRunnerOutput, LLMRunnerInput]):
    def invoke(self, data: LLMRunnerOutput) -> LLMRunnerInput:
        system = "You are an expert analyst. Answer the user question based on the relevant data provided."
        prompt = f"{system}\n\nRelevant Data: {data.raw_text}\n\nQuestion: {data.original_question}\nAnswer:"
        result = LLMRunnerInput(full_prompt=prompt, original_question=data.original_question, model_key=data.model_key)
        print(f"DEBUG: AnalysisStep returnerar: {type(result)}")
        return result
    
class DirectPromptBuilder(Runable[PromptBuilderInput, LLMRunnerInput]):
    name: str = "direct_prompt_builder"

    def invoke(self, data: PromptBuilderInput) -> LLMRunnerInput:
        head_data = data.context_stats.get('head', [])
        table_str = data.context_stats.get('head', '')
        
        prompt = f"You are a data analyst. Answer the question based on the data below.\n\nData:\n{table_str}\n\nQuestion: {data.question}\nAnswer:"
        return LLMRunnerInput(
            full_prompt=prompt,
            original_question=data.question,
            model_key=data.model_key,
        )


class LLMRunner(Runable[LLMRunnerInput, LLMRunnerOutput]):
    name: str = "smol_llm_runner"
 
    def invoke(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        if data.model_key == "claude":
            return self._invoke_claude(data)
        return self._invoke_local(data)
 
    def _invoke_local(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        generator = get_or_load_model(data.model_key)
        messages = [{"role": "user", "content": data.full_prompt}]
        outputs = generator(messages, max_new_tokens=100, do_sample=False)
        generated_text = outputs[0]["generated_text"][-1]["content"]
        return LLMRunnerOutput(
            raw_text=generated_text,
            original_question=data.original_question,
            model_key=data.model_key,
        )
 
    def _invoke_claude(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=MODELS["claude"],
            max_tokens=256,
            messages=[{"role": "user", "content": data.full_prompt}]
        )
        generated_text = message.content[0].text
        return LLMRunnerOutput(
            raw_text=generated_text,
            original_question=data.original_question,
            model_key=data.model_key,
        )

class ResponseParser(Runable[LLMRunnerOutput, AskResponse]):
    name: str = "response_parser"
    
    def invoke(self, data: LLMRunnerOutput) -> AskResponse:
        raw = data.raw_text
        if "Answer:" in raw:
            answer = raw.split("Answer:")[-1]
        elif "Svar:" in raw:
            answer = raw.split("Svar:")[-1]
        else:
            answer = raw
            
        answer = answer.split("Data:")[0].split("Question:")[0].strip()
            
        return AskResponse(
            question=data.original_question,
            answer=answer,
            model=MODELS.get(data.model_key, data.model_key)
        )
