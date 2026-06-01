#from __future__ import annotations
from pydantic import BaseModel
from transformers import pipeline
import torch
from chain.runable import Runable
from typing import Any
from schemas import AskResponse

class LLMRunnerInput(BaseModel):
    full_prompt: str
    original_question: str

class LLMRunnerOutput(BaseModel):
    raw_text: str
    original_question: str

class PromptBuilderInput(BaseModel):
    question: str
    context_stats: dict

class DataSelector(Runable[PromptBuilderInput, LLMRunnerInput]):
    name: str = "data_selector"
    
    def invoke(self, data: PromptBuilderInput) -> LLMRunnerInput:
        head_data = data.context_stats.get('head', [])
        system = "You are a data filter. Pick the most relevant data rows for the question."
        prompt = f"{system}\n\nData: {head_data}\n\nQuestion: {data.question}\nRelevant data:"
        
        result= LLMRunnerInput(full_prompt=prompt, original_question= data.question)
        print(f"DEBUG: DataSelector returnerar: {type(result)}")
        return result

class AnalysisStep(Runable[LLMRunnerOutput, LLMRunnerInput]):
    def invoke(self, data: LLMRunnerOutput) -> LLMRunnerInput:
        system = "You are an expert analyst. Answer the user question based on the relevant data provided."
        prompt = f"{system}\n\nRelevant Data: {data.raw_text}\n\nQuestion: {data.original_question}\nAnswer:"
        result = LLMRunnerInput(full_prompt=prompt, original_question=data.original_question)
        print(f"DEBUG: AnalysisStep returnerar: {type(result)}")
        return result


class LLMRunner(Runable[LLMRunnerInput, LLMRunnerOutput]):
    name: str = "smol_llm_runner"
    
    _generator: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self._generator = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            device="cpu",
            dtype=torch.float32
        )

    def invoke(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        
        messages = [
            {"role": "user", "content": data.full_prompt}
        ]
        
        outputs = self._generator(
            
            messages, 
            max_new_tokens=100, 
            do_sample=False,
        )
        
        generated_text = outputs[0]["generated_text"][-1]["content"]
        result = LLMRunnerOutput(raw_text=generated_text, original_question=data.original_question) 
        print(f"DEBUG: LLMRunner returnerar: {type(result)}")
        return result

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
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct"
        )
