from pydantic import BaseModel
from transformers import pipeline
import torch
from chain.runable import Runable
from typing import Any
from schemas import AskResponse

class LLMRunnerInput(BaseModel):
    full_prompt: str

class LLMRunnerOutput(BaseModel):
    raw_text: str
    original_question: str

class LLMRunner(Runable[LLMRunnerInput, LLMRunnerOutput]):
    name: str = "smol_llm_runner"
    
    _generator: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self._generator = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            device="cpu",
            dtype=torch.float32
        )

    def invoke(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        outputs = self._generator(
            data.full_prompt, 
            max_new_tokens=25, 
            do_sample=False,
        )
        
        generated_text = outputs[0]["generated_text"]
        
        return LLMRunnerOutput(raw_text=generated_text, original_question="fetch question")


class PromptBuilderInput(BaseModel):
    question: str
    context_stats: dict

class PromptBuilder(Runable[PromptBuilderInput, LLMRunnerInput]):
    name: str = "prompt_builder"
    
    def invoke(self, data: PromptBuilderInput) -> LLMRunnerInput:
        #csv_reducer = {k: {sk: v for sk, v in sv.items() if sk in ['max', 'mean']} for k, sv in data.context_stats.items()}
        head_data = data.context_stats.get('head', [])
        system = "You're an expert data analysis. You only give short answers."
        table_str = str(head_data)
        #stats = f"Statistik: {data.context_stats}"
        #prompt = f"{system}\n\nData: {csv_reducer}\n\nFråga: {data.question}\nSvar:"        
        prompt = f"{system}\n\ndata: {table_str}\n\nquestion: {data.question}\nanswer:"
        return LLMRunnerInput(full_prompt=prompt)
    

class ResponseParser(Runable[LLMRunnerOutput, AskResponse]):
    name: str = "response_parser"
    
    def invoke(self, data: LLMRunnerOutput) -> AskResponse:
        raw = data.raw_text
        if "Svar:" in raw:
            answer = raw.split("Svar:")[-1].strip()
        else:
            answer = raw.strip()
            
        answer = answer.split("Data:")[0].split("Question:")[0].strip()
            
        return AskResponse(
            question="[Frågan hämtas från tidigare steg]",
            answer=answer,
            model="HuggingFaceTB/SmolLM2-135M-Instruct"
        )
