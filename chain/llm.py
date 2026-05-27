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

class LLMRunner(Runable[LLMRunnerInput, LLMRunnerOutput]):
    name: str = "smol_llm_runner"
    
    _generator: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self._generator = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            device_map="auto",
            torch_dtype=torch.float32
        )

    def invoke(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        outputs = self._generator(
            data.full_prompt, 
            max_new_tokens=100, 
            do_sample=False
        )
        
        generated_text = outputs[0]["generated_text"]
        
        return LLMRunnerOutput(raw_text=generated_text)


class PromptBuilderInput(BaseModel):
    question: str
    context_stats: dict

class PromptBuilder(Runable[PromptBuilderInput, LLMRunnerInput]):
    name: str = "prompt_builder"
    
    def invoke(self, data: PromptBuilderInput) -> LLMRunnerInput:
        system = "Du är en analytiker. Svara kortfattat på frågan baserat på statistiken"
        stats = f"Statistik: {data.context_stats}"
        prompt = f"{system}\n\n{stats}\n\nFråga: {data.question}\nSvar:"
        
        return LLMRunnerInput(full_prompt=prompt)
