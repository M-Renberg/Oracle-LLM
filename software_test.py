# test_pipeline.py
import pytest
from pydantic import BaseModel
from chain.llm import (
    DataSelector, 
    AnalysisStep, 
    ResponseParser, 
    PromptBuilderInput, 
    LLMRunnerInput, 
    LLMRunnerOutput
)
from chain.runable import Runable


class MockLLMRunner(Runable[LLMRunnerInput, LLMRunnerOutput]):
    """En fejk-modell som returnerar ett förutbestämt svar för att testa kedjan."""
    mock_response: str = "Detta är ett mockat svar."
    
    def invoke(self, data: LLMRunnerInput) -> LLMRunnerOutput:
        return LLMRunnerOutput(
            raw_text=self.mock_response, 
            original_question=data.original_question
        )

# Tests

def test_data_selector_builds_correct_prompt():
    selector = DataSelector()
    stats = {"head": [{"Rank": 1, "Name": "Wii Sports", "Year": 2006.0}]}
    
    input_data = PromptBuilderInput(
        question="Which game is rank 1?", 
        context_stats=stats
    )
    
    result = selector.invoke(input_data)
    
    assert isinstance(result, LLMRunnerInput)
    assert "Wii Sports" in result.full_prompt
    assert result.original_question == "Which game is rank 1?"

def test_analysis_step_preserves_question():
    analyzer = AnalysisStep()
    input_data = LLMRunnerOutput(
        raw_text="The relevant row is Rank 1, Wii Sports.",
        original_question="What is the best selling game?"
    )
    
    result = analyzer.invoke(input_data)
    
    assert isinstance(result, LLMRunnerInput)
    assert "The relevant row is Rank 1" in result.full_prompt
    assert result.original_question == "What is the best selling game?"

def test_response_parser_cleans_output():
    parser = ResponseParser()
    
    dirty_response = "Answer: The game is Wii Sports. Data: {Rank: 1}"
    input_data = LLMRunnerOutput(
        raw_text=dirty_response,
        original_question="What game is this?"
    )
    
    result = parser.invoke(input_data)
    
    assert result.answer == "The game is Wii Sports."
    assert result.question == "What game is this?"
    assert result.model == "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def test_full_pipeline_with_mock():
    selector = DataSelector()
    mock_runner = MockLLMRunner(mock_response="Mockat analyserat svar.")
    analyzer = AnalysisStep()
    parser = ResponseParser()
    
    test_chain = selector | mock_runner | analyzer | mock_runner | parser
    
    stats = {"head": [{"Year": 2006.0}]}
    input_data = PromptBuilderInput(
        question="Testfråga?", 
        context_stats=stats
    )
    
    result = test_chain.invoke(input_data)
    
    assert result.question == "Testfråga?"
    assert result.answer == "Mockat analyserat svar."