from chain.llm import PromptBuilder, LLMRunner, ResponseParser, PromptBuilderInput
from schemas import AskResponse

prompt_builder = PromptBuilder()
llm_runner = LLMRunner()
response_parser = ResponseParser()

oraklet = prompt_builder | llm_runner | response_parser

def run_oracle(question: str, stats: dict) -> AskResponse:
    input_data = PromptBuilderInput(question=question, context_stats=stats)
    return oraklet.invoke(input_data)