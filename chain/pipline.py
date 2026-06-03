from chain.llm import LLMRunner, ResponseParser, PromptBuilderInput, AnalysisStep, DataSelector, DirectPromptBuilder
from schemas import AskResponse

#prompt_builder = PromptBuilder()
data_selector = DataSelector()
llm_runner = LLMRunner()
analysis_step = AnalysisStep()
response_parser = ResponseParser()
direct_prompt_builder = DirectPromptBuilder()

local_chain = (
    data_selector | llm_runner |
    analysis_step | llm_runner |
    response_parser
)

claude_chain = direct_prompt_builder | llm_runner | response_parser

def run_oracle(question: str, stats: dict, model: str = "smollm2") -> AskResponse:
    input_data = PromptBuilderInput(question=question, context_stats=stats, model_key=model)
    if model == "claude":
        return claude_chain.invoke(input_data)
    return local_chain.invoke(input_data)
