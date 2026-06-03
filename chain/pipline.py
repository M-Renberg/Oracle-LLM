from chain.llm import LLMRunner, ResponseParser, PromptBuilderInput, AnalysisStep, DataSelector
from schemas import AskResponse

#prompt_builder = PromptBuilder()
data_selector = DataSelector()
llm_runner = LLMRunner()
analysis_step = AnalysisStep()
response_parser = ResponseParser()

oraklet = (
    data_selector | llm_runner | 
    analysis_step | llm_runner | 
    response_parser
)

def run_oracle(question: str, stats: dict, model: str = "smollm2") -> AskResponse:
    input_data = PromptBuilderInput(question=question, context_stats=stats, model_key=model)
    return oraklet.invoke(input_data)
