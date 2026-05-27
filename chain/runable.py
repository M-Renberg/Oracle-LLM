from transformers import pipeline
from pydantic import BaseModel, ConfigDict, SerializeAsAny
from typing import Any, Callable, Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")
N = TypeVar("N")

class Runable(BaseModel, Generic[I, O]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str | None = None
    
    def invoke(self, data: I) -> O:
        raise NotImplementedError("Sub class error")
    
    def __or__(self, other: Any) -> "RunableSequence":
        if isinstance(other, Runable):
            return RunableSequence.model_construct(first=self, second=other)
        if callable(other):
            return RunableSequence.model_construct(
                first=self, 
                second=RunableLambda.model_construct(func=other, name=other.__name__), 
                name=other.__name__
            )
        return NotImplemented
    
    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunableSequence.model_construct(
                first=RunableLambda.model_construct(func=other), 
                second=self, 
                name=other.__name__
            )
        return NotImplemented

class RunableLambda(Runable[I, O]):
    func: Callable[[I], O]
    
    def invoke(self, data: I) -> O:
        return self.func(data)
    
class RunableSequence(Runable[I, O], Generic[I, N, O]):
    first: SerializeAsAny[Runable[I, N]]
    second: SerializeAsAny[Runable[N, O]]
    
    def invoke(self, data: I) -> O:
        return self.second.invoke(self.first.invoke(data))