from transformers import pipeline
from pydantic import BaseModel, ConfigDict, SerializeAsAny
from typing import Any, Callable, Generic, TypeVar


I = TypeVar("I")
O = TypeVar("O")
N = TypeVar("N")

class Runable(BaseModel, Generic[I,O]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str | None = None
    
    def invoke(self, data:I) -> O:
        raise NotImplementedError("Sub class error")
    
    def __or__(self, other:Any) -> RunableSequence:
        if isinstance(other, Runable):
            return RunableSequence.model_construct(first=self, second=other)
        if callable(other):
            return RunableSequence.model_construct(first=self, second=RunableLambda.model_construct(func=other, name=other.__name__), name=other.__name__)
        return NotImplemented
    
    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunableSequence.model_construct(first=RunableLambda.model_construct(func=other), second=self, name=other.__name__)
        return NotImplemented
    

class RunableLambda(Runable[I, O]):
    func: Callable[[I], O]
    
    def invoke(self, data: I) -> O:
        return self.func(data)
    
class RunableSequence(Runable[I,O], Generic[I,N,O]):
    first: SerializeAsAny[Runable[I,N]]
    second: SerializeAsAny[Runable[N,O]]
    
    def invoke(self, data: I) -> O:
        return self.second.invoke(self.first.invoke(data))
    
############################DTO#################################
#Input
class TicketInput(BaseModel):
    customer_id: int
    message: str
#Output
class ProcessedTicket(BaseModel):
    customer_id: int
    sentiment: str
    urgency: str
    summary: str
################################################################

class SentimentAnalyser(Runable[TicketInput, dict]):
    name: str = "sentiment_analyser"
    model_version: str = "2.1-stable"
        
    def invoke(self, ticket: TicketInput) -> dict:
        
        msg_lower = ticket.message.lower()
        
        sentiment = "negative" if "broken" in msg_lower or "Angry" in msg_lower else "neutral"
        urgency = "high" if "broken" in msg_lower or "urgent" in msg_lower else "low"
        
        return{
            "customer_id": ticket.customer_id,
            "sentiment": sentiment,
            "urgency": urgency,
            "summary": ticket.message[:40] + "..."
        }

class TicketParser(Runable[dict, ProcessedTicket]):
    name: str = "ticket_parser"
    
    def invoke(self, raw_dict: dict) -> ProcessedTicket:
        return ProcessedTicket(**raw_dict)


def route_ticket(ticket: ProcessedTicket) -> dict:
        destination = "enginnering_team" if "high" in ticket.urgency else "general"
        return{
            "status": "routed",
            "assigned_to": destination,
            "ticket_details": ticket.model_dump()
        }

Ticket_pipeline = SentimentAnalyser() | TicketParser() | route_ticket

incoming_ticket = TicketInput(
    customer_id=1337,
    message="the payment portal is broken! urgent! fix asap"
)

#final_output = Ticket_pipeline.invoke(incoming_ticket)