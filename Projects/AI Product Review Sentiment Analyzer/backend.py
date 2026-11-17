from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
 
load_dotenv()
 
# FIX: 100 tokens truncates the JSON output; raise the limit
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, max_completion_tokens=800)
 
 
# ---------------------------------------------------------------------------
# Pydantic schema + parser
# ---------------------------------------------------------------------------
class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "mixed"]
    summary: str
    key_themes: list[str]
    pros: list[str]
    cons: list[str]
    confidence: float = Field(ge=0, le=1)
 
 
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = parser.get_format_instructions()
 
# ---------------------------------------------------------------------------
# Analysis prompt
# ---------------------------------------------------------------------------
analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert product review analyst. Analyse the latest product review.
Identify sentiment, summary, themes, pros, cons and confidence.
{format_instructions}""",
            # FIX: removed the line saying sentiment may be "Neutral"/capitalized,
            # which contradicted the schema (positive | negative | mixed).
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Analyze this review: {input}"),
    ]
).partial(format_instructions=format_instructions)  # FIX: supply the variable
 
analysis_chain = analysis_prompt | model | parser
 
# ---------------------------------------------------------------------------
# Feedback prompts / chains
# ---------------------------------------------------------------------------
positive_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly customer relationship manager. Appreciate the customer's experience."),
        ("human", "Review Summary: {summary}\nPros: {pros}\nGenerate Positive feedback"),
    ]
)
negative_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an empathetic customer relationship manager. Acknowledge the problem with the customer's experience and suggest measures."),
        ("human", "Review Summary: {summary}\nCons: {cons}\nGenerate Negative feedback"),
    ]
)
mixed_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a customer experience manager. Generate balanced feedback."),
        ("human", "Summary: {summary}\nPros: {pros}\nCons: {cons}\nGenerate Balanced feedback"),
    ]
)
 
positive_chain = positive_prompt | model | StrOutputParser()
negative_chain = negative_prompt | model | StrOutputParser()
mixed_chain = mixed_prompt | model | StrOutputParser()
 
# FIX: the opening parenthesis must be on the same line as RunnableBranch,
# otherwise you just reference the class and never create an instance.
feedback_branch = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_chain),
    (lambda x: x["sentiment"] == "negative", negative_chain),
    mixed_chain,  # default (also covers "mixed")
)
 
# ---------------------------------------------------------------------------
# Review workflow: analyze -> dict -> add feedback
# ---------------------------------------------------------------------------
review_workflow = (
    analysis_chain
    | RunnableLambda(lambda x: x.model_dump())
    | RunnablePassthrough.assign(feedback=feedback_branch)
)
 
# Format the analysis response
analysis_response = review_workflow | RunnableLambda(
    lambda x: {
        "answer": (
            "Sentiment: {}\n"
            "Summary: {}\n"
            "Pros: {}\n"
            "Cons: {}\n"
            "Feedback: {}"
        ).format(x["sentiment"], x["summary"], x["pros"], x["cons"], x["feedback"]),
        "analysis": x,
    }
)
 
# ---------------------------------------------------------------------------
# Conversational Q&A about previous reviews
# ---------------------------------------------------------------------------
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an intelligent product review assistant.
Answer questions using the previous conversation history.
Never invent information.""",  # FIX: typo "intent" -> "invent"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
 
question_chain = (
    question_prompt
    | model
    | StrOutputParser()
    | RunnableLambda(lambda x: {"answer": x, "analysis": None})
)
 
# ---------------------------------------------------------------------------
# Main branch: analyze vs. question
# ---------------------------------------------------------------------------
main_chain = RunnableBranch(
    (lambda x: x["task"] == "analyze", analysis_response),
    question_chain,
)
 
# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
store = {}
 
 
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
 
 
# FIX: tell the wrapper where the input, the history, and the output live
conversation_chain = RunnableWithMessageHistory(
    main_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)