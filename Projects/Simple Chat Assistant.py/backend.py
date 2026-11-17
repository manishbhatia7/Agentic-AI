from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()
llm=ChatOpenAI(model="gpt-4.1-mini",temperature=0)
class TopicExplaination(BaseModel):
    topic:str=Field(description="Main topic which is the subject we are discussing")
    summary:str=Field(description="Summary of the subject we are discussing")
    key_points:list[str]=Field(description="Main highlights in bullet for the topic")
    examples:list[str]=Field(description="Examples of the topic")
    
structured_prompt=ChatPromptTemplate.from_messages([("system","Explain topics in simple language suitable for beginners.\n" "{format_instructions}"),("human","Explain the following topic\n : {topic}")])
quiz_prompt=ChatPromptTemplate.from_messages([("system","You are a teacher who creates simple quizzes for beginners."),("human","Generate 3 multiple-choice questions about {topic}. Include 4 options and the correct answer for each question.")])
question_prompt=ChatPromptTemplate.from_messages([("system","You are a SME who answers follow up questions from user"
"Use previous conversation to understand follow up questions"),MessagesPlaceholder(variable_name="history"),("human","{question}")])
question_chain=question_prompt | llm | StrOutputParser()
str_parser=StrOutputParser()
store={}
def get_message_history(session_id:str):
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]
conversion_chain=RunnableWithMessageHistory(
    question_chain,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history"
)
    
parser=PydanticOutputParser(pydantic_object=TopicExplaination)
clean_input=RunnableLambda(
    lambda x:{
        "topic":" ".join(x['topic'].split()).title(),
        "format_instructions":x['format_instructions']
    }
)
explanation_chain=clean_input | structured_prompt | llm | parser
quiz_chain=quiz_prompt | llm | str_parser
learning_chain=RunnableParallel(explanation=explanation_chain,quiz=quiz_chain)



   