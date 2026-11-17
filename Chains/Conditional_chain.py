from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda
load_dotenv()
model=ChatOpenAI(model='gpt-4.1-mini')
parser=StrOutputParser()
class feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Extract the sentiment from given text')
parser2=PydanticOutputParser(pydantic_object=feedback)
prompt1=PromptTemplate(template='Classify the sentiment as positive or negative with the given \n {feedback} \n {format_instruction}',input_variables=['feedback'],partial_variables={'format_instruction':parser2.get_format_instructions()})
prompt2=PromptTemplate(template='Write an appropriate respose to this positive feedback \n {feedback}',input_variables=['feedback'])
prompt3=PromptTemplate(template='Write an appropriate respose to this negative feedback \n {feedback}',input_variables=['feedback'])

branch_chain=RunnableBranch(
   (lambda x:x.sentiment == 'positive',prompt2 | model | parser),
   (lambda x:x.sentiment == 'negative',prompt3 | model | parser),
   RunnableLambda(lambda x:'could not find sentiment')
   
)

classifier_chain=prompt1 | model | parser2
final_chain=classifier_chain | branch_chain
result=final_chain.invoke({'feedback':'This is a terrible smartphone'})
print(result)