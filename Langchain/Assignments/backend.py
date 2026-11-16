from dotenv import load_dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnableWithMessageHistory,RunnableSequence,RunnablePassthrough,RunnableBranch
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()
llm=ChatOpenAI(model='gpt-4.1-mini',temperature=0,max_retries=2)
class Analyze_Feedback(BaseModel):
    sentiment:Literal['Positive','Negative','Mixed']=Field(description='One of the Sentiment selected')
    summary:str=Field(description='Summary of the feedback')
    pros:list[str]=Field(description='Pros of the Feedback')
    cons:list[str]=Field(description='Cons of the feedback')
    key_themes:list[str]=Field(description='Key themes of the feedback')

structured_prompt=ChatPromptTemplate.from_messages([("system","You are an expert Feedback Assistant \n""{format_instructions}"),("user","Extract the sentiment,summary,pros,cons and key_themes from {feedback}")])
parser=PydanticOutputParser(pydantic_object=Analyze_Feedback)
clean_input=RunnableLambda(lambda x:
    {
        "feedback":" ".join(x['feedback'].split()),
        "format_instructions":parser.get_format_instructions()
    })

pydantic_chain=RunnableSequence(clean_input,structured_prompt,llm,parser)
reply_prompt=ChatPromptTemplate.from_messages([('system','You are an Assistant whose job is to frame appropriate replies based on the sentiment extracted from the feedback.Use the analysis to guide its tone and avoid making unauthorized promises.Dont offer assistance, refunds, replacements, or escalation unless those services are explicitly supported by the company.'),('human','Customer Feedback:{feedback}\n' 'Sentiment:{sentiment}\n' 'Summary:{summary}\n' 'Write an appropriate customer service reply')])
positive_reply_prompt=ChatPromptTemplate.from_messages([('system','Act as a professional customer-service representative responding to positive feedback.Thank the customer, acknowledge specific positive points, and avoid unsupported promises.'),('human','Customer Feedback:{feedback}\n' 'Sentiment:{sentiment}\n' 'Summary:{summary}\n' 'Write a customer service reply')])
negative_reply_prompt=ChatPromptTemplate.from_messages([('system','Acknowledge the customers dissatisfaction.Express empathy and apologize appropriately.Address the specific complaints.Avoid promising refunds, replacements, escalations, or other actions that are not authorized.Keep the response professional and concise'),('human','Customer Feedback:{feedback}\n' 'Sentiment:{sentiment}\n' 'Summary:{summary}\n' 'Write a customer service reply')])
reply_parser=StrOutputParser()
reply_chain=RunnableSequence(reply_prompt,llm,reply_parser)
combined_chain=RunnablePassthrough.assign(
    analysis=pydantic_chain
    )

input_data={
    'feedback':'Samsung phone has a poor battery',
    'format_instructions':parser.get_format_instructions()
}
prepare_reply_input=RunnableLambda(lambda x:
    {
        'feedback':x['feedback'],
        'sentiment':x['analysis'].sentiment,
        'summary':x['analysis'].summary
    })
result1_chain=RunnableSequence(combined_chain,prepare_reply_input,reply_prompt,llm,reply_parser)
positive_reply_chain=RunnableSequence(positive_reply_prompt,llm,reply_parser)
negative_reply_chain=RunnableSequence(negative_reply_prompt,llm,reply_parser)
default_chain=RunnableSequence(reply_prompt,llm,reply_parser)
sentiment_branch=RunnableBranch(
    
        (lambda x:x['sentiment']=='Positive',positive_reply_chain),
        (lambda x:x['sentiment']=='Negative',negative_reply_chain),
        default_chain
)
final_reply_chain=RunnableSequence(combined_chain,prepare_reply_input,sentiment_branch)
response=final_reply_chain.invoke({'feedback':
    'Terrible phone. The battery drains quickly and the camera is disappointing.','format_instructions':parser.get_format_instructions()})
reply_generation=RunnableSequence(prepare_reply_input,sentiment_branch)
reply_with_analysis=RunnablePassthrough.assign(reply=reply_generation)
final_chain=RunnableSequence(combined_chain,reply_with_analysis)
ai_conversation_template=ChatPromptTemplate.from_messages([('system','Act as a customer feedback support assistant'),MessagesPlaceholder(variable_name='messages'),('human','\n''{question}')])
ai_conversation_chain=RunnableSequence(ai_conversation_template,llm,StrOutputParser())