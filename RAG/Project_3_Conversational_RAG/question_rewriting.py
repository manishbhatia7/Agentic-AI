from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_chroma import Chroma
import os
from pathlib import Path
load_dotenv()
embedding_model = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    temperature=0
)

collection_name = os.getenv("CHROMA_COLLECTION")


from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
history=InMemoryChatMessageHistory()
history.add_message(HumanMessage(content="How many casual leaves do employees get?"))
history.add_message(AIMessage(content="Employees get 8 casual leaves annually."))
question="Can I carry them forward?"
from langchain_core.output_parsers import StrOutputParser
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Company Policy Assistant.

Rewrite employee follow-up questions into standalone questions using the previous conversation. Do not answer the questions.

Instructions:
1. Identify what pronouns such as "them" refer to.
2. Convert follow-up questions into standalone questions.
3. Preserve the original meaning.
4. Never answer the question

"""
),MessagesPlaceholder(variable_name="chat_history"),("user","{question}")])
chain=prompt_template | llm | StrOutputParser()
rewritten_question=chain.invoke({"chat_history":history.messages,"question":question})
print(question)
print(rewritten_question)
BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = (
    BASE_DIR.parent
    / "Project_1_PDF_QA"
    / "chroma_db"
)
vectorstore=Chroma(collection_name=collection_name,embedding_function=embedding_model,persist_directory=str(CHROMA_PATH))
print("Total stored chunks:", vectorstore._collection.count())

retriever=vectorstore.as_retriever(search_type='mmr',search_kwargs={'k':5,'fetch_k':15})
documents=retriever.invoke(rewritten_question)
"""for index,document in enumerate(documents):
    print(document.page_content)
    print(document.metadata.get("page_label","Unknown"))
    source=document.metadata.get("source","")
    file_name=Path(source).name if source else "Unknown"
    print(file_name)
print(len(documents))"""
answer_prompt=ChatPromptTemplate.from_messages([("system",
                                   """
1.You are a Company Policy Question Answering Assistant.



2.Answer questions using only the retrieved company policy context.



3.Do not invent information.



4.If the context doesn't contain the answer, say that the information is unavailable.



5.Keep answers concise and factual.
                                   
context:{context} """),("human","{question}")
])

def single_string(documents):
   context=[document.page_content for document in documents]
   return "\n\n".join(context)
answer_chain=answer_prompt | llm | StrOutputParser()
context=single_string(documents)
answer=answer_chain.invoke({'context':context,'question':rewritten_question})
print(answer)

def get_filename_pageno(docs):
    sources=[]
    seen=set()
    for document in docs:
        page_label=document.metadata.get("page_label","Unknown")
        source=document.metadata.get("source","")
        file_name=Path(source).name if source else "Unknown"
        citation=(file_name,page_label)
        if citation not in seen:
            sources.append(citation)
            seen.add(citation)
            
        
    return sources

unique_sources=get_filename_pageno(documents)
print(unique_sources)
    
    
