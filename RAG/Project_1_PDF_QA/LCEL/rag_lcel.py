from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
vector_store=Chroma(collection_name="company_policies",embedding_function=model,persist_directory=r'D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\chroma_db')
print("Stored Records:",vector_store._collection.count())
retriever=vector_store.as_retriever(search_kwargs={"k":3})
question="What is the company's sick leave policy?"
documents=retriever.invoke(question)
print(len(documents))
#print(type(documents))
#print(documents[0])
passthrough=RunnablePassthrough()
result=passthrough.invoke(question)
#print(result)
def make_lowercase(string):
    return string.lower()
formatter=RunnableLambda(make_lowercase)
result=formatter.invoke('HELLO RAG')
#print(result)
def format_docs(documents):
    context=[]
    for document in documents:
        context.append(document.page_content)
    return "\n\n".join(context)
context_chain=retriever | RunnableLambda(format_docs)
result=context_chain.invoke(question)
#print(result)    
context_chain=retriever | RunnableLambda(format_docs)    
context=context_chain.invoke(question)
print(context)
    