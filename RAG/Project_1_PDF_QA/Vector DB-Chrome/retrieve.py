from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
vector_store=Chroma(collection_name="company_policies",embedding_function=model,persist_directory=r'D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\chroma_db')
print("Stored Records:",vector_store._collection.count())
question="What is the monthly work-from-home allowance at NimbusTech Solutions?"
#results=vector_store.similarity_search(question,k=3)
results=vector_store.similarity_search_with_score(question,k=3)
#print(results[0].page_content)
#print(results[0].metadata)
#for result in results:
    #print(result.metadata["source"])
    #print(result.metadata["page"])
    #print(result.page_content)

context=[]    
for document,distance_score in results:
    context.append(document.page_content)
context="\n\n".join(context)
prompt=ChatPromptTemplate.from_messages([('system','You are company policy assistant whose job is to provide answers for employee questions referring company documents.Answer only in the context provided.If the context {context} does not contain the answer, say you do not know.Dont invent company policies'),("human","{question}")])
formatted_prompt=prompt.invoke({
    'context':context,
    "question":question
}
)      
llm=ChatOpenAI(model="gpt-4.1-mini")
answer_chain=prompt | llm | StrOutputParser()
answer=answer_chain.invoke({
    'context':context,
    "question":question
})
print(answer)

