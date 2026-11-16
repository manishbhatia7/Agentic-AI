from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
vector_store=Chroma(collection_name="company_policies",embedding_function=model,persist_directory='./chroma_db')
#print("Stored Records:",vector_store._collection.count())
question="What is the company's leave policy?."
#results=vector_store.similarity_search(question,k=3)
results=vector_store.similarity_search_with_score(question,k=3)
print(len(results))
#print(results[0].page_content)
#print(results[0].metadata)
#for result in results:
    #print(result.metadata["source"])
    #print(result.metadata["page"])
    #print(result.page_content)
for document,distance_score in results:
    print(document.metadata["source"])
    print(document.metadata["page"])
    print(document.page_content)
    print(distance_score)
        
