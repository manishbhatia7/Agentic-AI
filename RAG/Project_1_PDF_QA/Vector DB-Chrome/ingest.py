from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
pdf1=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\IT_Company_Policies.pdf")
pdf2=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\HR_Policy.pdf")
pdf3=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\Leave_Policy.pdf")
document1=pdf1.load()
document2=pdf2.load()
document3=pdf3.load()
complete_document=document1+document2+document3
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.split_documents(complete_document)
texts=[chunk.page_content for chunk in chunks]
#pdf_vectors=model.embed_documents(texts)
for i, chunk in enumerate(chunks):
    chunk.id = str(i)
vector_store = Chroma(
    collection_name="company_policies",
    embedding_function=model,
    persist_directory="./chroma_db"
)

vector_store.add_documents(documents=chunks)
print("Stored records:", vector_store._collection.count())
