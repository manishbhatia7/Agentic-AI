from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf1=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\IT_Company_Policies.pdf")
pdf2=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\HR_Policy.pdf")
pdf3=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\Leave_Policy.pdf")
document1=pdf1.load()
document2=pdf2.load()
document3=pdf3.load()
complete_document=document1+document2+document3
#splitter1=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
#splitter2=RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)
#splitter3=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter1=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
splitter2=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
splitter3=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
chunks1=splitter1.split_documents(complete_document)
chunks2=splitter2.split_documents(complete_document)
chunks3=splitter3.split_documents(complete_document)
print("Configuration A:", len(chunks1))
print("Configuration B:", len(chunks2))
print("Configuration C:", len(chunks3))

# Step 6: Compare first chunk lengths
print("First chunk length A:", len(chunks1[0].page_content))
print("First chunk length B:", len(chunks2[0].page_content))

