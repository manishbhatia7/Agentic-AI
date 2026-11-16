from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf1=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\IT_Company_Policies.pdf")
pdf2=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\HR_Policy.pdf")
pdf3=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\Leave_Policy.pdf")
document1=pdf1.load()
document2=pdf2.load()
document3=pdf3.load()
complete_document=document1+document2+document3
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.split_documents(complete_document)
print(len(chunks))
print(chunks[0].page_content)
print(len(chunks[0].page_content))
print(chunks[0].metadata)
print(len(chunks[1].page_content))
