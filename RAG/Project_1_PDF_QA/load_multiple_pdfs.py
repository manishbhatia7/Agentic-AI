from langchain_community.document_loaders import PyPDFLoader
pdf1=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\IT_Company_Policies.pdf")
pdf2=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\HR_Policy.pdf")
pdf3=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\Leave_Policy.pdf")
document1=pdf1.load()
document2=pdf2.load()
document3=pdf3.load()
complete_document=document1+document2+document3
print(len(complete_document))
print(complete_document[0].metadata["source"])
print(complete_document[-1].metadata["source"])

