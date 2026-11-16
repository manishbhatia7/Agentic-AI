from langchain_community.document_loaders import PyPDFLoader
pdf=PyPDFLoader(r"D:\My work\Agentic-AI\RAG\Project_1_PDF_QA\IT_Company_Policies.pdf")
documents=pdf.load()
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
print(documents[1].page_content)
print(documents[1].metadata['page'])