from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
embedding=model.embed_query("Employees are entitled to annual leave.")
#print(type(embedding))
#print(embedding[0:6])
#print(len(embedding))
sentences = [
    "Employees are entitled to annual leave.",
    "Staff members can apply for vacation.",
    "The company provides laptops to employees."
]
vectors=model.embed_documents(sentences)
print(len(vectors))
print(len(vectors[1]))
print(vectors[1][:5])