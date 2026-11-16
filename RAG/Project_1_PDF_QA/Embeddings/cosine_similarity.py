from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
model=OpenAIEmbeddings(model='text-embedding-3-small')
sentences = [
    "Employees are entitled to annual leave.",
    "Staff members can apply for vacation.",
    "The company provides laptops to employees."
]
vectors=model.embed_documents(sentences)
similarity1=cosine_similarity([vectors[0],vectors[1]])
similarity2=cosine_similarity([vectors[1],vectors[2]])
similarity3=cosine_similarity([vectors[0],vectors[2]])
print(similarity1[0][1])
print(similarity2[0][1])
print(similarity3[0][1])

