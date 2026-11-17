from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import os
BASE_DIR=Path(__file__).resolve().parent
CHROMA_PATH=(
    BASE_DIR.parent
/ "Project_1_PDF_QA"
/ "chroma_db"
)
load_dotenv()
llm=os.getenv("OPENAI_MODEL")
embeddings_model=os.getenv("EMBEDDING_MODEL")
collections=os.getenv("CHROMA_COLLECTION")
model=OpenAIEmbeddings(model=embeddings_model)
vector_store=Chroma(collection_name=collections,embedding_function=model,persist_directory=CHROMA_PATH)
print("Stored Records:",vector_store._collection.count())

