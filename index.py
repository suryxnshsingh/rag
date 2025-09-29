from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).parent / "LLD.pdf"

#Loading the PDF file
loader = PyPDFLoader(pdf_path)
docs = loader.load()

#Splitting the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents=docs)

#Vectorizing the chunks
embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

#Creating the vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Vector store created successfully!")