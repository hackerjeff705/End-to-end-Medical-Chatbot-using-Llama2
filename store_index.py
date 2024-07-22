import os
import pinecone
from dotenv import load_dotenv
# from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
namespace = "wondervector5000"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

# Creating Embeddings for Each of the Text Chunks & storing
doc_chunks = [t.page_content for t in text_chunks]

try:
    docsearch = PineconeVectorStore.from_texts(
            doc_chunks,
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
    print("CHUNKS PROCESSED")
except:
    print("ERROR! Chunks can't be processed or server timeout")
