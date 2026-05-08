import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Setup Connection String (Postgres with pgvector)
connection = "postgresql+psycopg://postgres:ali@localhost:5432/my_vector_db"
collection_name = "my_knowledge_base"

def run_demo() -> None:
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/embedding-001")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is missing. Set it in the .env file to use Gemini.")

    # 2. Prepare Data
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=GOOGLE_API_KEY)
    text_data = "Gemini 3 Flash is a fast, efficient AI model released in 2026."
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text_data)]

    # 3. Ingest into PostgreSQL
    vector_store = PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    # 4. The Retrieval Pipeline
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, api_key=GOOGLE_API_KEY)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    # 5. Ask a question
    query = "When was Gemini 3 Flash released?"
    response = rag_chain.invoke(query)
    print(response["result"])


if __name__ == "__main__":
    run_demo()