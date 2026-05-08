import os
import uuid
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_knowledge_base")
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/gemini-embedding-001")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

app = FastAPI(title="RAG Document QA API")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing. Set it in the .env file to use Gemini.",
        )
    candidate_models = [
        EMBEDDING_MODEL_NAME,
        "models/gemini-embedding-001",
        "models/gemini-embedding-2",
    ]
    last_error: Exception | None = None
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=model_name, api_key=GOOGLE_API_KEY)
            # Quick probe so we fail early with a clear message.
            embeddings.embed_query("health check")
            return embeddings
        except Exception as exc:
            last_error = exc

    raise HTTPException(
        status_code=400,
        detail=(
            "No supported Gemini embedding model worked for this key. "
            "Try EMBEDDING_MODEL_NAME=models/gemini-embedding-001."
        ),
    ) from last_error


def get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is missing. Set it in the .env file to use Gemini.",
        )
    candidate_models = [
        MODEL_NAME,
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
    ]
    last_error: Exception | None = None
    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=GOOGLE_API_KEY)
            # Probe once to ensure the model is available for this key/API version.
            llm.invoke("Health check. Reply with OK.")
            return llm
        except Exception as exc:
            last_error = exc

    raise HTTPException(
        status_code=400,
        detail=(
            "No supported Gemini chat model worked for this key. "
            "Try MODEL_NAME=models/gemini-2.0-flash or models/gemini-2.5-flash."
        ),
    ) from last_error


class AskRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    document_id: str
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    vector_store = get_vector_store()
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "unknown"
    content_type = file.content_type or ""
    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

    if is_pdf:
        try:
            pdf_reader = PdfReader(BytesIO(raw_bytes))
            pages_text = [(page.extract_text() or "") for page in pdf_reader.pages]
            text = "\n".join(pages_text).strip()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to read PDF content. Upload a valid text-based PDF.",
            ) from exc
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload UTF-8 text or PDF.",
            ) from exc

    document_id = str(uuid.uuid4())
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text content found.")

    documents = [
        Document(
            page_content=chunk,
            metadata={"document_id": document_id, "filename": filename},
        )
        for chunk in chunks
    ]
    ids = [f"{document_id}-{index}" for index in range(len(documents))]
    try:
        vector_store.add_documents(documents=documents, ids=ids)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Embedding failed for model '{EMBEDDING_MODEL_NAME}'. "
                "Use a supported Gemini embedding model (for example: models/gemini-embedding-001)."
            ),
        ) from exc

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_stored": len(documents),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(
        query=payload.question,
        k=payload.top_k,
        filter={"document_id": payload.document_id},
    )
    if not retrieved_docs:
        raise HTTPException(
            status_code=404,
            detail="No content found for this document_id. Upload document first.",
        )

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.question}\n"
        "If the answer is not in context, say you do not know."
    )
    llm = get_llm()
    answer = llm.invoke(prompt).content
    return AskResponse(document_id=payload.document_id, answer=answer)
