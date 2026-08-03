import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services import rag

app = FastAPI(title="SmartLearn Lite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(
    chat_id: str = Query(..., description="Chat session identifier"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File must not be empty")

    try:
        pages = rag.extract_pages_from_bytes_for_rag(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    characters = sum(len(p["text"]) for p in pages)
    if characters == 0:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in this PDF. OCR is not supported.",
        )

    try:
        documents[chat_id] = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            pages=pages,
        )
    except Exception:
        documents.pop(chat_id, None)
        raise HTTPException(
            status_code=500,
            detail="Failed to process the PDF. Please try again.",
        )

    return rag.build_upload_response(documents[chat_id])


@app.get("/documents/{chat_id}/file")
async def get_document_file(chat_id: str):
    """Serve the uploaded PDF file for a given chat session."""
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"No document found for chat_id '{chat_id}'.")

    saved_path = document.get("saved_pdf_path")
    if saved_path is None or not os.path.exists(saved_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    return FileResponse(saved_path, media_type="application/pdf")


@app.post("/chat")
async def chat(request: ChatRequest):
    document = documents.get(request.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{request.chat_id}'. "
            "Please upload a PDF first via POST /upload.",
        )

    try:
        result = rag.answer_chat_turn(document, request.message)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"{type(e).__name__}: {e}"
        )

    return result
