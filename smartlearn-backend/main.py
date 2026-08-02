import os
import re

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.llm import answer_from_pages
from services.pdf import extract_pages

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

documents: dict[str, list[dict]] = {}


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
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    characters = sum(len(p["text"]) for p in pages)
    if characters == 0:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in this PDF. OCR is not supported.",
        )

    documents[chat_id] = pages

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": characters,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    pages = documents.get(request.chat_id)
    if pages is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{request.chat_id}'. "
            "Please upload a PDF first via POST /upload.",
        )

    try:
        answer = answer_from_pages(pages, request.message)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=502, detail="Upstream AI service failed"
        )

    cited = {int(m.group(1)) for m in re.finditer(r"[\[【]Page\s+(\d+)[\]】]", answer)}
    valid_pages = {p["page"] for p in pages}
    citations = sorted(cited & valid_pages)

    return {"answer": answer, "citations": citations}
