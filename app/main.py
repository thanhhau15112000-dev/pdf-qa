import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from .llm import ProviderError, answer_pdf, provider_settings
from .models import AskRequest, UnlockRequest
from .pdf import MAX_BYTES, InvalidPDF, extract_pdf

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")
TTL = 7200
documents = {}
jobs = {}
answers = {}  # Answer cache: identical PDF + question costs no provider quota.
tasks = set()
attempts = defaultdict(deque)
usage = defaultdict(deque)


@dataclass
class Document:
    owner: str
    name: str
    pages: list[str]
    warnings: list[str]
    created: float
    digest: str


def cleanup():
    cutoff = time.monotonic() - TTL
    for collection in (documents, jobs, answers):
        for key in list(collection):
            created = collection[key].created if collection is documents else collection[key]["created"]
            if created < cutoff:
                del collection[key]
    for collection in (attempts, usage):
        for key in list(collection):
            while collection[key] and collection[key][0] < cutoff:
                collection[key].popleft()
            if not collection[key]:
                del collection[key]


@asynccontextmanager
async def lifespan(app):
    if not os.getenv("APP_ACCESS_CODE") or not os.getenv("SESSION_SECRET"):
        raise RuntimeError("Cần cấu hình APP_ACCESS_CODE và SESSION_SECRET trong .env / Render Environment.")
    yield
    for task in tuple(tasks):
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="PDF Hỏi đáp", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET") or secrets.token_hex(32),
    same_site="lax",
    https_only=os.getenv("SECURE_COOKIES", "false").lower() == "true",
    max_age=TTL,
)


@app.middleware("http")
async def headers(request: Request, call_next):
    # Session-authenticated mutation must originate from this site's browser UI.
    origin = request.headers.get("origin")
    if request.method in {"POST", "DELETE"} and origin and urlparse(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Nguồn yêu cầu không hợp lệ."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; form-action 'self'; base-uri 'self'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def owner(request: Request) -> str:
    if not request.session.get("unlocked"):
        raise HTTPException(401, "Nhập mã truy cập để sử dụng app.")
    cleanup()
    return request.session["owner"]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/session")
def session(request: Request):
    provider, model = provider_settings()
    return {"unlocked": bool(request.session.get("unlocked")), "provider": provider, "model": model}


@app.post("/api/unlock")
def unlock(body: UnlockRequest, request: Request):
    cleanup()
    address = request.client.host if request.client else "unknown"
    now = time.monotonic()
    history = attempts[address]
    while history and history[0] < now - 60:
        history.popleft()
    if len(history) >= 10:
        raise HTTPException(429, "Nhập mã sai quá nhiều lần. Hãy thử lại sau 1 phút.")
    history.append(now)
    if not secrets.compare_digest(body.code.encode("utf-8"), os.getenv("APP_ACCESS_CODE", "").encode("utf-8")):
        raise HTTPException(401, "Mã truy cập không đúng.")
    request.session["unlocked"] = True
    request.session.setdefault("owner", secrets.token_urlsafe(24))
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request):
    identity = request.session.get("owner")
    for key in list(documents):
        if documents[key].owner == identity:
            del documents[key]
    request.session.clear()
    return {"ok": True}


@app.post("/api/documents")
async def upload(file: UploadFile, identity: str = Depends(owner)):
    try:
        data = await file.read(MAX_BYTES + 1)
    finally:
        await file.close()
    if len(documents) >= 30:
        raise HTTPException(503, "Server đang giữ quá nhiều PDF. Hãy thử lại sau.")
    if sum(doc.owner == identity for doc in documents.values()) >= 3:
        raise HTTPException(409, "Mỗi phiên tối đa 3 PDF. Hãy xóa file cũ trước.")
    try:
        pages, warnings = await run_in_threadpool(extract_pdf, data)
    except InvalidPDF as exc:
        raise HTTPException(400, str(exc)) from exc
    # Recheck limits after the asynchronous parser to avoid concurrent-upload overflow.
    if len(documents) >= 30 or sum(doc.owner == identity for doc in documents.values()) >= 3:
        raise HTTPException(409, "Đã đạt giới hạn lưu PDF. Hãy xóa file cũ.")
    name = Path((file.filename or "document.pdf").replace("\\", "/")).name[:200]
    document_id = secrets.token_urlsafe(24)
    digest = hashlib.sha256(data).hexdigest()
    documents[document_id] = Document(identity, name, pages, warnings, time.monotonic(), digest)
    return {
        "id": document_id,
        "name": name,
        "pages": len(pages),
        "characters": sum(map(len, pages)),
        "warnings": warnings,
    }


def get_document(document_id, identity):
    document = documents.get(document_id)
    if not document or document.owner != identity:
        raise HTTPException(404, "PDF không tồn tại hoặc đã hết phiên. Hãy tải lại file.")
    return document


@app.get("/api/documents/{document_id}/pages/{page}")
def source(document_id: str, page: int, identity: str = Depends(owner)):
    document = get_document(document_id, identity)
    if not 1 <= page <= len(document.pages):
        raise HTTPException(404, "Trang không tồn tại.")
    return {"page": page, "text": document.pages[page - 1]}


@app.delete("/api/documents/{document_id}")
def delete(document_id: str, identity: str = Depends(owner)):
    get_document(document_id, identity)
    del documents[document_id]
    return {"ok": True}


async def run_job(job_id, pages, question, mode, context_question="", cache_key=""):
    try:
        result = (
            await answer_pdf(pages, question, mode, context_question)
            if context_question
            else await answer_pdf(pages, question, mode)
        )
        jobs[job_id].update(status="done", result=result.model_dump())
        if cache_key:
            if len(answers) >= 200:
                del answers[next(iter(answers))]
            answers[cache_key] = {"created": time.monotonic(), "result": jobs[job_id]["result"]}
    except ProviderError as exc:
        jobs[job_id].update(status="error", error=str(exc))
    except Exception as exc:
        logging.getLogger(__name__).warning("Unexpected job failure: %s", type(exc).__name__)
        jobs[job_id].update(status="error", error="Không xử lý được câu hỏi. Hãy thử lại.")


@app.post("/api/ask", status_code=202)
async def ask(body: AskRequest, identity: str = Depends(owner)):
    document = get_document(body.document_id, identity)
    if body.mode == "ask" and not body.question.strip():
        raise HTTPException(400, "Hãy nhập câu hỏi.")
    if any(job["owner"] == identity and job["status"] == "pending" for job in jobs.values()):
        raise HTTPException(409, "Câu hỏi trước đang xử lý. Hãy chờ kết quả.")
    if sum(job["status"] == "pending" for job in jobs.values()) >= 4:
        raise HTTPException(503, "Server đang bận. Hãy thử lại sau.")
    now = time.monotonic()
    question, context = body.question.strip(), body.context_question.strip()
    cache_key = hashlib.sha256(
        json.dumps([document.digest, body.mode, question if body.mode == "ask" else "", context]).encode()
    ).hexdigest()
    if cache_key in answers:
        job_id = secrets.token_urlsafe(24)
        jobs[job_id] = {"owner": identity, "status": "done", "created": now, "result": answers[cache_key]["result"]}
        return {"job_id": job_id}
    # One access code is intentionally shared; the global limit also bounds new-session bypass.
    code_bucket = hashlib.sha256(os.getenv("APP_ACCESS_CODE", "").encode()).hexdigest()
    history = usage[code_bucket]
    while history and history[0] < now - 3600:
        history.popleft()
    if len(history) >= 60:
        raise HTTPException(429, "App đã đạt 60 lượt hỏi/giờ. Hãy thử lại sau.")
    history.append(now)
    job_id = secrets.token_urlsafe(24)
    jobs[job_id] = {"owner": identity, "status": "pending", "created": now}
    task = asyncio.create_task(run_job(job_id, document.pages, question, body.mode, context, cache_key))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, identity: str = Depends(owner)):
    job = jobs.get(job_id)
    if not job or job["owner"] != identity:
        raise HTTPException(404, "Lượt hỏi đã hết phiên. Hãy gửi lại câu hỏi.")
    return {key: job[key] for key in ("status", "result", "error") if key in job}


@app.get("/")
def index():
    return FileResponse(BASE / "app/static/index.html")


app.mount("/static", StaticFiles(directory=BASE / "app/static"), name="static")
