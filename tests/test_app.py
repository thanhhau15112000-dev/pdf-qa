import io
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app import llm, main
from app.models import Answer, AnswerItem, Citation
from app.pdf import InvalidPDF, extract_pdf, verify_answer


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-no-live-requests")
    monkeypatch.setenv("GROQ_API_KEY", "test-only-no-live-requests")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK", "false")
    monkeypatch.setenv("APP_ACCESS_CODE", "test-code")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-not-for-deployment")


def make_pdf(text="The report deadline is 12 October 2026. Budget is 120 USD.", encrypted=False, blank=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    if not blank:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
    if encrypted:
        writer.encrypt("test-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def answer(quote="Budget is 120 USD.", page=1, status="supported"):
    return Answer(
        items=[
            AnswerItem(title="Ngân sách", status=status, answer="120 USD", citations=[Citation(page=page, quote=quote)])
        ],
        notice="",
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "test-code")
    main.documents.clear()
    main.jobs.clear()
    main.answers.clear()
    main.attempts.clear()
    main.usage.clear()
    with TestClient(main.app) as client:
        assert client.post("/api/unlock", json={"code": "test-code"}).status_code == 200
        yield client


def upload(client):
    response = client.post("/api/documents", files={"file": ("exam.pdf", make_pdf(), "application/pdf")})
    assert response.status_code == 200
    return response.json()["id"]


def poll(client, job):
    for _ in range(100):
        state = client.get(f"/api/jobs/{job}").json()
        if state["status"] != "pending":
            return state
        time.sleep(0.01)
    pytest.fail("job remained pending")


def test_pdf_extraction_and_invalid_states():
    pages, warnings = extract_pdf(make_pdf())
    assert "120 USD" in pages[0] and warnings
    for data in (
        b"not-pdf",
        b"%PDF-broken",
        make_pdf(encrypted=True),
        make_pdf(blank=True),
        b"%PDF-" + b"x" * (10 * 1024 * 1024),
    ):
        with pytest.raises(InvalidPDF):
            extract_pdf(data)


@pytest.mark.parametrize(
    "result",
    [
        answer(quote="Budget is 999 USD."),
        answer(page=2),
        Answer(items=[AnswerItem(title="Test", status="inference", answer="Guess", citations=[])], notice=""),
        answer(status="conflict"),
    ],
)
def test_fabricated_or_missing_sources_block_entire_item(result):
    verified = verify_answer(result, ["Budget is 120 USD."])
    assert verified.items[0].status == "insufficient"
    assert "999" not in verified.items[0].answer
    assert verified.items[0].citations == []


def test_valid_quote_and_unicode_whitespace():
    result = verify_answer(answer(), ["Budget   is\n120 USD."])
    assert result.items[0].status == "supported"


def test_auth_and_same_origin(client):
    other = TestClient(main.app)
    assert other.post("/api/documents", files={"file": ("x.pdf", make_pdf())}).status_code == 401
    assert other.post("/api/unlock", json={"code": "wrong"}).status_code == 401
    assert other.post("/api/unlock", json={"code": "mã sai"}).status_code == 401
    response = client.post("/api/unlock", json={"code": "test-code"}, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert (
        client.post("/api/unlock", json={"code": "test-code"}, headers={"Origin": "http://testserver"}).status_code
        == 200
    )


def test_document_and_job_ownership_delete_and_expiry(client):
    doc = upload(client)
    assert client.get(f"/api/documents/{doc}/pages/1").status_code == 200
    assert client.get(f"/api/documents/{doc}/pages/2").status_code == 404
    other = TestClient(main.app)
    other.post("/api/unlock", json={"code": "test-code"})
    assert other.get(f"/api/documents/{doc}/pages/1").status_code == 404
    assert other.delete(f"/api/documents/{doc}").status_code == 404
    main.jobs["private-job"] = {"owner": main.documents[doc].owner, "status": "done", "created": time.monotonic()}
    assert other.get("/api/jobs/private-job").status_code == 404
    main.documents[doc].created -= main.TTL + 1
    assert client.get(f"/api/documents/{doc}/pages/1").status_code == 404
    fresh = upload(client)
    assert client.delete(f"/api/documents/{fresh}").status_code == 200
    assert client.get(f"/api/documents/{fresh}/pages/1").status_code == 404


def test_ask_and_solve_job(client, monkeypatch):
    seen = []

    async def fake(pages, question, mode):
        seen.append((question, mode))
        return answer()

    monkeypatch.setattr(main, "answer_pdf", fake)
    doc = upload(client)
    assert client.post("/api/ask", json={"document_id": doc, "question": "  "}).status_code == 400
    for mode in ("ask", "solve"):
        response = client.post("/api/ask", json={"document_id": doc, "question": "Budget?", "mode": mode})
        assert response.status_code == 202
        state = poll(client, response.json()["job_id"])
        assert state["status"] == "done"
        assert state["result"]["items"][0]["answer"] == "120 USD"
    assert [item[1] for item in seen] == ["ask", "solve"]


def test_repeated_question_is_served_from_cache(client, monkeypatch):
    calls = []

    async def fake(pages, question, mode):
        calls.append(question)
        return answer()

    monkeypatch.setattr(main, "answer_pdf", fake)
    doc = upload(client)
    for question in ("Budget?", "  Budget?  ", "Deadline?"):
        response = client.post("/api/ask", json={"document_id": doc, "question": question})
        assert poll(client, response.json()["job_id"])["result"]["items"][0]["answer"] == "120 USD"
    assert calls == ["Budget?", "Deadline?"]


def test_provider_error_is_user_visible_without_secret(client, monkeypatch):
    async def fake(*args):
        raise llm.ProviderError("Đã hết quota")

    monkeypatch.setattr(main, "answer_pdf", fake)
    doc = upload(client)
    response = client.post("/api/ask", json={"document_id": doc, "question": "Budget?"})
    state = poll(client, response.json()["job_id"])
    assert state == {"status": "error", "error": "Đã hết quota"}


def test_follow_up_passes_original_question(client, monkeypatch):
    seen = []

    async def fake(pages, question, mode, context_question):
        seen.append((question, context_question))
        return answer()

    monkeypatch.setattr(main, "answer_pdf", fake)
    doc = upload(client)
    response = client.post(
        "/api/ask", json={"document_id": doc, "question": "Project A", "context_question": "Budget?"}
    )
    assert poll(client, response.json()["job_id"])["status"] == "done"
    assert seen == [("Project A", "Budget?")]


def test_capacity_rate_limit_and_logout(client):
    docs = [upload(client) for _ in range(3)]
    assert client.post("/api/documents", files={"file": ("x.pdf", make_pdf())}).status_code == 409
    main.jobs["busy"] = {"owner": main.documents[docs[0]].owner, "status": "pending", "created": time.monotonic()}
    assert client.post("/api/ask", json={"document_id": docs[0], "question": "Budget?"}).status_code == 409
    main.jobs.clear()
    bucket = main.hashlib.sha256(b"test-code").hexdigest()
    main.usage[bucket].extend([time.monotonic()] * 60)
    assert client.post("/api/ask", json={"document_id": docs[0], "question": "Budget?"}).status_code == 429
    assert client.post("/api/logout").status_code == 200
    assert not main.documents
    assert client.get("/api/session").json()["unlocked"] is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_upstream_errors_never_forward_raw_response(monkeypatch, status):
    async def handler(request):
        return httpx.Response(status, json={"error": {"message": "secret-from-upstream"}})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    with pytest.raises(llm.ProviderError) as error:
        main.asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert "secret-from-upstream" not in str(error.value)


def test_live_payload_schema_and_fabricated_quote_rejection(monkeypatch):
    async def handler(request):
        payload = json.loads(request.content)
        assert "JSON SCHEMA BẮT BUỘC" in payload["system_instruction"]
        assert payload["response_format"]["mime_type"] == "application/json"
        assert payload["store"] is False
        assert "RANH GIỚI TIN CẬY" in payload["system_instruction"]
        result = answer(quote="Budget is 999 USD.").model_dump_json()
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": result}]}],
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    result = main.asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert result.items[0].status == "insufficient"


@pytest.mark.parametrize(
    "body",
    [
        {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "not JSON"}]}]},
        {"status": "incomplete", "steps": []},
        {
            "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": '{"items":[],"notice":""}'}]}],
        },
    ],
)
def test_malformed_and_incomplete_llm_outputs_fail_closed(monkeypatch, body):
    async def handler(request):
        return httpx.Response(200, json=body)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    with pytest.raises(llm.ProviderError):
        main.asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
