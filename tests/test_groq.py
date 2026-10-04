import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app import llm
from app.models import Answer, AnswerItem, Citation
from app.pdf import verify_answer


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-no-live-requests")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-no-live-requests")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_FALLBACK", "true")


def result(**changes):
    fields = dict(
        title="Ngân sách",
        status="supported",
        answer="120 USD",
        citations=[Citation(page=1, quote="Budget is 120 USD.")],
    )
    fields.update(changes)
    return Answer(items=[AnswerItem(**fields)], notice="")


def wire(answer, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": answer.model_dump_json()}}]}


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )


def test_strict_schema_and_clarification_context(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url.host == "api.groq.com"
        assert payload["model"] == "openai/gpt-oss-120b"
        schema = payload["response_format"]["json_schema"]
        assert schema["strict"] is True
        for node in [schema["schema"], *schema["schema"]["$defs"].values()]:
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        assert "title" in schema["schema"]["$defs"]["AnswerItem"]["properties"]
        content = json.loads(payload["messages"][1]["content"])
        assert content["original_question"] == "Budget?"
        assert content["question"] == "Project A"
        assert payload["max_completion_tokens"] == 4096
        return httpx.Response(200, json=wire(result()))

    mock_http(monkeypatch, handler)
    response = asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Project A", "ask", "Budget?"))
    assert response.provider == "groq" and response.items[0].status == "supported"
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_bounded_fallback(monkeypatch, status):
    calls = []

    async def handler(request):
        calls.append(request.url.host)
        if request.url.host == "api.groq.com":
            return httpx.Response(status, json={"error": "SECRET"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": result().model_dump_json()}]}],
            },
        )

    mock_http(monkeypatch, handler)
    response = asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert calls == ["api.groq.com", "generativelanguage.googleapis.com"]
    assert response.provider == "gemini" and "dự phòng" in response.notice


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_configuration_errors_do_not_fallback(monkeypatch, status):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "SECRET"})

    mock_http(monkeypatch, handler)
    with pytest.raises(llm.ProviderError) as error:
        asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert len(calls) == 1 and "SECRET" not in str(error.value)


def test_both_providers_fail_without_retry(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "20"}, json={"error": "SECRET"})

    mock_http(monkeypatch, handler)
    with pytest.raises(llm.ProviderError) as error:
        asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert len(calls) == 2 and "20 giây" in str(error.value) and "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [
        wire(result(), "length"),
        {"choices": []},
        {"choices": [{"finish_reason": "stop", "message": {"content": "invalid"}}]},
    ],
)
def test_invalid_or_truncated_output_never_regenerates(monkeypatch, body):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handler)
    with pytest.raises(llm.ProviderError):
        asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"answer": "999 USD"},
        {"answer": "999USD"},
        {"answer": "120 USD https://fabricated.example"},
        {"status": "conflict", "citations": [Citation(page=1, quote="Budget is 120 USD.")] * 2},
        {"status": "supported", "missing_information": ["Tên dự án"], "answer": "I guess 120 USD"},
        {"status": "insufficient", "answer": "I guess 120 USD"},
        {"status": "insufficient", "answer": ""},
        {"status": "supported", "answer": ""},
        {"assumptions": ["Assume budget"]},
    ],
)
def test_guards_remove_guess_and_ask_again(changes):
    verified = verify_answer(result(**changes), ["Budget is 120 USD."])
    item = verified.items[0]
    assert item.status == "insufficient"
    assert "120 USD" not in item.answer and "999" not in item.answer
    assert item.missing_information and item.clarifying_question and not item.assumptions


def test_calculation_label_and_strict_page_type():
    quote = "A car travels 120 km in 3 hours."
    verified = verify_answer(
        result(status="inference", answer="120 / 3 = 40 km/h", citations=[Citation(page=1, quote=quote)]), [quote]
    )
    assert verified.items[0].status == "inference"
    with pytest.raises(ValidationError):
        Citation(page="1", quote="Budget is 120 USD.")


def test_guard_failure_does_not_call_fallback(monkeypatch):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, json=wire(result(answer="999 USD")))

    mock_http(monkeypatch, handler)
    response = asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
    assert len(calls) == 1 and response.items[0].status == "insufficient"


@pytest.mark.parametrize("fallback", ["true", "false"])
def test_timeout_fallback_policy(monkeypatch, fallback):
    monkeypatch.setenv("LLM_FALLBACK", fallback)
    calls = []

    async def handler(request):
        calls.append(request.url.host)
        if request.url.host == "api.groq.com":
            raise httpx.ReadTimeout("test timeout", request=request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": result().model_dump_json()}]}],
            },
        )

    mock_http(monkeypatch, handler)
    if fallback == "false":
        with pytest.raises(llm.ProviderError):
            asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
        assert calls == ["api.groq.com"]
    else:
        response = asyncio.run(llm.answer_pdf(["Budget is 120 USD."], "Budget?", "ask"))
        assert response.provider == "gemini" and len(calls) == 2
