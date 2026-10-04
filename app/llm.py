import json
import os
import re
from pathlib import Path

import httpx
from pydantic import ValidationError

from .models import Answer, AnswerResult
from .pdf import verify_answer

ROOT = Path(__file__).parent
SYSTEM = (ROOT / "system_prompt.txt").read_text(encoding="utf-8")
EXAMPLES = json.loads((ROOT / "common_examples.json").read_text(encoding="utf-8"))


class ProviderError(Exception):
    def __init__(self, message: str, status: int = 502, fallback_allowed: bool = False):
        super().__init__(message)
        self.status = status
        self.fallback_allowed = fallback_allowed


def provider_settings():
    provider = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "gemini").strip().lower()
    if provider not in {"groq", "gemini"}:
        raise ProviderError("LLM_PROVIDER cần là groq hoặc gemini.", 503)
    default = "openai/gpt-oss-120b" if provider == "groq" else "gemini-3.8-flash"
    return provider, os.getenv(f"{provider.upper()}_MODEL", default).strip()


def strict_schema():
    schema = Answer.model_json_schema()

    def prepare(node):
        if isinstance(node, dict):
            if isinstance(node.get("title"), str):
                node.pop("title")
            for key in ("default", "minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum"):
                node.pop(key, None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            for value in node.values():
                prepare(value)
        elif isinstance(node, list):
            for value in node:
                prepare(value)

    prepare(schema)
    return schema


async def request_provider(provider, model, content):
    name = "Groq" if provider == "groq" else "Gemini"
    key = os.getenv(f"{provider.upper()}_API_KEY", "").strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/._-]{0,100}", model):
        raise ProviderError(f"Cấu hình key/model {name} trên server không hợp lệ.", 503)
    instruction = SYSTEM + "\nVÍ DỤ CHUẨN (KHÔNG PHẢI PDF THẬT):\n" + json.dumps(EXAMPLES, ensure_ascii=False)
    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
            "temperature": 0.1,
            "max_completion_tokens": 4096,
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "pdf_answer", "strict": True, "schema": strict_schema()},
            },
        }
    else:
        url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        headers = {"x-goog-api-key": key}
        payload = {
            "model": model,
            "store": False,
            "system_instruction": instruction
            + "\nJSON SCHEMA BẮT BUỘC:\n"
            + json.dumps(Answer.model_json_schema(), ensure_ascii=False),
            "input": content,
            "generation_config": {"max_output_tokens": 4096, "thinking_level": "low"},
            "response_format": {"type": "text", "mime_type": "application/json"},
        }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{name} phản hồi quá lâu. Hãy chia nhỏ PDF.", 504, True) from exc
    except httpx.RequestError as exc:
        raise ProviderError(f"Không kết nối được {name}. Hãy thử lại sau.", fallback_allowed=True) from exc
    if response.status_code != 200:
        messages = {
            400: f"{name} từ chối cấu hình/yêu cầu. Kiểm tra model trên server.",
            401: f"API key {name} không hợp lệ. Cập nhật key trên server.",
            403: f"API key không có quyền dùng {name} hoặc đã bị vô hiệu hóa.",
            404: f"Tài khoản chưa truy cập được model {name} đã cấu hình.",
            413: f"Yêu cầu quá lớn cho {name}. Hãy chia nhỏ PDF.",
            429: f"{name} hết quota hoặc vượt giới hạn tốc độ. Hãy thử lại sau.",
        }
        message = messages.get(response.status_code, f"{name} đang gặp lỗi. Hãy thử lại sau.")
        retry_after = response.headers.get("retry-after", "")
        if response.status_code == 429 and retry_after.isdigit():
            message += f" Nhà cung cấp đề nghị chờ {min(int(retry_after), 86400)} giây."
        raise ProviderError(message, fallback_allowed=response.status_code in {429, 500, 502, 503, 504})
    try:
        body = response.json()
        if provider == "groq":
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise ProviderError(f"{name} chưa hoàn tất câu trả lời. Hãy hỏi ít câu hơn hoặc chia nhỏ PDF.")
            result = choice["message"]["content"]
        else:
            outputs = [step for step in body.get("steps", []) if step.get("type") == "model_output"]
            if body.get("status") != "completed" or not outputs:
                raise ProviderError(f"{name} không hoàn tất câu trả lời. Hãy hỏi ngắn hơn hoặc chia nhỏ PDF.")
            result = "".join(part["text"] for part in outputs[-1]["content"] if part.get("type") == "text")
        return Answer.model_validate_json(result)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError) as exc:
        raise ProviderError(f"{name} trả về dữ liệu không đúng cấu trúc. Hãy thử lại.") from exc


async def answer_pdf(pages: list[str], question: str, mode: str, context_question: str = "") -> AnswerResult:
    provider, model = provider_settings()
    content = json.dumps(
        {
            "mode": mode,
            "question": question if mode == "ask" else "Tìm và giải tối đa 12 câu hỏi/bài tập đầu tiên trong PDF.",
            "original_question": context_question,
            "pdf_pages": [{"page": i + 1, "text": text} for i, text in enumerate(pages)],
        },
        ensure_ascii=False,
    )
    used_fallback = False
    try:
        answer = await request_provider(provider, model, content)
    except ProviderError as first:
        alternate = "gemini" if provider == "groq" else "groq"
        enabled = os.getenv("LLM_FALLBACK", "true").lower() == "true"
        if not first.fallback_allowed or not enabled or not os.getenv(f"{alternate.upper()}_API_KEY"):
            raise
        default = "openai/gpt-oss-120b" if alternate == "groq" else "gemini-3.8-flash"
        fallback_model = os.getenv(f"{alternate.upper()}_MODEL", default).strip()
        try:
            answer = await request_provider(alternate, fallback_model, content)
        except ProviderError as second:
            raise ProviderError(f"{first} Dự phòng cũng thất bại: {second}") from second
        provider, model, used_fallback = alternate, fallback_model, True
    answer = verify_answer(answer, pages)
    if used_fallback:
        answer.notice = (
            "Đã dùng nhà cung cấp dự phòng vì lỗi dịch vụ hoặc giới hạn lượt gọi. " + answer.notice[:1400]
        ).strip()
    return AnswerResult(**answer.model_dump(), provider=provider, model=model)
