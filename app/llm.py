import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from pydantic import ValidationError

from .models import Answer
from .pdf import verify_answer

ROOT = Path(__file__).parent
SYSTEM = (ROOT / "system_prompt.txt").read_text(encoding="utf-8")
EXAMPLES = json.loads((ROOT / "common_examples.json").read_text(encoding="utf-8"))


class ProviderError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


async def answer_pdf(pages: list[str], question: str, mode: str) -> Answer:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    if not key:
        raise ProviderError("Server chưa được cấu hình GEMINI_API_KEY.", 503)
    if not re.fullmatch(r"gemini-[a-z0-9.-]+", model):
        raise ProviderError("GEMINI_MODEL không hợp lệ.", 503)
    payload = {
        "model": model,
        "store": False,
        "system_instruction": SYSTEM
        + "\nVÍ DỤ CHUẨN (KHÔNG PHẢI PDF THẬT):\n"
        + json.dumps(EXAMPLES, ensure_ascii=False)
        + "\nJSON SCHEMA BẮT BUỘC:\n"
        + json.dumps(Answer.model_json_schema(), ensure_ascii=False),
        "input": json.dumps(
            {
                "mode": mode,
                "question": question if mode == "ask" else "Tìm và giải tối đa 12 câu hỏi/bài tập đầu tiên trong PDF.",
                "pdf_pages": [{"page": i + 1, "text": text} for i, text in enumerate(pages)],
            },
            ensure_ascii=False,
        ),
        "generation_config": {"max_output_tokens": 12000, "thinking_level": "low"},
        "response_format": {"type": "text", "mime_type": "application/json"},
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            for attempt in range(2):
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/interactions",
                    headers={"x-goog-api-key": key},
                    json=payload,
                )
                if response.status_code not in {502, 503, 504} or attempt == 1:
                    break
                await asyncio.sleep(1)
    except httpx.TimeoutException as exc:
        raise ProviderError("Gemini phản hồi quá lâu. Hãy thử lại hoặc chia nhỏ PDF.", 504) from exc
    except httpx.RequestError as exc:
        raise ProviderError("Không kết nối được Gemini. Hãy thử lại sau.") from exc
    if response.status_code != 200:
        messages = {
            400: "Gemini từ chối cấu hình/yêu cầu. Kiểm tra API key và model trên server.",
            401: "API key không hợp lệ. Hãy cập nhật key trên server.",
            403: "API key không có quyền dùng Gemini hoặc đã bị vô hiệu hóa.",
            404: "Tài khoản chưa truy cập được model đã cấu hình.",
            429: "Đã hết quota hoặc vượt giới hạn Gemini. Hãy thử lại sau.",
        }
        raise ProviderError(messages.get(response.status_code, "Gemini đang gặp lỗi. Hãy thử lại sau."))
    try:
        body = response.json()
        outputs = [step for step in body.get("steps", []) if step.get("type") == "model_output"]
        if body.get("status") != "completed" or not outputs:
            raise ProviderError("Gemini không hoàn tất câu trả lời. Hãy hỏi ngắn hơn hoặc chia nhỏ PDF.")
        result = "".join(part["text"] for part in outputs[-1]["content"] if part.get("type") == "text")
        answer = Answer.model_validate_json(result)
    except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
        raise ProviderError("Gemini trả về dữ liệu không đúng cấu trúc. Hãy thử lại.") from exc
    return verify_answer(answer, pages)
