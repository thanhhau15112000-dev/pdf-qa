import asyncio
import json

import httpx

from app import llm
from app.models import Answer, AnswerItem, Citation
from app.pdf import normalize, verify_answer
from app.retrieval import CHUNK_CHARS, CONTEXT_CHARS, GAP, chunk_pages, select_context

FILLER = "Nội dung ôn tập chung về quản lý dự án, chi phí và nguồn lực của nhóm. "


def long_pages(count=30):
    pages = [(FILLER * 20).strip() for _ in range(count)]
    pages[17] += "\nCâu 3: Một xe đi 120 km trong 3 giờ. Tính vận tốc trung bình."
    pages[24] += "\nĐịnh nghĩa rủi ro thanh khoản là khả năng không đáp ứng nghĩa vụ chi trả đúng hạn."
    return pages


def test_small_pdf_is_sent_whole():
    pages = ["Trang một.", "Trang hai."]
    assert select_context(pages, "bất kỳ", "ask") == (
        [{"page": 1, "text": "Trang một."}, {"page": 2, "text": "Trang hai."}],
        False,
    )


def test_chunks_are_bounded_and_quotable_from_original_page():
    pages = long_pages() + ["x" * 2000]
    chunks = chunk_pages(pages)
    assert all(len(text) <= CHUNK_CHARS for _, text in chunks)
    assert all(normalize(text) in normalize(pages[page - 1]) for page, text in chunks)


def test_large_pdf_retrieves_relevant_pages_within_budget():
    pages = long_pages()
    selected, excerpted = select_context(pages, "cau 3 van toc", "ask")  # unaccented query
    assert excerpted and sum(len(item["text"]) for item in selected) <= CONTEXT_CHARS
    assert any("Câu 3" in item["text"] and item["page"] == 18 for item in selected)
    assert [item["page"] for item in selected] == sorted(item["page"] for item in selected)
    definition, _ = select_context(pages, "Rủi ro thanh khoản là gì?", "ask")
    assert any("thanh khoản" in item["text"] for item in definition)


def test_explicit_page_and_solve_mode():
    pages = long_pages()
    selected, _ = select_context(pages, "Tóm tắt trang 5", "ask")
    assert any(item["page"] == 5 for item in selected)
    solve, _ = select_context(pages, "", "solve")
    assert [item["page"] for item in solve] == [18] and "Câu 3" in solve[0]["text"]


def test_non_contiguous_chunks_are_marked():
    filler = "\n".join([(FILLER * 9).strip()] * 20)
    pages = [f"Khóa mật mã ở đầu.\n{filler}\nKhóa mật mã ở cuối."]
    selected, excerpted = select_context(pages, "khóa mật mã", "ask")
    assert excerpted and len(selected) == 1 and GAP in selected[0]["text"]
    assert all(normalize(part) in normalize(pages[0]) for part in selected[0]["text"].split(GAP))


def test_excerpt_scope_reaches_provider_and_notice(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-no-live-requests")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_FALLBACK", "false")
    pages = long_pages()
    quote = "Câu 3: Một xe đi 120 km trong 3 giờ."
    answer = Answer(
        items=[
            AnswerItem(
                title="Câu 3",
                status="inference",
                answer="v = 120/3 = 40 km/h",
                citations=[Citation(page=18, quote=quote)],
            )
        ],
        notice="Lời giải đề xuất.",
    )

    async def handler(request):
        content = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert content["pdf_scope"] == "excerpts"
        assert sum(len(item["text"]) for item in content["pdf_pages"]) <= CONTEXT_CHARS
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer.model_dump_json()}}]}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        llm.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    )
    result = asyncio.run(llm.answer_pdf(pages, "Giải câu 3", "ask"))
    assert result.items[0].status == "inference"
    assert result.notice.startswith("PDF dài") and result.notice.endswith("Lời giải đề xuất.")


def test_rejection_keeps_model_notice():
    answer = Answer(
        items=[
            AnswerItem(
                title="X", status="supported", answer="999", citations=[Citation(page=1, quote="Không có trong PDF")]
            )
        ],
        notice="Chỉ giải 12/20 câu.",
    )
    notice = verify_answer(answer, ["Văn bản khác."]).notice
    assert "bị chặn" in notice and "12/20" in notice
