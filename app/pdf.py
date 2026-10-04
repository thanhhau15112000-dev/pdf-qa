import io
import re
import unicodedata

from pypdf import PdfReader

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 80
MAX_CHARS = 180_000


class InvalidPDF(ValueError):
    pass


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def extract_pdf(data: bytes) -> tuple[list[str], list[str]]:
    if len(data) > MAX_BYTES:
        raise InvalidPDF("PDF tối đa 10 MB.")
    if not data.startswith(b"%PDF-"):
        raise InvalidPDF("File không có định dạng PDF hợp lệ.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPDF("PDF có mật khẩu. Hãy tải bản đã bỏ mật khẩu.")
        if not 1 <= len(reader.pages) <= MAX_PAGES:
            raise InvalidPDF("PDF cần có 1–80 trang. Hãy chia file lớn thành phần nhỏ.")
        pages = []
        total = 0
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            total += len(text)
            if total > MAX_CHARS:
                raise InvalidPDF("PDF có quá nhiều văn bản. Hãy chia thành file nhỏ hơn.")
            pages.append(text)
    except InvalidPDF:
        raise
    except Exception as exc:
        raise InvalidPDF("Không đọc được PDF. File có thể bị hỏng.") from exc
    if not any(normalize(page) for page in pages):
        raise InvalidPDF("PDF không có văn bản đọc được. Hãy OCR file scan trước khi tải lên.")
    warnings = ["Chỉ đọc lớp văn bản; hình ảnh và biểu đồ chưa được phân tích."]
    empty = [str(i + 1) for i, text in enumerate(pages) if not normalize(text)]
    if empty:
        warnings.append("Không đọc được văn bản ở trang: " + ", ".join(empty) + ".")
    return pages, warnings


def verify_answer(answer, pages: list[str]):
    """Reject an entire item if any claimed quote/page is invalid; never hide bad sources."""
    rejected = 0
    for item in answer.items:
        bad_quote = any(
            citation.page > len(pages) or normalize(citation.quote) not in normalize(pages[citation.page - 1])
            for citation in item.citations
        )
        missing = item.status != "insufficient" and not item.citations
        conflict_missing = item.status == "conflict" and len(item.citations) < 2
        if bad_quote or missing or conflict_missing:
            rejected += 1
            item.status = "insufficient"
            item.answer = (
                "Không thể xác minh nguồn cho phần trả lời này. Hãy hỏi lại cụ thể hơn hoặc kiểm tra trang liên quan."
            )
            item.citations = []
    if rejected:
        answer.notice = "Một số phần bị chặn vì thiếu nguồn hoặc trích dẫn không khớp PDF."
    return answer
