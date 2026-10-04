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
    """Conservative checks of sources and literal numbers, not semantic entailment."""
    rejected = 0
    for item in answer.items:
        unique = {(c.page, normalize(c.quote)): c for c in item.citations}
        item.citations = list(unique.values())
        bad_quote = any(
            citation.page > len(pages) or normalize(citation.quote) not in normalize(pages[citation.page - 1])
            for citation in item.citations
        )
        missing = item.status != "insufficient" and not item.citations
        conflict_missing = item.status == "conflict" and len(item.citations) < 2
        quotes = " ".join(c.quote for c in item.citations)

        def numbers(text):
            return set(re.findall(r"[+-]?\d+(?:[.,]\d+)*", normalize(text)))

        unsupported_number = item.status == "supported" and not numbers(item.answer).issubset(numbers(quotes))
        unsupported_url = any(url not in quotes for url in re.findall(r"https?://[^\s<>]+", item.answer))
        unlabelled_assumptions = bool(item.assumptions) and item.status != "inference"
        empty_answer = item.status != "insufficient" and not item.answer.strip()
        if (
            bad_quote
            or missing
            or conflict_missing
            or unsupported_number
            or unsupported_url
            or unlabelled_assumptions
            or empty_answer
        ):
            rejected += 1
            item.status = "insufficient"
            item.citations = []
            item.missing_information = ["Dẫn chứng hợp lệ hỗ trợ kết luận"]
            item.clarifying_question = "Bạn có thể chỉ rõ câu hoặc trang chứa dữ kiện cần dùng không?"
        if item.missing_information or item.clarifying_question:
            item.status = "insufficient"
        if item.status == "insufficient":
            item.answer = "Chưa đủ thông tin để kết luận. Cần làm rõ hoặc bổ sung dữ kiện trước khi trả lời."
            item.assumptions = []
            if not item.missing_information:
                item.missing_information = ["Dữ kiện cần thiết trong tài liệu hoặc phạm vi câu hỏi"]
            if not item.clarifying_question:
                item.clarifying_question = "Bạn có thể chỉ rõ câu/trang cần hỏi hoặc bổ sung dữ kiện còn thiếu không?"
    if rejected:
        answer.notice = "Một số phần bị chặn vì nguồn, số liệu hoặc giả định chưa được xác minh."
    return answer
