# PDF Hỏi đáp

App: [https://pdf-qa-hic0.onrender.com/](https://pdf-qa-hic0.onrender.com/). Mở trực tiếp, không cần mã truy cập.

Một service Python/FastAPI phục vụ cả frontend HTML/CSS/JS và API. Model chính `openai/gpt-oss-120b` trên Groq (nhóm production, strict structured outputs). Gemini là dự phòng; nhãn production không chứng minh mọi đáp án đúng. Không cần Node, database hay vector store.

## Chạy local

1. Cài Python 3.12 trở lên.
2. Copy `.env.example` thành `.env`, đặt `GROQ_API_KEY`, (tùy chọn `GEMINI_API_KEY` để dự phòng), `SESSION_SECRET` (chuỗi ngẫu nhiên tối thiểu 32 ký tự). Giữ `SECURE_COOKIES=false` khi chạy HTTP local.
3. Chạy `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
4. Mở http://127.0.0.1:8000 và tải PDF.

API key chỉ nằm ở server, không đưa vào frontend hoặc Git. Không chia sẻ `.env`.

## Deploy Render

Một Python Web Service, region Singapore, plan Free. Build `pip install -r requirements.lock`; start `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`. Health check `/api/health`.

Thiết lập Environment theo `render.yaml`: `GROQ_API_KEY`, `GROQ_MODEL=openai/gpt-oss-120b`, `LLM_PROVIDER=groq`, `LLM_FALLBACK=true`, tùy chọn `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.8-flash`, `SESSION_SECRET`, `SECURE_COOKIES=true`, `PYTHON_VERSION=3.12.7`. Có thể dùng Blueprint với repo riêng tư hoặc tạo Web Service qua dashboard.

Không thêm nhiều workers/instances: tài liệu và jobs đang giữ trong RAM của một process. Render Free có thể ngủ khi không dùng; lúc khởi động lại phải tải lại PDF. Không cần persistent disk.

## Phạm vi và kiểm soát câu trả lời

- Upload PDF có văn bản, tối đa 10 MB, 30 trang, 180.000 ký tự; không cắt ngầm phần vượt giới hạn khi upload.
- Hỏi về tài liệu hoặc tìm/giải tối đa 12 câu hỏi đầu tiên. Hỏi câu cụ thể để giải tiếp phần còn lại.
- PDF đến 10.000 ký tự: gửi toàn bộ văn bản mỗi lượt. PDF dài hơn: RAG từ khóa (`app/retrieval.py`), chia chunk theo trang/dòng (tối đa 800 ký tự), xếp hạng BM25 không phân biệt dấu, gửi các chunk liên quan nhất trong ngân sách 10.000 ký tự; câu hỏi có "trang N" ưu tiên trang đó; chế độ giải bài lấy các đoạn chứa "Câu/Bài + số" theo thứ tự tài liệu. Ngân sách đo theo Groq free tier (8.000 token/phút/request, tiếng Việt khoảng 0,32 token/ký tự). Khi chỉ gửi đoạn trích, notice ghi trang đã gửi và câu trả lời thiếu dữ kiện chỉ nói các đoạn đã truy xuất chưa chứa thông tin. Câu hỏi tóm tắt toàn bộ hoặc diễn đạt khác từ ngữ tài liệu có thể bỏ sót trên PDF dài.
- Cache câu trả lời trong RAM theo nội dung PDF + chế độ + câu hỏi (tối đa 200 mục, 2 giờ); hỏi lại cùng câu không gọi provider và không tính vào giới hạn lượt hỏi.
- System prompt: `app/system_prompt.txt`; bộ 11 ví dụ chuẩn: `app/common_examples.json`.
- JSON schema tách dữ kiện, suy luận, thiếu thông tin và mâu thuẫn. Backend kiểm tra số trang và nguyên văn trích dẫn; nếu không khớp, chặn cả phần trả lời liên quan.
- Groq nhận JSON schema strict; backend tiếp tục kiểm tra kiểu dữ liệu, giới hạn độ dài, trang/nguyên văn nguồn, trích dẫn trùng và số liệu mới trong câu trả lời supported. Phép tính phải gắn inference; không kiểm chứng tự động tính đúng của mọi phép tính hoặc quan hệ ngữ nghĩa.
- Thiếu dữ kiện: chặn đáp án, liệt kê phần thiếu và hỏi lại. Nút Bổ sung thông tin giữ câu hỏi gốc; lời bổ sung chỉ xác định phạm vi/giả định được cho phép, không trở thành nguồn PDF.
- Mỗi lượt tối đa một request mỗi provider; chỉ fallback khi timeout/kết nối/429/500/502/503/504, không lặp lại nếu JSON hoặc kiểm tra nguồn thất bại. Có thể tắt fallback bằng LLM_FALLBACK=false. Giới hạn output 4096 token; output bị cắt sẽ bị chặn, cần hỏi ít câu hơn.
- Kiểm tra trích dẫn chỉ chứng minh đoạn trích tồn tại, không chứng minh nó hỗ trợ đúng kết luận. Prompt không bảo đảm loại bỏ hoàn toàn bịa/sai hoặc prompt injection.
- Bài tập có thể dùng phương pháp phổ thông, luôn gắn nhãn suy luận/lời giải. Không tự nhận là đáp án chính thức.
- PDF scan cần OCR trước; hình/biểu đồ chưa phân tích, bảng phức tạp có thể mất bố cục khi trích văn bản. Trang nguồn là số trang vật lý, bắt đầu từ 1.
- Phiên riêng qua cookie ký, nguồn request cùng origin. Tối đa 60 lượt hỏi/giờ toàn app, 4 lượt chạy đồng thời, mỗi phiên 1 lượt; 3 PDF/phiên, 30 PDF/server. Nội dung giữ tối đa 2 giờ (dọn khi có request), mất khi restart.
- Đây là app công khai dùng phiên ẩn danh; không có tài khoản cá nhân, RBAC hay cam kết bảo mật doanh nghiệp. Nội dung gửi tới Groq/Gemini chịu điều khoản của nhà cung cấp tương ứng.

## Kiểm tra

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q app tests
.venv\Scripts\ruff.exe check app tests
node --check app/static/app.js
```

Tests dùng provider giả để kiểm tra backend và kiểm soát nguồn, không thay thế thử nghiệm LLM thật hoặc QA trên Render.
