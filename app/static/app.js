"use strict";
const $ = (id) => document.getElementById(id);
let documentInfo = null;
let busy = false;
let lastResult = null;
const labels = {supported: "Có dữ kiện trong PDF", inference: "Lời giải / suy luận", insufficient: "Chưa đủ thông tin", conflict: "Thông tin mâu thuẫn"};

function showError(message = "") { $("error").textContent = message; $("error").hidden = !message; }
function setBusy(value, message = "") {
  busy = value;
  $("activity").hidden = !value;
  $("activity").textContent = message;
  for (const id of ["question", "ask-button", "solve-button"]) $(id).disabled = value || !documentInfo;
  $("pdf-file").disabled = value;
  $("delete-file").disabled = value;
  $("logout").disabled = value;
  for (const button of document.querySelectorAll(".suggestions button")) button.disabled = value || !documentInfo;
}
async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, {...options, credentials: "same-origin"}); }
  catch { throw new Error("Không kết nối được server. Kiểm tra mạng rồi thử lại."); }
  let data;
  try { data = await response.json(); } catch { throw new Error("Server chưa sẵn sàng. Hãy chờ rồi thử lại."); }
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/unlock") { $("unlock-panel").hidden = false; $("workspace").hidden = true; }
    throw new Error(typeof data.detail === "string" ? data.detail : "Dữ liệu không hợp lệ. Hãy kiểm tra lại.");
  }
  return data;
}
function jsonOptions(body) { return {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}; }
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function resetDocument() {
  documentInfo = null; lastResult = null;
  $("pdf-file").value = "";
  $("document-info").hidden = true;
  $("results").replaceChildren(element("p", "empty-state", "Tải một PDF để bắt đầu hỏi đáp."));
  setBusy(false);
}
async function upload(file) {
  if (!file || busy) return;
  showError();
  if (!file.name.toLowerCase().endsWith(".pdf")) { showError("Hãy chọn file .pdf."); return; }
  if (file.size > 10 * 1024 * 1024) { showError("File tối đa 10 MB."); return; }
  setBusy(true, "Đang đọc văn bản PDF…");
  try {
    const previous = documentInfo;
    const body = new FormData(); body.append("file", file);
    documentInfo = await api("/api/documents", {method: "POST", body});
    $("filename").textContent = documentInfo.name;
    $("filemeta").textContent = `${documentInfo.pages} trang · ${documentInfo.characters.toLocaleString("vi-VN")} ký tự`;
    $("warnings").replaceChildren(...documentInfo.warnings.map((text) => element("li", "", text)));
    $("document-info").hidden = false;
    $("results").replaceChildren(element("p", "empty-state", "Tài liệu đã sẵn sàng. Nhập câu hỏi hoặc chọn giải bài tập."));
    lastResult = null;
    if (previous) {
      try { await api(`/api/documents/${encodeURIComponent(previous.id)}`, {method: "DELETE"}); }
      catch { showError("File mới đã tải lên; chưa xóa được file cũ khỏi phiên."); }
    }
  } catch (error) { showError(error.message); }
  finally { setBusy(false); $("pdf-file").value = ""; }
}
async function showSource(docId, citation) {
  showError();
  try {
    const page = await api(`/api/documents/${encodeURIComponent(docId)}/pages/${citation.page}`);
    $("source-title").textContent = `Trang ${citation.page} · văn bản trích xuất`;
    $("source-text").textContent = `TRÍCH DẪN\n${citation.quote}\n\nVĂN BẢN TRANG\n${page.text}`;
    $("source-dialog").showModal();
  } catch (error) { showError(error.message); }
}
function renderAnswer(result, question, docId) {
  if (!lastResult) $("results").replaceChildren();
  const group = element("article", "answer-group");
  group.append(element("p", "asked", question));
  for (const item of result.items) {
    const section = element("section", "answer-item");
    section.append(element("span", `tag ${item.status}`, labels[item.status]));
    section.append(element("h3", "", item.title));
    section.append(element("p", "", item.answer));
    const sources = element("div", "citations");
    for (const citation of item.citations) {
      const button = element("button", "", `Trang ${citation.page} · xem nguồn`);
      button.title = citation.quote;
      button.addEventListener("click", () => showSource(docId, citation));
      sources.append(button);
    }
    section.append(sources); group.append(section);
  }
  if (result.notice) group.append(element("p", "notice", result.notice));
  $("results").prepend(group);
  lastResult = {question, ...result};
  const download = element("button", "quiet", "Tải câu trả lời (.txt)");
  download.addEventListener("click", () => {
    const text = [question, ...result.items.map((item) => `${item.title} [${labels[item.status]}]\n${item.answer}\n${item.citations.map((c) => `Trang ${c.page}: ${c.quote}`).join("\n")}`), result.notice].join("\n\n");
    const link = document.createElement("a");
    const url = URL.createObjectURL(new Blob([text], {type: "text/plain;charset=utf-8"}));
    link.href = url; link.download = "pdf-hoi-dap.txt"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  group.append(download);
}
async function ask(mode) {
  if (busy || !documentInfo) return;
  const question = $("question").value.trim();
  if (mode === "ask" && !question) { showError("Hãy nhập câu hỏi."); return; }
  showError(); setBusy(true, mode === "solve" ? "Đang tìm và giải câu hỏi trong PDF…" : "Đang đọc tài liệu và đối chiếu nguồn…");
  const docId = documentInfo.id;
  try {
    const job = await api("/api/ask", jsonOptions({document_id: docId, question, mode}));
    const deadline = Date.now() + 130000;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const state = await api(`/api/jobs/${encodeURIComponent(job.job_id)}`);
      if (state.status === "error") throw new Error(state.error);
      if (state.status === "done") {
        renderAnswer(state.result, mode === "solve" ? "Giải các câu hỏi trong PDF" : question, docId);
        return;
      }
    }
    throw new Error("Chưa nhận được kết quả. Hãy thử lại sau.");
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}
$("unlock-form").addEventListener("submit", async (event) => {
  event.preventDefault(); showError();
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  try { await api("/api/unlock", jsonOptions({code: $("access-code").value})); $("access-code").value = ""; $("unlock-panel").hidden = true; $("workspace").hidden = false; $("logout").hidden = false; }
  catch (error) { showError(error.message); } finally { button.disabled = false; }
});
$("pdf-file").addEventListener("change", (event) => upload(event.target.files[0]));
const dropzone = $("dropzone");
for (const name of ["dragenter", "dragover"]) dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
for (const name of ["dragleave", "drop"]) dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragover"); });
dropzone.addEventListener("drop", (event) => upload(event.dataTransfer.files[0]));
$("question-form").addEventListener("submit", (event) => { event.preventDefault(); ask("ask"); });
$("question").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); ask("ask"); } });
$("solve-button").addEventListener("click", () => ask("solve"));
for (const button of document.querySelectorAll(".suggestions button")) button.addEventListener("click", () => { $("question").value = button.dataset.question; $("question").focus(); });
$("delete-file").addEventListener("click", async () => {
  if (!documentInfo || busy) return;
  showError(); setBusy(true, "Đang xóa tài liệu…");
  try { await api(`/api/documents/${encodeURIComponent(documentInfo.id)}`, {method: "DELETE"}); resetDocument(); }
  catch (error) { showError(error.message); } finally { setBusy(false); }
});
$("logout").addEventListener("click", async () => {
  showError();
  try { await api("/api/logout", {method: "POST"}); resetDocument(); $("workspace").hidden = true; $("unlock-panel").hidden = false; $("logout").hidden = true; }
  catch (error) { showError(error.message); }
});
$("close-source").addEventListener("click", () => $("source-dialog").close());
api("/api/session").then((session) => {
  $("model").textContent = session.model;
  $("unlock-panel").hidden = session.unlocked;
  $("workspace").hidden = !session.unlocked;
  $("logout").hidden = !session.unlocked;
  setBusy(false);
}).catch((error) => showError(error.message));
