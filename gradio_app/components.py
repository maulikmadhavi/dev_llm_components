"""Node registry shared by the Gradio pipeline builder and the React Flow canvas (served via /api).

Each node is a spec (ports + params, rendered generically by both UIs) plus a run function
`fn(params, inputs) -> outputs`. Port types: text, image (data URL), video (dict), chunks (list).
"""
from __future__ import annotations

import base64
import csv
import math
import re
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Files that reader nodes have listed from the user's own folders/CSVs. Later nodes (VLM, chunker)
# and /api/media may open these paths; anything else outside UPLOAD_DIR is refused.
_KNOWN_MEDIA: set[Path] = set()


class NodeError(Exception):
    """User-facing failure; surfaced as an HTTP 400 / Gradio message."""


# ---------- helpers ----------

def save_upload(src, name: str) -> Path:
    """Copy a path or file object into UPLOAD_DIR under a unique name."""
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}_{Path(name).name}"
    if isinstance(src, (str, Path)):
        shutil.copy2(src, dest)
    else:
        with open(dest, "wb") as f:
            shutil.copyfileobj(src, f)
    return dest


def upload_path(p) -> Path:
    """Resolve a client-supplied path, only allowing files inside UPLOAD_DIR."""
    if not p:
        raise NodeError("Upload a file first.")
    path = Path(p).resolve()
    if UPLOAD_DIR.resolve() not in path.parents or not path.is_file():
        raise NodeError("File not found. Please upload it again.")
    return path


def media_path(p) -> Path:
    """Like upload_path, but also accepts files a reader node listed earlier."""
    if not p:
        raise NodeError("Missing file path.")
    path = Path(p).resolve()
    if path in _KNOWN_MEDIA and path.is_file():
        return path
    return upload_path(p)


def _local_file(raw: str, base_dir: str = "") -> Path:
    """Resolve a user-typed path (optionally relative to base_dir) to an existing file."""
    path = Path(raw.strip())
    if not path.is_absolute() and base_dir:
        path = Path(base_dir.strip()) / path
    path = path.resolve()
    if not path.is_file():
        raise NodeError(f"File not found: {raw}")
    return path


def _folder_files(params: dict, exts: set[str]) -> list[Path]:
    folder = Path((params.get("folder") or "").strip())
    if not folder.is_dir():
        raise NodeError(f"Folder not found: {folder}")
    limit = max(int(params.get("max_items") or 50), 1)
    pattern = (params.get("pattern") or "*").strip() or "*"
    files = sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise NodeError(f"No matching files in {folder}")
    return files[:limit]


def _csv_rows(params: dict, column: str) -> list[tuple[Path, dict]]:
    """(file, remaining columns) for each CSV row, with the file read from `column`."""
    csv_path = upload_path(params.get("file"))
    column = (params.get("column") or column).strip()
    limit = max(int(params.get("max_items") or 50), 1)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise NodeError(f"Column '{column}' not found. CSV columns: {', '.join(reader.fieldnames or [])}")
        rows = []
        for row in reader:
            if not (row.get(column) or "").strip():
                continue
            meta = {k: v for k, v in row.items() if k != column and k is not None}
            rows.append((_local_file(row[column], params.get("base_dir") or ""), meta))
            if len(rows) >= limit:
                break
    if not rows:
        raise NodeError("The CSV has no rows with a file path.")
    return rows


def _image_item(path: Path, meta: dict | None = None) -> dict:
    if path.suffix.lower() not in IMAGE_MIME:
        raise NodeError(f"Unsupported image type: {path.name}")
    _KNOWN_MEDIA.add(path)
    return {"name": path.name, "path": str(path), "meta": meta or {}}


def _video_item(path: Path, meta: dict | None = None) -> dict:
    if path.suffix.lower() not in VIDEO_EXTS:
        raise NodeError(f"Unsupported video type: {path.name}")
    cap, fps, duration = _open_video(path)
    try:
        info = {
            "path": str(path),
            "name": path.name,
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "meta": meta or {},
        }
    finally:
        cap.release()
    _KNOWN_MEDIA.add(path)
    return info


def image_data_url(path: Path, max_side: int = 1024) -> str:
    """Load an image from disk and return a downscaled JPEG data URL."""
    path = media_path(path)
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise NodeError(f"{path.name} is larger than 10 MB.")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise NodeError(f"Could not decode image: {path.name}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def _image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def chat(params: dict, content) -> str:
    """One chat-completion call against an OpenAI-compatible server (LM Studio by default)."""
    messages = []
    if params.get("system"):
        messages.append({"role": "system", "content": params["system"]})
    messages.append({"role": "user", "content": content})
    headers = {"Authorization": f"Bearer {params['api_key']}"} if params.get("api_key") else {}
    try:
        r = requests.post(
            f"{params['base_url'].rstrip('/')}/chat/completions",
            json={
                "model": params["model"],
                "messages": messages,
                "temperature": float(params.get("temperature") or 0),
                "max_tokens": int(params.get("max_tokens") or 512),
            },
            headers=headers,
            timeout=600,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = choice["message"]["content"] or ""
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        raise NodeError(f"Model server error: {e}") from e
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        hint = " The model ran out of tokens while thinking; raise Max tokens." if choice.get("finish_reason") == "length" else ""
        raise NodeError(f"Model returned an empty reply.{hint}")
    return text


def _open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if not cap.isOpened() or fps <= 0 or n <= 0:
        cap.release()
        raise NodeError("Could not read this video file.")
    return cap, fps, n / fps


def _frame_url(cap, fps: float, t: float, max_side: int = 768) -> str | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ---------- node implementations ----------

def run_text_input(params, inputs):
    return {"text": params.get("text", "")}


def run_image_input(params, inputs):
    path = upload_path(params.get("file"))
    mime = IMAGE_MIME.get(path.suffix.lower())
    if not mime:
        raise NodeError(f"Unsupported image type: {path.suffix}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise NodeError("Image is larger than 10 MB.")
    return {"image": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()}


def run_file_reader(params, inputs):
    path = upload_path(params.get("file"))
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    limit = int(params.get("max_chars") or 20000)
    return {"text": text[:limit]}


def run_video_reader(params, inputs):
    path = upload_path(params.get("file"))
    cap, fps, duration = _open_video(path)
    try:
        video = {
            "path": str(path),
            "name": path.name.split("_", 1)[-1],
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return {"video": video, "frame": _frame_url(cap, fps, duration / 2)}
    finally:
        cap.release()


def chunk_video(path: Path, size: float, per_chunk: int, max_chunks: int) -> list[dict]:
    """Split a video into `size`-second chunks, sampling `per_chunk` frames from each."""
    cap, fps, duration = _open_video(path)
    try:
        chunks = []
        for i in range(min(math.ceil(duration / size), max_chunks)):
            start, end = i * size, min(duration, (i + 1) * size)
            times = [start + (j + 0.5) * (end - start) / per_chunk for j in range(per_chunk)]
            frames = [u for u in (_frame_url(cap, fps, t) for t in times) if u]
            chunks.append({"index": i, "start": round(start, 2), "end": round(end, 2), "frames": frames})
        return chunks
    finally:
        cap.release()


def run_video_chunker(params, inputs):
    chunks = chunk_video(
        media_path(inputs["video"].get("path")),
        max(float(params.get("chunk_seconds") or 10), 0.5),
        max(int(params.get("frames_per_chunk") or 1), 1),
        max(int(params.get("max_chunks") or 20), 1),
    )
    return {"chunks": chunks}


def run_image_folder_reader(params, inputs):
    return {"images": [_image_item(p) for p in _folder_files(params, set(IMAGE_MIME))]}


def run_video_folder_reader(params, inputs):
    return {"videos": [_video_item(p) for p in _folder_files(params, VIDEO_EXTS)]}


def run_csv_reader_image(params, inputs):
    return {"images": [_image_item(p, meta) for p, meta in _csv_rows(params, "image_path")]}


def run_csv_reader_video(params, inputs):
    return {"videos": [_video_item(p, meta) for p, meta in _csv_rows(params, "video_path")]}


def _describe_chunks(params, prompt: str, chunks: list[dict]) -> list[dict]:
    segments = []
    for c in chunks:
        span = f"{fmt_time(c['start'])}–{fmt_time(c['end'])}"
        content = [{"type": "text", "text": f"{prompt}\n(Video segment {span})"}] + [_image_part(f) for f in c["frames"]]
        segments.append({"start": c["start"], "end": c["end"], "text": chat(params, content)})
    return segments


def _segment_text(segments: list[dict]) -> str:
    return "\n".join(f"[{fmt_time(s['start'])}–{fmt_time(s['end'])}] {s['text']}" for s in segments)


def run_vlm(params, inputs):
    """Describe an image, video chunks, or a whole list of images/videos.

    Always returns `text` (readable summary) and `results` (structured, for CSV writers).
    """
    prompt = (params.get("prompt") or "").strip()
    if not prompt:
        raise NodeError("Enter a prompt.")
    if inputs.get("context"):
        prompt += f"\n\nContext:\n{inputs['context']}"

    results = []
    if inputs.get("videos"):
        size = max(float(params.get("chunk_seconds") or 10), 0.5)
        per_chunk = max(int(params.get("frames_per_chunk") or 1), 1)
        for v in inputs["videos"]:
            segments = _describe_chunks(params, prompt, chunk_video(media_path(v["path"]), size, per_chunk, 20))
            results.append({"name": v["name"], "path": v["path"], "segments": segments,
                            "text": _segment_text(segments), "meta": v.get("meta", {})})
    elif inputs.get("images"):
        for item in inputs["images"]:
            text = chat(params, [{"type": "text", "text": prompt}, _image_part(image_data_url(item["path"]))])
            results.append({"name": item["name"], "path": item["path"], "text": text, "meta": item.get("meta", {})})
    elif inputs.get("chunks"):
        segments = _describe_chunks(params, prompt, inputs["chunks"])
        results.append({"name": "video", "path": "", "segments": segments, "text": _segment_text(segments), "meta": {}})
    elif inputs.get("image"):
        text = chat(params, [{"type": "text", "text": prompt}, _image_part(inputs["image"])])
        results.append({"name": "image", "path": "", "text": text, "meta": {}})
    else:
        raise NodeError("Connect an image, images, video chunks or videos.")

    batch = bool(inputs.get("videos") or inputs.get("images"))
    text = "\n\n".join(f"{r['name']}:\n{r['text']}" for r in results) if batch else results[0]["text"]
    return {"text": text, "results": results}


def _write_csv(filename: str, default: str, header: list[str], rows: list[dict]) -> dict:
    name = Path((filename or "").strip() or default).name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    extra = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in header]
    with open(OUTPUT_DIR / name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {"file": {"name": name, "rows": len(rows)}}


def run_csv_writer_image(params, inputs):
    rows = [{"image": r.get("path", ""), "name": r["name"], "text": r["text"], **r.get("meta", {})} for r in inputs["results"]]
    return _write_csv(params.get("filename"), "image_results.csv", ["image", "name", "text"], rows)


def run_csv_writer_video(params, inputs):
    rows = []
    for r in inputs["results"]:
        base = {"video": r.get("path", ""), "name": r["name"], **r.get("meta", {})}
        for s in r.get("segments") or [{"start": "", "end": "", "text": r["text"]}]:
            rows.append({**base, "start": s["start"], "end": s["end"], "text": s["text"]})
    return _write_csv(params.get("filename"), "video_results.csv", ["video", "name", "start", "end", "text"], rows)


def run_llm(params, inputs):
    instruction = (params.get("instruction") or "").strip()
    return {"text": chat(params, f"{instruction}\n\n{inputs['text']}".strip())}


# ---------- registry ----------

def P(name, label, kind="text", default="", **extra):
    return {"name": name, "label": label, "kind": kind, "default": default, **extra}


def model_params(model: str) -> list[dict]:
    """Settings that make a VLM/LLM node 'custom': any OpenAI-compatible endpoint and model."""
    return [
        P("base_url", "Base URL", default="http://localhost:1234/v1", advanced=True),
        P("model", "Model", default=model, advanced=True),
        P("system", "System prompt", "textarea", "", advanced=True),
        P("temperature", "Temperature", "number", 0.2, advanced=True, step=0.1),
        P("max_tokens", "Max tokens", "number", 2048, advanced=True),
        P("api_key", "API key (optional)", "password", "", advanced=True),
    ]


def port(name, type_, optional=False):
    return {"name": name, "type": type_, "optional": optional}


NODES = {
    "text_input": dict(
        label="Text Input", icon="📝", category="Sources", inputs=[], outputs=[port("text", "text")],
        params=[P("text", "Text", "textarea")], run=run_text_input),
    "image_input": dict(
        label="Image Input", icon="🖼️", category="Sources", inputs=[], outputs=[port("image", "image")],
        params=[P("file", "Image", "file", None, accept="image/*")], run=run_image_input),
    "file_reader": dict(
        label="File Reader", icon="📄", category="Sources", inputs=[], outputs=[port("text", "text")],
        params=[P("file", "File (txt, md, csv, pdf, ...)", "file", None, accept=".txt,.md,.csv,.json,.py,.pdf"),
                P("max_chars", "Max characters", "number", 20000)], run=run_file_reader),
    "video_reader": dict(
        label="Video Reader", icon="🎞️", category="Sources", inputs=[],
        outputs=[port("video", "video"), port("frame", "image")],
        params=[P("file", "Video", "file", None, accept="video/*")], run=run_video_reader),
    "video_chunker": dict(
        label="Video Chunker", icon="✂️", category="Processing", inputs=[port("video", "video")],
        outputs=[port("chunks", "chunks")],
        params=[P("chunk_seconds", "Chunk length (s)", "number", 10),
                P("frames_per_chunk", "Frames per chunk", "number", 1),
                P("max_chunks", "Max chunks", "number", 20)], run=run_video_chunker),
    "image_folder_reader": dict(
        label="Image Folder Reader", icon="🗂️", category="Sources", inputs=[], outputs=[port("images", "images")],
        params=[P("folder", "Folder path (on the server)"), P("pattern", "Pattern", default="*"),
                P("max_items", "Max images", "number", 50)], run=run_image_folder_reader),
    "video_folder_reader": dict(
        label="Video Folder Reader", icon="📁", category="Sources", inputs=[], outputs=[port("videos", "videos")],
        params=[P("folder", "Folder path (on the server)"), P("pattern", "Pattern", default="*"),
                P("max_items", "Max videos", "number", 20)], run=run_video_folder_reader),
    "csv_reader_image": dict(
        label="CSV Reader (Image)", icon="🧾", category="Sources", inputs=[], outputs=[port("images", "images")],
        params=[P("file", "CSV file", "file", None, accept=".csv"), P("column", "Image path column", default="image_path"),
                P("base_dir", "Base folder for relative paths (optional)"), P("max_items", "Max rows", "number", 50)],
        run=run_csv_reader_image),
    "csv_reader_video": dict(
        label="CSV Reader (Video)", icon="📋", category="Sources", inputs=[], outputs=[port("videos", "videos")],
        params=[P("file", "CSV file", "file", None, accept=".csv"), P("column", "Video path column", default="video_path"),
                P("base_dir", "Base folder for relative paths (optional)"), P("max_items", "Max rows", "number", 20)],
        run=run_csv_reader_video),
    "vlm": dict(
        label="VLM", icon="👁️", category="Models",
        inputs=[port("image", "image", True), port("chunks", "chunks", True), port("images", "images", True),
                port("videos", "videos", True), port("context", "text", True)],
        outputs=[port("text", "text"), port("results", "results")],
        params=[P("prompt", "Prompt", "textarea", "Describe this image."),
                P("chunk_seconds", "Video: chunk length (s)", "number", 10),
                P("frames_per_chunk", "Video: frames per chunk", "number", 1)] + model_params("qwen3-vl-4b"),
        run=run_vlm),
    "csv_writer_image": dict(
        label="CSV Writer (Image)", icon="💾", category="Outputs", inputs=[port("results", "results")],
        outputs=[port("file", "file")],
        params=[P("filename", "File name", default="image_results.csv")], run=run_csv_writer_image),
    "csv_writer_video": dict(
        label="CSV Writer (Video)", icon="📼", category="Outputs", inputs=[port("results", "results")],
        outputs=[port("file", "file")],
        params=[P("filename", "File name", default="video_results.csv")], run=run_csv_writer_video),
    "llm": dict(
        label="LLM", icon="💬", category="Models", inputs=[port("text", "text")], outputs=[port("text", "text")],
        params=[P("instruction", "Instruction", "textarea", "Summarize in one sentence:")] + model_params("qwen/qwen3-4b"),
        run=run_llm),
}


def specs() -> list[dict]:
    return [{"type": t, **{k: v for k, v in n.items() if k != "run"}} for t, n in NODES.items()]


def run_node(type_: str, params: dict, inputs: dict) -> dict:
    node = NODES.get(type_)
    if node is None:
        raise NodeError(f"Unknown node type: {type_}")
    for p in node["inputs"]:
        if not p["optional"] and inputs.get(p["name"]) in (None, ""):
            raise NodeError(f"Missing input '{p['name']}'. Connect an upstream node and run it first.")
    return node["run"](params, inputs)
