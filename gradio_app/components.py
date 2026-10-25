"""Node registry shared by the Gradio pipeline builder and the React Flow canvas (served via /api).

Each node is a spec (ports + params, rendered generically by both UIs) plus a run function
`fn(params, inputs) -> outputs`. Port types: text, image (data URL), video (dict), chunks (list).
"""
from __future__ import annotations

import base64
import csv
import json
import math
import re
import shutil
import uuid
from collections import Counter
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


def _decode_data_url(url: str):
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)


def _load_bgr(item: dict):
    """Full-resolution pixels for a list item ({path}) or a single-image item ({image: data URL})."""
    if item.get("path"):
        path = media_path(item["path"])
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    elif item.get("image"):
        img = _decode_data_url(item["image"])
    else:
        raise NodeError("Item has no image.")
    if img is None:
        raise NodeError(f"Could not decode image: {item.get('name', '')}")
    return img


def _json_lists(text: str):
    """Yield each JSON list of objects found in `text` (tolerates prose, <think> blocks, several lists)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            val, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(val, list) and all(isinstance(v, dict) for v in val):
            yield val


def _box_item(d: dict, scale: float = 1000) -> dict | None:
    """Normalise one {bbox_2d|bbox|box, label, score} dict to 0-1 coordinates; None if unusable."""
    coords = d.get("bbox_2d") or d.get("bbox") or d.get("box")
    try:
        x1, y1, x2, y2 = (min(max(float(v) / scale, 0.0), 1.0) for v in coords)
        score = d.get("score", d.get("confidence"))
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return {"label": str(d.get("label") or d.get("name") or "object"), "score": score,
            "box": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]}


def _parse_boxes(text: str, max_boxes: int) -> list[dict]:
    """Pull [{bbox_2d:[x1,y1,x2,y2], label}] (Qwen-VL style, 0-1000 coordinates) out of a model reply."""
    raw = next(_json_lists(text), None)
    if raw is None:  # e.g. a reply cut off by max_tokens: salvage the complete objects
        dec, raw = json.JSONDecoder(), []
        for m in re.finditer(r"\{", text):
            try:
                val, _ = dec.raw_decode(text[m.start():])
            except ValueError:
                continue
            if isinstance(val, dict) and any(k in val for k in ("bbox_2d", "bbox", "box")):
                raw.append(val)
        if not raw:
            raise NodeError(f"The model did not return a JSON list of boxes: {text[:200]}")
    return [b for b in map(_box_item, raw) if b][:max_boxes]


def run_object_detector(params, inputs):
    """Ask a VLM for boxes. `detections` items are {name, path|image, boxes:[{label, score, box (0-1 x1,y1,x2,y2)}], meta}."""
    if inputs.get("images"):
        items = [{"name": i["name"], "path": i["path"], "meta": i.get("meta", {})} for i in inputs["images"]]
    elif inputs.get("image"):
        items = [{"name": "image", "image": inputs["image"], "meta": {}}]
    else:
        raise NodeError("Connect an image or images.")
    labels = (params.get("labels") or "").strip()
    target = f"every instance of: {labels}" if labels else "all distinct objects"
    prompt = (f"Detect {target}. Reply with only a JSON list like "
              '[{"bbox_2d": [x1, y1, x2, y2], "label": "name"}] using integer coordinates from 0 to 1000 '
              "relative to the image width and height. Reply [] if nothing is found.")
    max_boxes = max(int(params.get("max_boxes") or 20), 1)
    detections, lines = [], []
    for it in items:
        url = image_data_url(it["path"]) if it.get("path") else it["image"]
        boxes = _parse_boxes(chat(params, [{"type": "text", "text": prompt}, _image_part(url)]), max_boxes)
        detections.append({**it, "boxes": boxes})
        lines.append(f"{it['name']}: " + (", ".join(b["label"] for b in boxes) or "no objects"))
    return {"detections": detections, "text": "\n".join(lines)}


def run_image_crop(params, inputs):
    """Crop each detected box (loose adds a margin around each box), or one fixed fractional region per image when nothing is detected."""
    if inputs.get("detections"):
        wanted = {s.strip().lower() for s in (params.get("labels") or "").split(",") if s.strip()}
        min_score = float(params.get("min_score") or 0)
        try:
            margin = float(params.get("margin") if params.get("margin") not in (None, "") else 20)
        except (TypeError, ValueError) as e:
            raise NodeError("Margin must be a number (percent).") from e
        pad = max(margin, 0.0) / 100 if (params.get("mode") or "loose") == "loose" else 0.0  # each side, relative to box size
        regions = []
        for d in inputs["detections"]:
            d.setdefault("name", "image")
            for raw_box in d.get("boxes") or []:
                b = _box_item({**raw_box, "box": raw_box.get("box")}, scale=1)
                if b and (not wanted or b["label"].lower() in wanted) and (b["score"] is None or b["score"] >= min_score):
                    regions.append((d, b))
        if not regions:
            raise NodeError("No boxes matched the label/score filter.")
    else:
        if inputs.get("images"):
            sources = inputs["images"]
        elif inputs.get("image"):
            sources = [{"name": "image", "image": inputs["image"], "meta": {}}]
        else:
            raise NodeError("Connect detections, an image or images.")
        try:
            x, y, w, h = (min(max(float(params.get(k) if params.get(k) not in (None, "") else d), 0.0), 1.0)
                          for k, d in (("x", 0.25), ("y", 0.25), ("w", 0.5), ("h", 0.5)))
        except (TypeError, ValueError) as e:
            raise NodeError("Crop x, y, w and h must be numbers.") from e
        if w <= 0 or h <= 0:
            raise NodeError("Crop width and height must be above 0.")
        pad = 0.0
        regions = [(s, {"label": "crop", "score": None, "box": [x, y, min(x + w, 1.0), min(y + h, 1.0)]}) for s in sources]

    max_crops = 200
    if len(regions) > max_crops:
        regions = regions[:max_crops]
    cache: dict[str, object] = {}
    crops = []
    for i, (src, b) in enumerate(regions):
        key = src.get("path") or src["name"]
        if key not in cache:
            cache[key] = _load_bgr(src)
        img = cache[key]
        H, W = img.shape[:2]
        x1, y1, x2, y2 = b["box"]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        l, t = int(max(x1 - px, 0) * W), int(max(y1 - py, 0) * H)
        r, btm = int(math.ceil(min(x2 + px, 1) * W)), int(math.ceil(min(y2 + py, 1) * H))
        if r - l < 2 or btm - t < 2:
            continue
        ok, buf = cv2.imencode(".jpg", img[t:btm, l:r], [cv2.IMWRITE_JPEG_QUALITY, 92])
        dest = UPLOAD_DIR / f"crop_{uuid.uuid4().hex[:10]}.jpg"
        dest.write_bytes(buf.tobytes())
        stem = Path(src["name"]).stem
        label = re.sub(r"\W+", "_", b["label"]).strip("_") or "crop"
        meta = {**src.get("meta", {}), "source": src["name"], "label": b["label"]}
        if b.get("score") is not None:
            meta["score"] = b["score"]
        item = _image_item(dest, meta)
        item["name"] = f"{stem}_{label}_{i}.jpg"
        crops.append(item)
    if not crops:
        raise NodeError("All crops were empty.")
    return {"images": crops, "image": image_data_url(crops[0]["path"])}


def _norm_answer(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def _fuse_one(params: dict, candidates: list[str], image_url: str | None) -> tuple[str, float]:
    """Combine candidate answers. Returns (decision, share of candidates that agree with it)."""
    method = params.get("method") or "judge"
    counts = Counter(_norm_answer(c) for c in candidates)
    top, n = counts.most_common(1)[0]
    if method == "first":
        decision = candidates[0]
    elif method == "majority":
        decision = next(c for c in candidates if _norm_answer(c) == top) if n > 1 else candidates[0]
    elif method == "judge":
        listing = "\n\n".join(f"Candidate {i + 1}:\n{c}" for i, c in enumerate(candidates))
        task = (params.get("instruction") or "").strip()
        look = " Use the image to check which candidate is correct." if image_url else ""
        prompt = (f"{task}\n\nSeveral models answered the same task.{look} Decide the final answer.\n\n{listing}\n\n"
                  "Write the final answer text itself (copy the best candidate or merge them). "
                  "Do not mention candidate numbers and add no commentary.").strip()
        decision = chat(params, [{"type": "text", "text": prompt}, _image_part(image_url)] if image_url else prompt)
        # Small models sometimes reply "Candidate 2" anyway: map that back to the candidate's text.
        pick = re.fullmatch(r"(?:candidate\s*)?(\d+)\W*", decision.strip(), re.I)
        if pick and 1 <= int(pick.group(1)) <= len(candidates):
            decision = candidates[int(pick.group(1)) - 1]
        elif _norm_answer(decision) not in counts:
            return decision, None  # rewritten answer: agreement is undefined
    else:
        raise NodeError(f"Unknown fusion method: {method}")
    return decision, round(counts[_norm_answer(decision)] / len(candidates), 3)


def _fuse(params: dict, lists: list[list[dict]], images: list[str | None] | None = None) -> dict:
    """Fuse aligned result lists item by item (aligned by position)."""
    lists = [l for l in lists if l]
    if len(lists) < 2:
        raise NodeError("Connect at least two candidate inputs.")
    if len({len(l) for l in lists}) > 1:
        raise NodeError("Candidate lists have different lengths: " + ", ".join(str(len(l)) for l in lists))
    results = []
    for i, group in enumerate(zip(*lists)):
        url = images[i] if images and i < len(images) else None
        decision, agreement = _fuse_one(params, [g.get("text") or "" for g in group], url)
        results.append({"name": group[0].get("name") or f"item_{i + 1}", "path": group[0].get("path", ""), "text": decision,
                        "meta": {**group[0].get("meta", {}), "agreement": agreement, "candidates": len(group)}})
    text = results[0]["text"] if len(results) == 1 else "\n\n".join(f"{r['name']}:\n{r['text']}" for r in results)
    return {"text": text, "results": results}


def run_llm_decision_fusion(params, inputs):
    lists = [[{"name": "text", "path": "", "text": inputs[k], "meta": {}}] for k in ("a", "b", "c") if inputs.get(k)]
    lists += [inputs[k] for k in ("results_a", "results_b", "results_c") if inputs.get(k)]
    return _fuse(params, lists)


def run_vlm_decision_fusion(params, inputs):
    lists = [inputs[k] for k in ("results_a", "results_b", "results_c") if inputs.get(k)]
    if inputs.get("images"):
        urls = [image_data_url(i["path"]) for i in inputs["images"]] if (params.get("method") or "judge") == "judge" else None
    else:
        urls = [inputs["image"]] if inputs.get("image") else None
    return _fuse(params, lists, urls)


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
    "object_detector": dict(
        label="Object Detector", icon="🎯", category="Models",
        inputs=[port("image", "image", True), port("images", "images", True)],
        outputs=[port("detections", "detections"), port("text", "text")],
        params=[P("labels", "Objects to find (comma-separated, blank = all)"), P("max_boxes", "Max boxes per image", "number", 20)]
        + model_params("qwen3-vl-4b"),
        run=run_object_detector),
    "image_crop": dict(
        label="Image Crop", icon="🔲", category="Processing",
        inputs=[port("detections", "detections", True), port("image", "image", True), port("images", "images", True)],
        outputs=[port("images", "images"), port("image", "image")],
        params=[P("labels", "Only these labels (comma-separated, blank = all)"), P("min_score", "Min score", "number", 0, step=0.05),
                P("mode", "Crop mode", "select", "loose", options=["loose", "tight"]),
                P("margin", "Loose margin (% of box size, each side)", "number", 20, step=5),
                P("x", "No detections: left (0-1)", "number", 0.25, step=0.05), P("y", "No detections: top (0-1)", "number", 0.25, step=0.05),
                P("w", "No detections: width (0-1)", "number", 0.5, step=0.05), P("h", "No detections: height (0-1)", "number", 0.5, step=0.05)],
        run=run_image_crop),
    "vlm_decision_fusion": dict(
        label="VLM Decision Fusion", icon="⚖️", category="Models",
        inputs=[port("results_a", "results"), port("results_b", "results"), port("results_c", "results", True),
                port("image", "image", True), port("images", "images", True)],
        outputs=[port("text", "text"), port("results", "results")],
        params=[P("method", "Method", "select", "judge", options=["judge", "majority", "first"]),
                P("instruction", "Task the candidates answered", "textarea", "Which answer is most accurate for this image?")]
        + model_params("qwen3-vl-4b"),
        run=run_vlm_decision_fusion),
    "llm_decision_fusion": dict(
        label="LLM Decision Fusion", icon="🧮", category="Models",
        inputs=[port("a", "text", True), port("b", "text", True), port("c", "text", True),
                port("results_a", "results", True), port("results_b", "results", True), port("results_c", "results", True)],
        outputs=[port("text", "text"), port("results", "results")],
        params=[P("method", "Method", "select", "judge", options=["judge", "majority", "first"]),
                P("instruction", "Task the candidates answered", "textarea", "Pick the best-supported answer, or merge them if they complement each other.")]
        + model_params("qwen/qwen3-4b"),
        run=run_llm_decision_fusion),
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
