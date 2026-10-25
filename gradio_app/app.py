"""AI components: Gradio UI + HTTP API (for the React Flow canvas) on one server.

Run: pixi run dev  ->  http://127.0.0.1:7860  (Gradio)  and  /flow/ (drag & drop canvas, if built)
LM Studio must be serving the models: lms server start --cors
"""
import base64
import io
import uuid
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from components import OUTPUT_DIR, NodeError, media_path, run_node, save_upload, specs

FLOW_DIST = Path(__file__).parent.parent / "reactflow_app" / "dist"
SPECS = {s["type"]: s for s in specs()}

# ---------- HTTP API (used by the React Flow canvas) ----------

api = FastAPI(title="AI components")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    type: str
    params: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)


@api.get("/api/nodes")
def list_nodes():
    return specs()


@api.post("/api/upload")
def upload(file: UploadFile):
    dest = save_upload(file.file, file.filename or "upload")
    return {"path": str(dest), "name": file.filename}


@api.get("/api/media")
def media(path: str):
    """Serve a file that an upload or reader node listed (used for thumbnails)."""
    try:
        return FileResponse(media_path(path))
    except NodeError as e:
        raise HTTPException(404, str(e)) from e


@api.get("/api/outputs/{name}")
def download_output(name: str):
    file = OUTPUT_DIR / Path(name).name
    if not file.is_file():
        raise HTTPException(404, "No such output file.")
    return FileResponse(file, filename=file.name)


@api.post("/api/run")
def run(req: RunRequest):
    try:
        return {"outputs": run_node(req.type, req.params, req.inputs)}
    except NodeError as e:
        raise HTTPException(400, str(e)) from e


# ---------- Gradio pipeline builder ----------

def gr_file_types(accept: str | None):
    if not accept:
        return None
    return {"image/*": ["image"], "video/*": ["video"]}.get(accept, accept.split(","))


def param_component(p: dict, key: str):
    label, default = p["label"], p["default"]
    match p["kind"]:
        case "textarea":
            return gr.Textbox(label=label, value=default, lines=3, key=key)
        case "number":
            return gr.Number(label=label, value=default, key=key)
        case "password":
            return gr.Textbox(label=label, value=default, type="password", key=key)
        case "file":
            return gr.File(label=label, type="filepath", file_types=gr_file_types(p.get("accept")), key=key)
        case _:
            return gr.Textbox(label=label, value=default, key=key)


def to_image(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def summarize(spec: dict, outputs: dict):
    """Turn node outputs into (text summary, gallery update) for the UI."""
    lines, gallery = [], []
    for port in spec["outputs"]:
        value = outputs.get(port["name"])
        match port["type"]:
            case "text":
                lines.append(value)
            case "image":
                if value:
                    gallery.append((to_image(value), port["name"]))
                    lines.append(f"[{port['name']}: image]")
            case "video":
                lines.append(f"video: {value['name']} · {value['duration']}s · {value['fps']} fps · {value['width']}x{value['height']}")
            case "chunks":
                lines.append(f"{len(value)} chunks")
                gallery += [(to_image(c["frames"][0]), f"{c['start']}s–{c['end']}s") for c in value if c["frames"]]
            case "images":
                lines.append(f"{len(value)} images: " + ", ".join(i["name"] for i in value[:8]) + (" …" if len(value) > 8 else ""))
                gallery += [(i["path"], i["name"]) for i in value[:12]]
            case "videos":
                lines.append(f"{len(value)} videos: " + "; ".join(f"{v['name']} ({v['duration']}s)" for v in value[:8]))
            case "results":
                lines += [f"{len(value)} results:"] + [f"- {r['name']}: {r['text'][:160]}" for r in value[:8]]
            case "file":
                lines.append(f"Wrote {value['rows']} rows to {OUTPUT_DIR / value['name']}")
    return "\n".join(lines), gr.update(value=gallery or None, visible=bool(gallery))


def wire(spec: dict, upstream: list) -> dict:
    """Each input takes the nearest upstream output of the same type."""
    inputs = {}
    for port in spec["inputs"]:
        for up_spec, outs in reversed(upstream):
            hit = next((o["name"] for o in up_spec["outputs"] if o["type"] == port["type"] and o["name"] in outs), None)
            if hit:
                inputs[port["name"]] = outs[hit]
                break
    return inputs


def build_ui() -> gr.Blocks:
    choices = [(f"{s['icon']} {s['label']}", s["type"]) for s in SPECS.values()]

    with gr.Blocks(title="AI Components") as demo:
        gr.Markdown("# AI Components")
        with gr.Tabs():
            with gr.Tab("Canvas (drag & drop)"):
                if FLOW_DIST.exists():
                    gr.HTML('<iframe src="/flow/" style="width:100%;height:78vh;border:1px solid #cbd5e1;border-radius:8px"></iframe>')
                else:
                    gr.Markdown("Canvas not built yet. Run `pixi run build` in `reactflow_app`, then restart this server.")

            with gr.Tab("Pipeline builder"):
                gr.Markdown("Add components, set their options, then run. Each input takes the nearest upstream output of the same type.")
                nodes = gr.State([])
                with gr.Row():
                    kind = gr.Dropdown(choices, value="vlm", label="Component", scale=3)
                    add = gr.Button("➕ Add", scale=1)
                    clear = gr.Button("Clear", scale=1)
                add.click(lambda s, k: s + [{"id": uuid.uuid4().hex[:6], "type": k}], [nodes, kind], nodes)
                clear.click(lambda: [], None, nodes)

                @gr.render(inputs=nodes)
                def draw(items):
                    if not items:
                        gr.Markdown("_Add a component to start._")
                        return
                    specs_in_order = [SPECS[n["type"]] for n in items]
                    param_comps, outs = [], []

                    def mover(idx, step):
                        def move(s):
                            j = idx + step
                            if 0 <= j < len(s):
                                s = list(s)
                                s[idx], s[j] = s[j], s[idx]
                            return s
                        return move

                    for i, (n, spec) in enumerate(zip(items, specs_in_order)):
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown(f"### {spec['icon']} {i + 1}. {spec['label']}")
                                up = gr.Button("↑", scale=0, min_width=48)
                                down = gr.Button("↓", scale=0, min_width=48)
                                rm = gr.Button("✕", scale=0, min_width=48)
                            up.click(mover(i, -1), nodes, nodes)
                            down.click(mover(i, 1), nodes, nodes)
                            rm.click(lambda s, idx=i: [x for k, x in enumerate(s) if k != idx], nodes, nodes)

                            comps = []
                            advanced = []
                            for p in spec["params"]:
                                (advanced if p.get("advanced") else comps).append(p)
                            comps = [param_component(p, f"{n['id']}-{p['name']}") for p in comps]
                            if advanced:
                                with gr.Accordion("Model settings", open=False):
                                    comps += [param_component(p, f"{n['id']}-{p['name']}") for p in advanced]
                            param_comps.append(comps)
                            outs += [gr.Textbox(label="Output", lines=3, interactive=False, key=f"{n['id']}-out"),
                                     gr.Gallery(label="Frames", visible=False, columns=4, height=160, key=f"{n['id']}-gal")]

                    run_btn = gr.Button("▶ Run pipeline", variant="primary")

                    def run_all(*values):
                        it = iter(values)
                        shown = [gr.update()] * len(outs)
                        upstream = []
                        for i, (n, spec) in enumerate(zip(items, specs_in_order)):
                            # values arrive in widget order: plain params first, then advanced
                            ordered = [p for p in spec["params"] if not p.get("advanced")] + [p for p in spec["params"] if p.get("advanced")]
                            params = {p["name"]: next(it) for p in ordered}
                            shown[2 * i] = "⏳ Running…"
                            yield list(shown)
                            try:
                                for p in spec["params"]:
                                    if p["kind"] == "file" and params[p["name"]]:
                                        params[p["name"]] = str(save_upload(params[p["name"]], Path(params[p["name"]]).name))
                                result = run_node(n["type"], params, wire(spec, upstream))
                            except NodeError as e:
                                shown[2 * i] = f"❌ {e}"
                                yield list(shown)
                                return
                            upstream.append((spec, result))
                            shown[2 * i], shown[2 * i + 1] = summarize(spec, result)
                            yield list(shown)

                    run_btn.click(run_all, [c for comps in param_comps for c in comps], outs)
    return demo


# Order matters: API and static routes first, Gradio catches the rest at "/".
if FLOW_DIST.exists():
    api.mount("/flow", StaticFiles(directory=FLOW_DIST, html=True), name="flow")
app = gr.mount_gradio_app(api, build_ui(), path="/")
