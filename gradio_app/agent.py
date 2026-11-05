"""Flow agent: turns a chat message plus the current canvas into validated edit actions.

The LLM only *proposes* edits as JSON. Every action is checked here against the node specs
(real node types, real ports, matching port types, valid param values) before the React canvas
applies it, so a small local model cannot put an impossible graph on screen.

Action shapes returned to the canvas (all ids are concrete node ids):
  add_node    {op, id, type, params}
  connect     {op, source, sourceHandle, target, targetHandle}
  disconnect  {op, source, sourceHandle, target, targetHandle}
  set_params  {op, id, params}
  remove_node {op, id}
  arrange     {op}
  clear       {op}
"""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from typing import Callable

from components import NodeError, chat_messages

MAX_ACTIONS = 40
MAX_HISTORY = 10
MAX_MESSAGE_CHARS = 4000
OPS = {"add_node", "connect", "disconnect", "set_params", "remove_node", "arrange", "clear"}
DEFAULT_BASE_URL = "http://localhost:1234/v1"
# Settings the model may never change: pointing a node at another server would send that node's
# browser-held API key and its inputs there on the next run.
USER_ONLY_PARAMS = {"base_url"}
# Nodes whose inputs are all optional but that need at least N of them wired to run.
NEEDS_ANY_OF = {
    "vlm": (("image", "chunks", "images", "videos"), 1),
    "object_detector": (("image", "images"), 1),
    "image_crop": (("detections", "image", "images"), 1),
    "llm_decision_fusion": (("a", "b", "c", "results_a", "results_b", "results_c"), 2),
}


def agent_settings(raw: dict | None) -> dict:
    """Model settings for the agent: request values, then env vars, then LM Studio defaults."""
    raw = raw or {}

    def pick(key: str, env: str, default: str) -> str:
        return str(raw.get(key) or os.environ.get(env) or default).strip()

    base_url = pick("base_url", "AGENT_BASE_URL", DEFAULT_BASE_URL)
    if not re.match(r"https?://", base_url):
        raise NodeError("Agent base URL must start with http:// or https://")
    # The server's own key only goes to the server URL it was configured for, never to a caller-chosen one.
    configured = (os.environ.get("AGENT_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    env_key = os.environ.get("AGENT_API_KEY", "") if base_url.rstrip("/") == configured else ""
    return {
        "base_url": base_url,
        "model": pick("model", "AGENT_MODEL", "qwen/qwen3-4b"),
        "api_key": str(raw.get("api_key") or env_key),
        "temperature": 0.2,
        "max_tokens": 3000,
    }


# ---------- prompt ----------

def _param_hint(p: dict) -> str:
    if p["kind"] == "select":
        return f"{p['name']}=one of {'|'.join(p['options'])}"
    return p["name"]


def catalog(specs: list[dict]) -> str:
    """One compact line per node type: what it needs, what it produces, which options matter."""
    lines = []
    for s in specs:
        ins = ", ".join(f"{p['name']}:{p['type']}{'?' if p['optional'] else ''}" for p in s["inputs"]) or "none"
        outs = ", ".join(f"{p['name']}:{p['type']}" for p in s["outputs"]) or "none"
        params = ", ".join(_param_hint(p) for p in s["params"] if not p.get("advanced") and p["kind"] != "file")
        needs_file = " (user uploads a file in the node)" if any(p["kind"] == "file" for p in s["params"]) else ""
        lines.append(f"- {s['type']} [{s['category']}] {s['label']}: in({ins}) out({outs}) params({params or 'none'}){needs_file}")
    return "\n".join(lines)


SYSTEM_TEMPLATE = """You are the assistant inside a drag-and-drop AI pipeline canvas. You help the user pick the right \
boxes (nodes), connect them, arrange them and review the result. You can edit the canvas by returning actions.

NODE TYPES (type [category] label: in(port:type, '?' = optional) out(port:type) params):
{catalog}

RULES
- Connect an output port to an input port of the SAME type only (text->text, images->images, ...).
- An input accepts one connection. A source node (no inputs) starts a pipeline. Writers end it.
- Use only node types and port names listed above. Never invent ids: new nodes get a short ref of your choice \
(e.g. "a", "b") that you can use in later actions of the same reply; existing nodes use their canvas id.
- In "connect", write "ref.port" (e.g. "a.images"). If you leave the port out ("a"), the matching ports are picked for you.
- You cannot upload files, type secrets or run the flow; if a node needs a file or API key, tell the user.
- The canvas is auto-arranged after you add or connect nodes; add {{"op":"arrange"}} only when the user asks to tidy up.
- If the user asks a question or wants a review, use no actions and answer with concrete recommendations \
(missing inputs, better node choices, settings worth changing). Base reviews on the CANVAS and PROBLEMS given.
- Keep replies short and specific.

Reply with ONLY one JSON object, no markdown:
{{"reply": "<text for the user>", "actions": [ ... ]}}

Action formats:
{{"op":"add_node","ref":"a","type":"<node type>","params":{{"<param>":"<value>"}}}}
{{"op":"connect","from":"a.images","to":"b.images"}}
{{"op":"disconnect","from":"a.images","to":"b.images"}}
{{"op":"set_params","node":"<id or ref>","params":{{"<param>":"<value>"}}}}
{{"op":"remove_node","node":"<id or ref>"}}
{{"op":"arrange"}}
{{"op":"clear"}}

Example. User: "describe every image in a folder and save a CSV"
{{"reply":"Added a folder reader -> VLM -> CSV writer. Set the folder path in the first node.","actions":[
{{"op":"add_node","ref":"a","type":"image_folder_reader"}},
{{"op":"add_node","ref":"b","type":"vlm","params":{{"prompt":"Describe this image."}}}},
{{"op":"add_node","ref":"c","type":"csv_writer_image"}},
{{"op":"connect","from":"a.images","to":"b.images"}},
{{"op":"connect","from":"b.results","to":"c.results"}}]}}"""


def system_prompt(specs: list[dict]) -> str:
    return SYSTEM_TEMPLATE.format(catalog=catalog(specs))


def _short(value, n: int = 120) -> str:
    s = str(value)
    return s if len(s) <= n else s[:n] + "…"


def describe_graph(graph: dict, specs_by_type: dict) -> str:
    nodes, edges = graph["nodes"], graph["edges"]
    if not nodes:
        return "(empty canvas)"
    lines = []
    for n in nodes.values():
        spec = specs_by_type[n["type"]]
        shown = {k: _short(v) for k, v in n["params"].items()
                 if v not in (None, "") and k in {p["name"] for p in spec["params"] if p["kind"] not in ("file", "password")}}
        status = f" status={n['status']}" if n.get("status") not in (None, "", "idle") else ""
        lines.append(f"- {n['id']} = {n['type']}{status} params={json.dumps(shown, ensure_ascii=False)}")
    lines += [f"- edge {s}.{sh} -> {t}.{th}" for (s, sh, t, th) in sorted(edges)] or ["- (no connections)"]
    return "\n".join(lines)


# ---------- graph state ----------

def parse_graph(raw: dict | None, specs_by_type: dict) -> dict:
    """Normalise the client's canvas into {nodes: {id: node}, edges: {(src, out, dst, in)}}.

    Unknown node types and dangling edges are dropped so the rest of the module can trust the shape.
    """
    raw = raw or {}
    nodes: dict[str, dict] = {}
    for n in raw.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") and n.get("type") in specs_by_type:
            nodes[str(n["id"])] = {
                "id": str(n["id"]), "type": n["type"],
                "params": dict(n.get("params") or {}), "status": n.get("status"), "error": n.get("error") or "",
            }
    edges = set()
    for e in raw.get("edges") or []:
        if not isinstance(e, dict):
            continue
        key = (str(e.get("source")), str(e.get("sourceHandle")), str(e.get("target")), str(e.get("targetHandle")))
        if key[0] in nodes and key[2] in nodes:
            edges.add(key)
    return {"nodes": nodes, "edges": edges}


def _port(spec: dict, direction: str, name: str) -> dict | None:
    return next((p for p in spec[direction] if p["name"] == name), None)


def _pick_ports(src: dict, dst: dict, out_name: str | None, in_name: str | None, taken: set[str]):
    """Resolve (output port, input port), filling in a missing side by matching port types."""
    outs = [p for p in src["outputs"] if p["name"] == out_name] if out_name else src["outputs"]
    ins = [p for p in dst["inputs"] if p["name"] == in_name] if in_name else dst["inputs"]
    pairs = [(o, i) for o in outs for i in ins if o["type"] == i["type"]]
    # Prefer an input that is still free so "connect a to b" fills different inputs in turn.
    pairs.sort(key=lambda oi: oi[1]["name"] in taken)
    return pairs[0] if pairs else None


def _reaches(edges, start: str, goal: str) -> bool:
    """True if `goal` can be reached from `start` by following edges downstream."""
    seen, todo = set(), [start]
    while todo:
        node = todo.pop()
        if node == goal:
            return True
        if node not in seen:
            seen.add(node)
            todo += [e[2] for e in edges if e[0] == node]
    return False


def _split_ref(text) -> tuple[str, str | None]:
    text = str(text or "").strip()
    node, _, port = text.rpartition(".")
    return (node, port) if node else (text, None)


def _coerce_param(spec_param: dict, value):
    """Return (ok, value) for a model-proposed param value."""
    kind = spec_param["kind"]
    if kind == "number":
        if isinstance(value, bool):
            return False, None
        try:
            num = float(value)
        except (TypeError, ValueError, OverflowError):
            return False, None
        if not math.isfinite(num):  # "NaN" / "Infinity" / "1e999" parse, but cannot be sent back as JSON
            return False, None
        return True, int(num) if num.is_integer() else num
    if kind == "select":
        return (True, value) if value in spec_param["options"] else (False, None)
    if isinstance(value, (dict, list)) or value is None:
        return False, None
    return True, str(value)


def _clean_params(spec: dict, raw, notes: list[str]) -> dict:
    out = {}
    by_name = {p["name"]: p for p in spec["params"]}
    for key, value in (raw if isinstance(raw, dict) else {}).items():
        p = by_name.get(key)
        if p is None:
            notes.append(f"{spec['label']} has no setting '{key}'.")
        elif p["kind"] in ("file", "password") or key in USER_ONLY_PARAMS:
            notes.append(f"'{p['label']}' on {spec['label']} must be filled in by you.")
        else:
            ok, coerced = _coerce_param(p, value)
            if ok:
                out[key] = coerced
            else:
                notes.append(f"Ignored invalid value for '{p['label']}' on {spec['label']}.")
    return out


def validate_actions(raw_actions, graph: dict, specs_by_type: dict) -> tuple[list[dict], list[str]]:
    """Check proposed actions one by one against a simulated copy of the canvas.

    Returns (actions that are safe to apply, notes about the ones that were dropped). `graph` is mutated
    to the state after the valid actions, which is what lint() should look at.
    """
    nodes, edges = graph["nodes"], graph["edges"]
    refs: dict[str, str] = {}
    actions: list[dict] = []
    notes: list[str] = []

    def resolve(name) -> str | None:
        name = str(name or "").strip()
        node_id = refs.get(name, name)
        return node_id if node_id in nodes else None  # refs outlive remove_node / clear

    def drop_edges(pred):
        for e in [e for e in edges if pred(e)]:
            edges.discard(e)

    if not isinstance(raw_actions, list):
        raw_actions = []
    if len(raw_actions) > MAX_ACTIONS:
        notes.append(f"Only the first {MAX_ACTIONS} actions were applied.")

    for act in raw_actions[:MAX_ACTIONS]:
        op = act.get("op") if isinstance(act, dict) else None
        if not isinstance(op, str) or op not in OPS:
            notes.append(f"Skipped an unknown action: {_short(act, 60)}")
            continue

        if op == "add_node":
            node_type = act.get("type")
            spec = specs_by_type.get(node_type) if isinstance(node_type, str) else None
            if spec is None:
                notes.append(f"Skipped unknown node type '{act.get('type')}'.")
                continue
            node_id = f"{spec['type']}_{uuid.uuid4().hex[:6]}"
            params = {p["name"]: p["default"] for p in spec["params"]}
            params.update(_clean_params(spec, act.get("params"), notes))
            nodes[node_id] = {"id": node_id, "type": spec["type"], "params": params, "status": "idle", "error": ""}
            if act.get("ref"):
                refs[str(act["ref"])] = node_id
            actions.append({"op": op, "id": node_id, "type": spec["type"], "params": params})

        elif op in ("connect", "disconnect"):
            (a, a_port), (b, b_port) = _split_ref(act.get("from")), _split_ref(act.get("to"))
            src, dst = resolve(a), resolve(b)
            if src is None or dst is None:
                notes.append(f"Skipped {op}: unknown node in '{act.get('from')}' -> '{act.get('to')}'.")
                continue
            if src == dst:
                notes.append(f"Skipped {op}: a node cannot connect to itself.")
                continue
            src_spec, dst_spec = specs_by_type[nodes[src]["type"]], specs_by_type[nodes[dst]["type"]]
            if op == "disconnect":
                hit = [e for e in edges if e[0] == src and e[2] == dst
                       and a_port in (None, e[1]) and b_port in (None, e[3])]
                for e in hit:
                    edges.discard(e)
                    actions.append({"op": op, "source": e[0], "sourceHandle": e[1], "target": e[2], "targetHandle": e[3]})
                continue
            taken = {e[3] for e in edges if e[2] == dst}
            pair = _pick_ports(src_spec, dst_spec, a_port, b_port, taken)
            if pair is None:
                notes.append(f"Cannot connect {src_spec['label']} to {dst_spec['label']}: no matching port types.")
                continue
            out_p, in_p = pair
            kept = {e for e in edges if not (e[2] == dst and e[3] == in_p["name"])}
            if _reaches(kept, dst, src):
                notes.append(f"Skipped connect: {src_spec['label']} -> {dst_spec['label']} would make a loop.")
                continue
            drop_edges(lambda e: e[2] == dst and e[3] == in_p["name"])  # an input takes one connection
            edges.add((src, out_p["name"], dst, in_p["name"]))
            actions.append({"op": op, "source": src, "sourceHandle": out_p["name"], "target": dst, "targetHandle": in_p["name"]})

        elif op == "set_params":
            node_id = resolve(act.get("node"))
            if node_id is None:
                notes.append(f"Skipped set_params: unknown node '{act.get('node')}'.")
                continue
            params = _clean_params(specs_by_type[nodes[node_id]["type"]], act.get("params"), notes)
            if params:
                nodes[node_id]["params"].update(params)
                actions.append({"op": op, "id": node_id, "params": params})

        elif op == "remove_node":
            node_id = resolve(act.get("node"))
            if node_id is None:
                notes.append(f"Skipped remove_node: unknown node '{act.get('node')}'.")
                continue
            del nodes[node_id]
            drop_edges(lambda e: node_id in (e[0], e[2]))
            actions.append({"op": op, "id": node_id})

        elif op == "clear":
            nodes.clear()
            edges.clear()
            actions.append({"op": op})

        else:  # arrange
            actions.append({"op": op})

    return actions, notes


# ---------- review ----------

def lint(graph: dict, specs_by_type: dict) -> list[str]:
    """Hard facts about the canvas that would stop it from running; also fed to the model as context."""
    nodes, edges = graph["nodes"], graph["edges"]
    issues = []
    for n in nodes.values():
        spec = specs_by_type[n["type"]]
        name = f"{spec['label']} ({n['id']})"
        wired = {e[3] for e in edges if e[2] == n["id"]}
        for p in spec["inputs"]:
            if not p["optional"] and p["name"] not in wired:
                issues.append(f"{name}: input '{p['name']}' ({p['type']}) is not connected.")
        if n["type"] in NEEDS_ANY_OF:
            names, need = NEEDS_ANY_OF[n["type"]]
            if len(wired & set(names)) < need:
                issues.append(f"{name}: connect {'at least two' if need > 1 else 'one'} of its inputs ({', '.join(names)}).")
        elif spec["inputs"] and not wired and all(p["optional"] for p in spec["inputs"]):
            issues.append(f"{name}: nothing is connected to its inputs ({', '.join(p['name'] for p in spec['inputs'])}).")
        for p in spec["params"]:
            value = n["params"].get(p["name"])
            if p["kind"] == "file" and not value:
                issues.append(f"{name}: no file uploaded for '{p['label']}'.")
            elif p["name"] == "folder" and not str(value or "").strip():
                issues.append(f"{name}: '{p['label']}' is empty.")
            elif p["name"] == "prompt" and not str(value or "").strip():  # the VLM refuses to run without one
                issues.append(f"{name}: '{p['label']}' is empty.")
        if n.get("status") == "error" and n.get("error"):
            issues.append(f"{name}: last run failed: {_short(n['error'], 160)}")
    if any(_reaches({e for e in edges if e != x}, x[2], x[0]) for x in edges):
        issues.append("The flow contains a loop, so Run flow will not start. Remove one of the connections in it.")
    return issues


# ---------- LLM round trip ----------

def parse_reply(text: str) -> tuple[str, list]:
    """Extract {"reply", "actions"} from a model answer; fall back to treating it all as plain chat."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    candidates = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)]
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for blob in candidates:
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        if isinstance(data, dict) and ("reply" in data or "actions" in data):
            actions = data.get("actions")
            return str(data.get("reply") or "").strip(), actions if isinstance(actions, list) else []
    return text, []


def build_messages(specs: list[dict], specs_by_type: dict, history: list, graph: dict, issues: list[str]) -> list[dict]:
    turns = []
    for m in history[-MAX_HISTORY:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip():
            turns.append({"role": m["role"], "content": str(m["content"])[:MAX_MESSAGE_CHARS]})
    if not turns or turns[-1]["role"] != "user":
        raise NodeError("Send a message first.")
    state = (f"CANVAS (current):\n{describe_graph(graph, specs_by_type)}\n\n"
             f"PROBLEMS:\n{chr(10).join('- ' + i for i in issues) if issues else '- none found'}\n\n"
             f"USER REQUEST:\n{turns[-1]['content']}")
    turns[-1] = {"role": "user", "content": state}
    return [{"role": "system", "content": system_prompt(specs)}, *turns]


def run_agent(payload: dict, specs: list[dict], llm: Callable[[dict, list[dict]], str] = chat_messages) -> dict:
    """Answer one chat turn. Returns {reply, actions, notes, issues}."""
    by_type = {s["type"]: s for s in specs}
    settings = agent_settings(payload.get("settings"))
    graph = parse_graph(payload.get("graph"), by_type)
    messages = build_messages(specs, by_type, payload.get("messages") or [], graph, lint(graph, by_type))

    reply, raw_actions = parse_reply(llm(settings, messages))
    actions, notes = validate_actions(raw_actions, graph, by_type)
    # `graph` now holds the canvas after the valid actions: report what is still missing there.
    issues = lint(graph, by_type)
    if not reply:
        reply = "Done." if actions else "I could not turn that into canvas changes. Try rephrasing."
    return {"reply": reply, "actions": actions, "notes": notes, "issues": issues}
