"""Tests for the flow agent. The LLM is stubbed, so no model server is needed.

Run: pixi run test   (or: python -m pytest test_agent.py)
"""
import json

import pytest

from agent import agent_settings, lint, parse_reply, run_agent
from components import NodeError, specs

SPECS = specs()


def ask(reply, graph=None, text="do it"):
    """Run one chat turn where the model answers with `reply` (dict -> JSON, str -> raw)."""
    raw = reply if isinstance(reply, str) else json.dumps(reply)
    seen = {}

    def llm(settings, messages):
        seen["messages"] = messages
        return raw

    out = run_agent({"messages": [{"role": "user", "content": text}], "graph": graph or {}}, SPECS, llm=llm)
    out["seen"] = seen["messages"]
    return out


def test_builds_pipeline_with_refs_resolved_to_real_ids():
    out = ask({"reply": "ok", "actions": [
        {"op": "add_node", "ref": "a", "type": "image_folder_reader", "params": {"folder": "/data/imgs"}},
        {"op": "add_node", "ref": "b", "type": "vlm", "params": {"prompt": "What is this?"}},
        {"op": "add_node", "ref": "c", "type": "csv_writer_image"},
        {"op": "connect", "from": "a.images", "to": "b.images"},
        {"op": "connect", "from": "b", "to": "c"},  # ports inferred from types
    ]})
    adds = [a for a in out["actions"] if a["op"] == "add_node"]
    ids = {a["type"]: a["id"] for a in adds}
    assert len(adds) == 3 and out["notes"] == []
    assert adds[0]["params"]["folder"] == "/data/imgs" and adds[0]["params"]["pattern"] == "*"  # defaults filled in
    assert adds[1]["params"]["prompt"] == "What is this?"
    edges = [a for a in out["actions"] if a["op"] == "connect"]
    assert edges[0] == {"op": "connect", "source": ids["image_folder_reader"], "sourceHandle": "images",
                        "target": ids["vlm"], "targetHandle": "images"}
    assert (edges[1]["sourceHandle"], edges[1]["targetHandle"]) == ("results", "results")
    assert edges[1]["target"] == ids["csv_writer_image"]


def test_rejects_hallucinated_types_ports_and_mismatched_connections():
    out = ask({"reply": "x", "actions": [
        {"op": "add_node", "ref": "a", "type": "magic_box"},
        {"op": "add_node", "ref": "b", "type": "text_input"},
        {"op": "add_node", "ref": "c", "type": "video_chunker"},
        {"op": "connect", "from": "b.text", "to": "c.video"},   # text -> video
        {"op": "connect", "from": "b.nope", "to": "c.video"},   # missing port
        {"op": "connect", "from": "ghost.text", "to": "c.video"},
        {"op": "bogus"},
    ]})
    assert [a["op"] for a in out["actions"]] == ["add_node", "add_node"]
    assert len(out["notes"]) == 5


def test_existing_nodes_can_be_referenced_and_inputs_take_one_connection():
    graph = {"nodes": [
        {"id": "t1", "type": "text_input", "params": {"text": "a"}},
        {"id": "t2", "type": "text_input", "params": {"text": "b"}},
        {"id": "l1", "type": "llm", "params": {"instruction": "Summarize"}},
    ], "edges": [{"source": "t1", "sourceHandle": "text", "target": "l1", "targetHandle": "text"}]}
    out = ask({"reply": "rewired", "actions": [{"op": "connect", "from": "t2.text", "to": "l1.text"}]}, graph)
    assert out["actions"][0]["source"] == "t2"
    assert not any(i for i in out["issues"] if "not connected" in i)


def test_set_params_coerces_values_and_refuses_files_and_secrets():
    graph = {"nodes": [{"id": "c1", "type": "video_chunker", "params": {}},
                       {"id": "v1", "type": "vlm", "params": {}},
                       {"id": "r1", "type": "video_reader", "params": {}}]}
    out = ask({"reply": "", "actions": [
        {"op": "set_params", "node": "c1", "params": {"chunk_seconds": "5", "max_chunks": "lots", "bogus": 1}},
        {"op": "set_params", "node": "v1", "params": {"api_key": "sk-123", "temperature": 0.5}},
        {"op": "set_params", "node": "r1", "params": {"file": "/etc/passwd"}},
    ]}, graph)
    assert out["actions"][0] == {"op": "set_params", "id": "c1", "params": {"chunk_seconds": 5}}
    assert out["actions"][1]["params"] == {"temperature": 0.5}
    assert len(out["actions"]) == 2  # nothing valid left for the file node
    assert not any("sk-123" in json.dumps(a) for a in out["actions"])
    assert len(out["notes"]) == 4


def test_remove_and_clear():
    graph = {"nodes": [{"id": "a", "type": "text_input", "params": {}}, {"id": "b", "type": "llm", "params": {}}],
             "edges": [{"source": "a", "sourceHandle": "text", "target": "b", "targetHandle": "text"}]}
    out = ask({"reply": "", "actions": [{"op": "remove_node", "node": "a"}]}, graph)
    assert out["actions"] == [{"op": "remove_node", "id": "a"}]
    assert any("'text'" in i and "not connected" in i for i in out["issues"])  # LLM lost its input
    assert ask({"reply": "", "actions": [{"op": "clear"}]}, graph)["issues"] == []


def test_lint_reports_missing_inputs_files_and_failed_runs():
    from agent import parse_graph

    by_type = {s["type"]: s for s in SPECS}
    graph = parse_graph({"nodes": [
        {"id": "w", "type": "csv_writer_image", "params": {}},
        {"id": "v", "type": "vlm", "params": {"prompt": ""}, "status": "error", "error": "Model server error"},
        {"id": "i", "type": "image_input", "params": {"file": None}},
        {"id": "f", "type": "image_folder_reader", "params": {"folder": " "}},
    ], "edges": []}, by_type)
    text = "\n".join(lint(graph, by_type))
    for needle in ("input 'results'", "nothing is connected", "'Prompt' is empty", "last run failed", "no file uploaded", "'Folder path"):
        assert needle in text


def test_prompt_contains_canvas_problems_and_node_catalog_but_not_secrets():
    graph = {"nodes": [{"id": "v1", "type": "vlm", "params": {"prompt": "hi", "api_key": "sk-secret"}}]}
    out = ask({"reply": "fine", "actions": []}, graph, text="review my flow")
    system, user = out["seen"][0]["content"], out["seen"][-1]["content"]
    assert "image_folder_reader" in system and "csv_writer_video" in system
    assert "v1 = vlm" in user and "review my flow" in user and "nothing is connected" in user
    assert "sk-secret" not in user and "sk-secret" not in system


def test_parse_reply_handles_fences_thinking_and_plain_text():
    assert parse_reply('<think>hmm</think>```json\n{"reply":"hi","actions":[]}\n```') == ("hi", [])
    assert parse_reply('Sure! {"reply":"hi","actions":[{"op":"clear"}]} done')[1] == [{"op": "clear"}]
    assert parse_reply("Just chatting, no JSON.") == ("Just chatting, no JSON.", [])
    assert parse_reply('{"not": "ours"}') == ('{"not": "ours"}', [])


def test_plain_text_answer_is_returned_as_reply_with_no_actions():
    out = ask("You could add a CSV writer after the VLM.")
    assert out["reply"].startswith("You could add") and out["actions"] == []


def test_input_validation():
    with pytest.raises(NodeError):
        run_agent({"messages": []}, SPECS, llm=lambda s, m: "")
    with pytest.raises(NodeError):
        run_agent({"messages": [{"role": "assistant", "content": "hi"}]}, SPECS, llm=lambda s, m: "")
    with pytest.raises(NodeError):
        agent_settings({"base_url": "file:///etc/passwd"})
