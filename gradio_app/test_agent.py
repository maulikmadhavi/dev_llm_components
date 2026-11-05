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
    for needle in ("input 'results'", "connect one of its inputs", "'Prompt' is empty", "last run failed", "no file uploaded", "'Folder path"):
        assert needle in text


def test_prompt_contains_canvas_problems_and_node_catalog_but_not_secrets():
    graph = {"nodes": [{"id": "v1", "type": "vlm", "params": {"prompt": "hi", "api_key": "sk-secret"}}]}
    out = ask({"reply": "fine", "actions": []}, graph, text="review my flow")
    system, user = out["seen"][0]["content"], out["seen"][-1]["content"]
    assert "image_folder_reader" in system and "csv_writer_video" in system
    assert "v1 = vlm" in user and "review my flow" in user and "connect one of its inputs" in user
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


# ---------- review fixes ----------

def test_env_api_key_is_only_sent_to_the_configured_server(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "server-secret")
    monkeypatch.setenv("AGENT_BASE_URL", "http://llm.internal/v1")
    monkeypatch.setenv("AGENT_ALLOW_CUSTOM_BASE_URL", "1")  # opt in so another URL can be tried at all
    assert agent_settings({})["api_key"] == "server-secret"
    assert agent_settings({"base_url": "http://llm.internal/v1/"})["api_key"] == "server-secret"
    assert agent_settings({"base_url": "http://other.example/v1"})["api_key"] == ""
    assert agent_settings({"base_url": "http://other.example/v1", "api_key": "mine"})["api_key"] == "mine"
    monkeypatch.delenv("AGENT_BASE_URL")
    assert agent_settings({})["api_key"] == "server-secret"  # default URL is the configured one
    assert agent_settings({"base_url": "http://other.example/v1"})["api_key"] == ""


def test_caller_cannot_pick_another_model_server_unless_the_admin_opts_in(monkeypatch):
    monkeypatch.delenv("AGENT_ALLOW_CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    assert agent_settings({"base_url": "http://localhost:1234/v1/"})["base_url"] == "http://localhost:1234/v1/"
    for url in ("http://169.254.169.254/latest", "http://10.0.0.5:8080/v1", "https://evil.example/v1"):
        with pytest.raises(NodeError, match="configured assistant model server"):
            agent_settings({"base_url": url})
    monkeypatch.setenv("AGENT_BASE_URL", "http://llm.internal/v1")
    assert agent_settings({})["base_url"] == "http://llm.internal/v1"
    with pytest.raises(NodeError):
        agent_settings({"base_url": "http://localhost:1234/v1"})  # the old default is no longer the configured one
    monkeypatch.setenv("AGENT_ALLOW_CUSTOM_BASE_URL", "true")
    assert agent_settings({"base_url": "http://other.example/v1"})["base_url"] == "http://other.example/v1"


def test_agent_calls_have_a_short_bounded_timeout(monkeypatch):
    monkeypatch.delenv("AGENT_TIMEOUT", raising=False)
    assert agent_settings({})["timeout"] == 120
    for env, want in (("30", 30), ("1", 5), ("99999", 600), ("inf", 120), ("nan", 120), ("soon", 120)):
        monkeypatch.setenv("AGENT_TIMEOUT", env)
        assert agent_settings({})["timeout"] == want


def test_chat_messages_uses_the_timeout_and_hides_error_details_from_the_agent_path(monkeypatch):
    import requests

    import components

    seen = {}

    def boom(url, **kw):
        seen["timeout"] = kw["timeout"]
        raise requests.ConnectionError("HTTPConnectionPool(host='10.0.0.5', port=8080): refused")

    monkeypatch.setattr(components.requests, "post", boom)
    params = {"base_url": "http://x/v1", "model": "m", "quiet_errors": True, "timeout": 7}
    with pytest.raises(NodeError) as err:
        components.chat_messages(params, [{"role": "user", "content": "hi"}])
    assert seen["timeout"] == 7 and "10.0.0.5" not in str(err.value) and "refused" not in str(err.value)

    def slow(url, **kw):
        raise requests.Timeout("Read timed out (host 10.0.0.5)")

    monkeypatch.setattr(components.requests, "post", slow)
    with pytest.raises(NodeError, match="took too long") as err:
        components.chat_messages(params, [{"role": "user", "content": "hi"}])
    assert "10.0.0.5" not in str(err.value)
    # ordinary nodes keep their detailed errors and the 10 minute default
    with pytest.raises(NodeError, match="10.0.0.5"):
        components.chat_messages({"base_url": "http://x/v1", "model": "m"}, [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(components.requests, "post", boom)
    with pytest.raises(NodeError):
        components.chat_messages({"base_url": "http://x/v1", "model": "m"}, [])
    assert seen["timeout"] == 600


def test_model_cannot_redirect_a_node_to_another_server():
    graph = {"nodes": [{"id": "v1", "type": "vlm", "params": {"base_url": "http://localhost:1234/v1"}}]}
    out = ask({"reply": "", "actions": [
        {"op": "set_params", "node": "v1", "params": {"base_url": "http://evil.example/v1", "model": "other"}},
        {"op": "add_node", "type": "llm", "params": {"base_url": "http://evil.example/v1"}},
    ]}, graph)
    assert out["actions"][0]["params"] == {"model": "other"}
    assert out["actions"][1]["params"]["base_url"] == "http://localhost:1234/v1"
    assert len(out["notes"]) == 2


def test_connections_that_would_form_a_loop_are_rejected():
    graph = {"nodes": [{"id": "a", "type": "llm", "params": {"instruction": "x"}},
                       {"id": "b", "type": "llm", "params": {"instruction": "x"}}],
             "edges": [{"source": "a", "sourceHandle": "text", "target": "b", "targetHandle": "text"}]}
    out = ask({"reply": "", "actions": [{"op": "connect", "from": "b.text", "to": "a.text"}]}, graph)
    assert out["actions"] == [] and "loop" in out["notes"][0]
    # replacing the edge that feeds the input is fine: b no longer depends on a afterwards
    out = ask({"reply": "", "actions": [{"op": "disconnect", "from": "a", "to": "b"},
                                        {"op": "connect", "from": "b.text", "to": "a.text"}]}, graph)
    assert [a["op"] for a in out["actions"]] == ["disconnect", "connect"] and out["notes"] == []


def test_lint_reports_loops_in_existing_canvases():
    cyc = {"nodes": [{"id": "a", "type": "llm", "params": {}}, {"id": "b", "type": "llm", "params": {}}],
           "edges": [{"source": "a", "sourceHandle": "text", "target": "b", "targetHandle": "text"},
                     {"source": "b", "sourceHandle": "text", "target": "a", "targetHandle": "text"}]}
    assert any("loop" in i for i in ask({"reply": "ok", "actions": []}, cyc)["issues"])


def test_non_finite_numbers_are_ignored_and_the_response_is_json_serialisable():
    graph = {"nodes": [{"id": "c1", "type": "video_chunker", "params": {}}]}
    out = ask({"reply": "", "actions": [{"op": "set_params", "node": "c1", "params": {
        "chunk_seconds": "NaN", "frames_per_chunk": "Infinity", "max_chunks": "1e999"}}]}, graph)
    json.dumps(out["notes"]) and json.dumps(out["actions"])
    assert out["actions"] == [] and len(out["notes"]) == 3


def test_refs_do_not_resolve_after_remove_or_clear():
    for kill in ({"op": "remove_node", "node": "a"}, {"op": "clear"}):
        out = ask({"reply": "", "actions": [
            {"op": "add_node", "ref": "a", "type": "llm"}, kill,
            {"op": "set_params", "node": "a", "params": {"instruction": "x"}},
            {"op": "connect", "from": "a", "to": "a"},
            {"op": "remove_node", "node": "a"}]})
        assert [a["op"] for a in out["actions"]] == ["add_node", kill["op"]]
        assert len(out["notes"]) == 3


def test_malformed_op_and_type_become_notes_not_errors():
    out = ask({"reply": "x", "actions": [
        {"op": ["add_node"]}, {"op": {"a": 1}}, {"op": "add_node", "type": ["llm"]}, {"op": "add_node", "type": {"a": 1}},
        {"op": "set_params", "node": ["a"], "params": {}}, "not even a dict", None]})
    assert out["actions"] == [] and len(out["notes"]) == 7


def test_lint_matches_what_nodes_actually_require():
    from agent import parse_graph

    by_type = {s["type"]: s for s in SPECS}

    def lint_of(nodes, edges=()):
        return "\n".join(lint(parse_graph({"nodes": nodes, "edges": list(edges)}, by_type), by_type))

    def edge(s, sh, t, th):
        return {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}

    # a VLM with only `context` wired cannot run; with an image it can
    nodes = [{"id": "t", "type": "text_input", "params": {}}, {"id": "v", "type": "vlm", "params": {"prompt": "p"}}]
    assert "connect one of its inputs" in lint_of(nodes, [edge("t", "text", "v", "context")])
    nodes.append({"id": "i", "type": "image_input", "params": {"file": "x"}})
    assert "connect one of" not in lint_of(nodes, [edge("i", "image", "v", "image")])
    # LLM fusion needs two candidates
    fus = [{"id": "t", "type": "text_input", "params": {}}, {"id": "t2", "type": "text_input", "params": {}},
           {"id": "f", "type": "llm_decision_fusion", "params": {}}]
    assert "at least two" in lint_of(fus, [edge("t", "text", "f", "a")])
    assert "at least two" not in lint_of(fus, [edge("t", "text", "f", "a"), edge("t2", "text", "f", "b")])
    # an empty LLM instruction or a first/majority fusion instruction is not a blocker; an empty VLM prompt is
    llm = [{"id": "t", "type": "text_input", "params": {}}, {"id": "l", "type": "llm", "params": {"instruction": ""}}]
    assert "is empty" not in lint_of(llm, [edge("t", "text", "l", "text")])
    assert "'Prompt' is empty" in lint_of([{"id": "v", "type": "vlm", "params": {"prompt": ""}}])
