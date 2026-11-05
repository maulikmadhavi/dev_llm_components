import { useCallback, useContext, useEffect, useRef, useState } from 'react';
import { useReactFlow } from '@xyflow/react';
import { SpecsContext } from './SpecsContext';
import { chatApi } from './api';
import { applyActions, describeSummary, serializeGraph } from './graph';

const SETTINGS_KEY = 'flow-agent-settings';
const SUGGESTIONS = [
  'Describe every image in a folder and save the results to a CSV',
  'Find objects in an image and crop them',
  'Summarize a PDF with an LLM',
  'Review my flow and recommend improvements',
];

const loadSettings = () => {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) ?? {}; } catch { return {}; }
};
const saveSettings = (s) => {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch { /* private mode: keep in memory only */ }
};

const inputCls = 'w-full rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-indigo-500 focus:outline-none';

function Message({ m }) {
  const mine = m.role === 'user';
  return (
    <div className={`flex ${mine ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[92%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm ${
          mine ? 'bg-indigo-600 text-white' : m.error ? 'border border-red-200 bg-red-50 text-red-700' : 'bg-slate-100 text-slate-800'
        }`}
      >
        {m.content}
        {m.applied && <p className="mt-1.5 text-[11px] font-medium text-indigo-600">✓ {m.applied}</p>}
        {m.notes?.length > 0 && (
          <ul className="mt-1.5 list-disc pl-4 text-[11px] text-amber-700">{m.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        )}
        {m.issues?.length > 0 && (
          <div className="mt-1.5 rounded-lg bg-white/70 p-2 text-[11px] text-slate-600">
            <p className="font-semibold text-slate-700">Still to do</p>
            <ul className="list-disc pl-4">{m.issues.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatPanel({ onClose }) {
  const specs = useContext(SpecsContext);
  const { getNodes, getEdges, setNodes, setEdges, fitView } = useReactFlow();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(loadSettings);
  const endRef = useRef(null);
  // Each request gets a turn number; a reply only counts while its turn is still the current one.
  const turn = useRef(0);
  const abortRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }); }, [messages, busy]);

  const cancelPending = useCallback(() => {
    turn.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);
  // Closing the panel unmounts it: drop the pending request so it cannot edit the canvas unseen.
  useEffect(() => cancelPending, [cancelPending]);

  const newChat = () => {
    cancelPending();
    setMessages([]);
    setBusy(false);
  };

  const updateSettings = (key, value) => setSettings((s) => {
    const next = { ...s, [key]: value };
    saveSettings(next);
    return next;
  });

  const send = useCallback(async (text) => {
    const content = text.trim();
    if (!content || busy) return;
    const history = [...messages, { role: 'user', content }];
    cancelPending();
    const id = turn.current;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setMessages(history);
    setInput('');
    setBusy(true);
    try {
      const wire = history.filter((m) => !m.error).map(({ role, content: c }) => ({ role, content: c }));
      const res = await chatApi(wire, serializeGraph(getNodes(), getEdges(), specs), settings, ctrl.signal);
      if (turn.current !== id) return; // chat was reset or closed while waiting
      // Re-read the canvas: the user may have moved or deleted nodes while the model was thinking.
      const out = applyActions(getNodes(), getEdges(), res.actions ?? [], specs);
      if (res.actions?.length) {
        setNodes(out.nodes);
        setEdges(out.edges);
        if (out.summary.added || out.summary.arranged) setTimeout(() => fitView({ padding: 0.2, duration: 300 }), 60);
      }
      setMessages((m) => [...m, {
        role: 'assistant', content: res.reply, applied: describeSummary(out.summary), notes: res.notes, issues: res.issues,
      }]);
    } catch (err) {
      if (turn.current !== id) return;
      setMessages((m) => [...m, { role: 'assistant', error: true, content: err.message || String(err) }]);
    } finally {
      if (turn.current === id) setBusy(false);
    }
  }, [busy, messages, settings, specs, cancelPending, getNodes, getEdges, setNodes, setEdges, fitView]);

  return (
    <aside className="flex h-full w-96 shrink-0 flex-col border-l border-slate-200 bg-white" aria-label="Flow assistant">
      <header className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5">
        <h2 className="text-sm font-semibold text-slate-800">💬 Flow assistant</h2>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button type="button" onClick={newChat} className="text-xs text-slate-500 hover:text-slate-800">New chat</button>
          )}
          <button type="button" aria-label="Close assistant" onClick={onClose} className="rounded px-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700">✕</button>
        </div>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto p-4" role="log" aria-live="polite">
        {messages.length === 0 && (
          <div>
            <p className="mb-2 text-sm text-slate-600">
              Describe what you want to do. I will pick the boxes, connect and arrange them, and suggest improvements.
            </p>
            <div className="flex flex-col gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s} type="button" onClick={() => send(s)}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-left text-xs text-slate-700 hover:border-indigo-400 hover:bg-indigo-50"
                >{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} />)}
        {busy && <p className="text-xs text-slate-400">Thinking…</p>}
        <div ref={endRef} />
      </div>

      <details className="border-t border-slate-200 px-4 py-2">
        <summary className="cursor-pointer text-xs font-medium text-indigo-600">Assistant model</summary>
        <div className="mt-2 space-y-2">
          <label className="block text-[11px] text-slate-600">Base URL
            <input className={inputCls} placeholder="http://localhost:1234/v1" value={settings.base_url ?? ''} onChange={(e) => updateSettings('base_url', e.target.value)} />
          </label>
          <label className="block text-[11px] text-slate-600">Model
            <input className={inputCls} placeholder="qwen/qwen3-4b" value={settings.model ?? ''} onChange={(e) => updateSettings('model', e.target.value)} />
          </label>
          <p className="text-[11px] text-slate-400">Blank = the model server this backend is configured for. Other URLs need AGENT_ALLOW_CUSTOM_BASE_URL=1 on the backend.</p>
        </div>
      </details>

      <form
        className="flex items-end gap-2 border-t border-slate-200 p-3"
        onSubmit={(e) => { e.preventDefault(); send(input); }}
      >
        <textarea
          rows={2} value={input} disabled={busy} aria-label="Message the assistant"
          placeholder="e.g. add a detector after the image input"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(input); } }}
          className="nowheel flex-1 resize-none rounded-lg border border-slate-300 px-2 py-1.5 text-sm focus:border-indigo-500 focus:outline-none disabled:bg-slate-50"
        />
        <button
          type="submit" disabled={busy || !input.trim()}
          className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >Send</button>
      </form>
    </aside>
  );
}
