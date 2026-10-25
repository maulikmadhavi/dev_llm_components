import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow, ReactFlowProvider, Background, Controls, MiniMap, Panel, useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import AiNode from './AiNode';
import { typeColor } from './portColors';
import { SpecsContext } from './SpecsContext';
import { fetchSpecs } from './api';
import { useRunner } from './runner';

const nodeTypes = { ai: AiNode };
const CATEGORIES = ['Sources', 'Processing', 'Models'];

function Flow({ specs }) {
  const { addNodes, getNode, setNodes, setEdges, screenToFlowPosition } = useReactFlow();
  const { runAll } = useRunner();
  const [running, setRunning] = useState(false);

  const grouped = useMemo(() => {
    const list = Object.values(specs);
    const cats = [...new Set([...CATEGORIES, ...list.map((s) => s.category)])];
    return cats.map((c) => [c, list.filter((s) => s.category === c)]).filter(([, l]) => l.length);
  }, [specs]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('application/reactflow');
    const spec = specs[type];
    if (!spec) return;
    const params = {};
    for (const p of spec.params) params[p.name] = p.default;
    addNodes({
      id: `${type}_${Date.now().toString(36)}`,
      type: 'ai',
      position: screenToFlowPosition({ x: e.clientX, y: e.clientY }),
      data: { type, params, fileNames: {}, status: 'idle', error: '', result: {} },
    });
  }, [specs, addNodes, screenToFlowPosition]);

  const portType = useCallback((nodeId, handle, dir) => {
    const spec = specs[getNode(nodeId)?.data.type];
    return spec?.[dir].find((p) => p.name === handle)?.type;
  }, [specs, getNode]);

  const isValidConnection = useCallback((c) => {
    if (c.source === c.target) return false;
    const s = portType(c.source, c.sourceHandle, 'outputs');
    const t = portType(c.target, c.targetHandle, 'inputs');
    return !!s && s === t;
  }, [portType]);

  const onConnect = useCallback((c) => {
    const color = typeColor(portType(c.source, c.sourceHandle, 'outputs'));
    setEdges((eds) => [
      // an input accepts a single connection
      ...eds.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle)),
      { ...c, id: `e_${c.source}_${c.sourceHandle}_${c.target}_${c.targetHandle}`, style: { stroke: color, strokeWidth: 2 } },
    ]);
  }, [portType, setEdges]);

  const onRunAll = async () => {
    setRunning(true);
    try { await runAll(); } finally { setRunning(false); }
  };

  return (
    <div className="flex h-full w-full">
      <aside className="w-52 shrink-0 overflow-y-auto border-r border-slate-200 bg-white p-3">
        {grouped.map(([cat, list]) => (
          <section key={cat} className="mb-4">
            <h2 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">{cat}</h2>
            {list.map((s) => (
              <div
                key={s.type} draggable
                onDragStart={(e) => { e.dataTransfer.setData('application/reactflow', s.type); e.dataTransfer.effectAllowed = 'move'; }}
                className="mb-1 cursor-grab rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm text-slate-700 hover:border-indigo-400"
              >{s.icon} {s.label}</div>
            ))}
          </section>
        ))}
      </aside>
      <div className="h-full flex-1">
        <ReactFlow
          defaultNodes={[]} defaultEdges={[]} nodeTypes={nodeTypes}
          onDragOver={onDragOver} onDrop={onDrop}
          isValidConnection={isValidConnection} onConnect={onConnect}
        >
          <Panel position="top-right" className="flex gap-2">
            <button
              type="button" onClick={onRunAll} disabled={running}
              className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
            >▶ Run flow</button>
            <button
              type="button" onClick={() => { setNodes([]); setEdges([]); }}
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
            >Clear</button>
          </Panel>
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
    </div>
  );
}

function Loader() {
  const [specs, setSpecs] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    fetchSpecs()
      .then((list) => { if (!cancelled) setSpecs(Object.fromEntries(list.map((s) => [s.type, s]))); })
      .catch((err) => { if (!cancelled) setError(err.message); });
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return (
      <div role="alert" className="m-8 max-w-xl rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        {error}
        <p className="mt-1 text-red-600">Start the backend with <code>pixi run dev</code> in gradio_app, then reload.</p>
      </div>
    );
  }
  if (!specs) return <p className="m-8 text-sm text-slate-500">Loading nodes…</p>;
  return (
    <SpecsContext.Provider value={specs}>
      <ReactFlowProvider>
        <Flow specs={specs} />
      </ReactFlowProvider>
    </SpecsContext.Provider>
  );
}

export default function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Loader />
    </div>
  );
}
