import { useContext, useState } from 'react';
import { Handle, Position, useReactFlow } from '@xyflow/react';
import { SpecsContext } from './SpecsContext';
import { API_BASE, uploadFile } from './api';
import { useRunner } from './runner';
import { typeColor } from './portColors';

const inputCls = 'nodrag w-full rounded-md border border-slate-300 px-2 py-1 text-xs focus:border-indigo-500 focus:outline-none';

function Port({ port, side }) {
  const isIn = side === 'in';
  return (
    <div className={`relative -mx-4 flex items-center px-4 py-0.5 text-[11px] text-slate-500 ${isIn ? '' : 'justify-end'}`}>
      <Handle
        id={port.name}
        type={isIn ? 'target' : 'source'}
        position={isIn ? Position.Left : Position.Right}
        style={{ background: typeColor(port.type) }}
        className="!w-3 !h-3"
      />
      <span>{port.name} · {port.type}{port.optional ? ' (opt)' : ''}</span>
    </div>
  );
}

function ParamField({ param, data, onChange, onFile, uploading, uploadError }) {
  const value = data.params?.[param.name];
  const label = <span className="mb-0.5 block text-[11px] font-medium text-slate-600">{param.label}</span>;
  if (param.kind === 'file') {
    const name = data.fileNames?.[param.name];
    return (
      <div>
        {label}
        <label className="nodrag flex cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-slate-300 px-2 py-3 text-xs text-slate-500 hover:border-indigo-400 hover:text-indigo-600">
          <input type="file" hidden accept={param.accept} onChange={(e) => onFile(param, e)} />
          {uploading ? 'Uploading…' : name || 'Click to upload'}
        </label>
        {uploadError && <p role="alert" className="mt-1 text-xs text-red-600">{uploadError}</p>}
      </div>
    );
  }
  if (param.kind === 'textarea') {
    return (
      <label className="block">
        {label}
        <textarea rows={3} className={`${inputCls} nowheel resize-y`} value={value ?? ''} onChange={(e) => onChange(param.name, e.target.value)} />
      </label>
    );
  }
  if (param.kind === 'number') {
    return (
      <label className="block">
        {label}
        <input
          type="number" step={param.step ?? 'any'} className={inputCls} value={value ?? ''}
          onChange={(e) => onChange(param.name, e.target.value === '' ? '' : Number(e.target.value))}
        />
      </label>
    );
  }
  if (param.kind === 'select') {
    return (
      <label className="block">
        {label}
        <select className={`${inputCls} nodrag`} value={value ?? param.default} onChange={(e) => onChange(param.name, e.target.value)}>
          {param.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="block">
      {label}
      <input type={param.kind === 'password' ? 'password' : 'text'} className={inputCls} value={value ?? ''} onChange={(e) => onChange(param.name, e.target.value)} />
    </label>
  );
}

const clamp = (t, n = 300) => {
  const s = String(t ?? '');
  return s.length > n ? `${s.slice(0, n)}…` : s;
};

const fmtTime = (sec) => {
  const t = Math.max(0, Math.floor(Number(sec) || 0));
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
};

function Preview({ port, value }) {
  if (value === undefined || value === null) return null;
  let body = null;
  if (port.type === 'text') {
    body = <pre className="nowheel max-h-32 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-2 text-xs text-slate-700">{String(value)}</pre>;
  } else if (port.type === 'image') {
    body = <img src={value} alt={port.name} className="max-h-40 w-full rounded object-contain" />;
  } else if (port.type === 'video') {
    body = <p className="text-xs text-slate-600">{value.name} · {Number(value.duration).toFixed(1)}s · {value.width}×{value.height} · {value.fps}fps</p>;
  } else if (port.type === 'chunks') {
    const arr = Array.isArray(value) ? value : [];
    body = (
      <div>
        <p className="text-xs text-slate-600">{arr.length} chunks</p>
        <div className="nowheel mt-1 flex gap-1 overflow-x-auto">
          {arr.slice(0, 6).map((c) => (
            <figure key={c.index} className="shrink-0">
              {c.frames?.[0] && <img src={c.frames[0]} alt={`chunk ${c.index}`} className="h-12 rounded" />}
              <figcaption className="text-[10px] text-slate-500">{Number(c.start).toFixed(0)}–{Number(c.end).toFixed(0)}s</figcaption>
            </figure>
          ))}
        </div>
      </div>
    );
  } else if (port.type === 'images') {
    const arr = Array.isArray(value) ? value : [];
    body = (
      <div>
        <p className="text-xs text-slate-600">{arr.length} images</p>
        <div className="nowheel mt-1 flex gap-1 overflow-x-auto">
          {arr.slice(0, 6).map((item, i) => (
            <img
              key={`${item.path}-${i}`} loading="lazy" src={`${API_BASE}/api/media?path=${encodeURIComponent(item.path)}`}
              alt={item.name} title={item.name} className="h-14 w-14 shrink-0 rounded object-cover"
            />
          ))}
        </div>
      </div>
    );
  } else if (port.type === 'videos') {
    const arr = Array.isArray(value) ? value : [];
    body = (
      <div>
        <p className="text-xs text-slate-600">{arr.length} videos</p>
        {arr.slice(0, 5).map((v, i) => (
          <p key={`${v.path}-${i}`} className="truncate text-xs text-slate-600" title={v.name}>{v.name} · {Number(v.duration).toFixed(1)}s</p>
        ))}
        {arr.length > 5 && <p className="text-xs text-slate-400">+{arr.length - 5} more</p>}
      </div>
    );
  } else if (port.type === 'results') {
    const arr = Array.isArray(value) ? value : [];
    body = (
      <div>
        <p className="text-xs text-slate-600">{arr.length} results</p>
        <div className="nowheel mt-1 max-h-40 space-y-1 overflow-auto rounded bg-slate-50 p-2 text-xs text-slate-700">
          {arr.slice(0, 20).map((r, i) => (
            <div key={`${r.name}-${i}`}>
              <b>{r.name}</b>
              {r.segments?.length ? (
                r.segments.slice(0, 20).map((s, j) => (
                  <p key={j} className="whitespace-pre-wrap">[{fmtTime(s.start)}–{fmtTime(s.end)}] {clamp(s.text)}</p>
                ))
              ) : (
                <p className="whitespace-pre-wrap">{clamp(r.text)}</p>
              )}
            </div>
          ))}
          {arr.length > 20 && <p className="text-slate-400">+{arr.length - 20} more</p>}
        </div>
      </div>
    );
  } else if (port.type === 'detections') {
    const arr = Array.isArray(value) ? value : [];
    body = (
      <div>
        <p className="text-xs text-slate-600">{arr.reduce((n, d) => n + d.boxes.length, 0)} boxes in {arr.length} images</p>
        <div className="nowheel mt-1 flex gap-1 overflow-x-auto">
          {arr.slice(0, 4).map((d, i) => (
            <div key={`${d.name}-${i}`} className="relative inline-block shrink-0">
              <img
                src={d.path ? `${API_BASE}/api/media?path=${encodeURIComponent(d.path)}` : d.image}
                alt={d.name} title={d.name} className="block h-24 rounded"
              />
              {d.boxes.map((b, j) => (
                <span
                  key={j} title={b.label} className="absolute border-2 border-fuchsia-500"
                  style={{ left: `${b.box[0] * 100}%`, top: `${b.box[1] * 100}%`, width: `${(b.box[2] - b.box[0]) * 100}%`, height: `${(b.box[3] - b.box[1]) * 100}%` }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  } else if (port.type === 'file') {
    body = (
      <p className="text-xs text-slate-600">
        Wrote {value.rows} rows{' '}
        <a
          href={`${API_BASE}/api/outputs/${encodeURIComponent(value.name)}`} download target="_blank" rel="noreferrer"
          className="nodrag text-indigo-600 underline"
        >{value.name}</a>
      </p>
    );
  } else {
    body = <pre className="nowheel max-h-32 overflow-auto text-xs">{JSON.stringify(value)}</pre>;
  }
  return (
    <div className="mt-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-400">{port.name}</p>
      {body}
    </div>
  );
}

export default function AiNode({ id, data }) {
  const specs = useContext(SpecsContext);
  const spec = specs[data.type];
  const { updateNodeData, deleteElements } = useReactFlow();
  const { runOne } = useRunner();
  const [uploading, setUploading] = useState({});
  const [uploadErrors, setUploadErrors] = useState({});

  if (!spec) {
    return <div className="w-80 rounded-xl border border-red-300 bg-white p-4 text-sm text-red-600">Unknown node type: {data.type}</div>;
  }

  const setParam = (name, value) => updateNodeData(id, (n) => ({ params: { ...n.data.params, [name]: value } }));

  const onFile = async (param, e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setUploading((u) => ({ ...u, [param.name]: true }));
    setUploadErrors((u) => ({ ...u, [param.name]: '' }));
    try {
      const { path, name } = await uploadFile(file);
      updateNodeData(id, (n) => ({
        params: { ...n.data.params, [param.name]: path },
        fileNames: { ...n.data.fileNames, [param.name]: name },
      }));
    } catch (err) {
      setUploadErrors((u) => ({ ...u, [param.name]: err.message }));
    } finally {
      setUploading((u) => ({ ...u, [param.name]: false }));
    }
  };

  const field = (p) => (
    <ParamField
      key={p.name} param={p} data={data} onChange={setParam} onFile={onFile}
      uploading={uploading[p.name]} uploadError={uploadErrors[p.name]}
    />
  );
  const basic = spec.params.filter((p) => !p.advanced);
  const advanced = spec.params.filter((p) => p.advanced);
  const running = data.status === 'running';

  return (
    <div className="w-80 rounded-xl border border-slate-200 bg-white p-4 shadow-md">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-800">{spec.icon} {spec.label}</span>
        <button
          type="button" aria-label="Delete node" title="Delete node"
          className="nodrag rounded px-1 text-slate-400 hover:bg-slate-100 hover:text-red-600"
          onClick={() => deleteElements({ nodes: [{ id }] })}
        >✕</button>
      </div>

      {spec.inputs.map((p) => <Port key={p.name} port={p} side="in" />)}
      {spec.outputs.map((p) => <Port key={p.name} port={p} side="out" />)}

      <div className="mt-2 space-y-2">
        {basic.map(field)}
        {advanced.length > 0 && (
          <details className="nodrag">
            <summary className="cursor-pointer text-xs font-medium text-indigo-600">Model settings</summary>
            <div className="mt-2 space-y-2">{advanced.map(field)}</div>
          </details>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2">
        <button
          type="button" disabled={running} onClick={() => runOne(id)}
          className="nodrag rounded-md bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >Run node</button>
        <span className="text-xs text-slate-500">{data.status}</span>
      </div>
      {data.error && <p role="alert" className="mt-2 text-xs text-red-600">{data.error}</p>}
      {spec.outputs.map((p) => <Preview key={p.name} port={p} value={data.result?.[p.name]} />)}
    </div>
  );
}
