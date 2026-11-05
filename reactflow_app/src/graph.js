import { typeColor } from './portColors';

// Canvas helpers shared by drag & drop, the chat agent and the "Arrange" button.
// Everything here is pure (arrays in, arrays out) so it can be used with setNodes/setEdges.

export const NODE_WIDTH = 320; // AiNode renders at w-80
const COL_GAP = 110;
const ROW_GAP = 40;

export const newNodeId = (type) => `${type}_${Math.random().toString(36).slice(2, 8)}`;

export function makeNode(spec, position, { id, params } = {}) {
  const defaults = {};
  for (const p of spec.params) defaults[p.name] = p.default;
  return {
    id: id ?? newNodeId(spec.type),
    type: 'ai',
    position,
    data: { type: spec.type, params: { ...defaults, ...params }, fileNames: {}, status: 'idle', error: '', result: {} },
  };
}

export const portType = (specs, node, handle, dir) =>
  specs[node?.data.type]?.[dir].find((p) => p.name === handle)?.type;

export function makeEdge(c, type) {
  return {
    source: c.source, sourceHandle: c.sourceHandle, target: c.target, targetHandle: c.targetHandle,
    id: `e_${c.source}_${c.sourceHandle}_${c.target}_${c.targetHandle}`,
    style: { stroke: typeColor(type), strokeWidth: 2 },
  };
}

// An input accepts a single connection: adding an edge replaces whatever fed that input before.
export function withEdge(edges, edge) {
  return [...edges.filter((e) => !(e.target === edge.target && e.targetHandle === edge.targetHandle)), edge];
}

/** What the agent is allowed to see of the canvas. Secrets and uploaded results stay in the browser. */
export function serializeGraph(nodes, edges, specs) {
  return {
    nodes: nodes.map((n) => {
      const spec = specs[n.data.type];
      const secret = new Set((spec?.params ?? []).filter((p) => p.kind === 'password').map((p) => p.name));
      const params = Object.fromEntries(Object.entries(n.data.params ?? {}).filter(([k]) => !secret.has(k)));
      return { id: n.id, type: n.data.type, params, status: n.data.status, error: n.data.error };
    }),
    edges: edges.map((e) => ({ source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle })),
  };
}

// ---------- layout ----------

/** Rough rendered height of a node, used until React Flow has measured it. */
export function estimateHeight(spec) {
  if (!spec) return 160;
  const fields = spec.params.filter((p) => !p.advanced);
  const fieldH = (p) => (p.kind === 'file' ? 86 : p.kind === 'textarea' ? 92 : 52);
  const advanced = spec.params.some((p) => p.advanced) ? 24 : 0;
  return 110 + 22 * (spec.inputs.length + spec.outputs.length) + fields.reduce((h, p) => h + fieldH(p), 0) + advanced;
}

const heightOf = (node, specs) => node.measured?.height ?? estimateHeight(specs[node.data.type]);

/** Layer index per node = longest path from a source. Bounded so a cycle can't loop forever. */
function depths(nodes, edges) {
  const depth = new Map(nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < nodes.length; pass++) {
    let changed = false;
    for (const e of edges) {
      if (!depth.has(e.source) || !depth.has(e.target)) continue;
      const d = depth.get(e.source) + 1;
      if (d > depth.get(e.target) && d < nodes.length) { depth.set(e.target, d); changed = true; }
    }
    if (!changed) break;
  }
  return depth;
}

/** Re-lay every node in left-to-right columns by pipeline stage, rows ordered to limit edge crossings. */
export function arrange(nodes, edges, specs) {
  const depth = depths(nodes, edges);
  const columns = new Map();
  for (const n of nodes) {
    const d = depth.get(n.id);
    if (!columns.has(d)) columns.set(d, []);
    columns.get(d).push(n);
  }
  const rowOf = new Map(); // node id -> vertical center, for ordering the next column
  const pos = new Map();
  for (const d of [...columns.keys()].sort((a, b) => a - b)) {
    const col = columns.get(d);
    const parentY = (n) => {
      const ys = edges.filter((e) => e.target === n.id && rowOf.has(e.source)).map((e) => rowOf.get(e.source));
      return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : Infinity;
    };
    col.sort((a, b) => parentY(a) - parentY(b)); // stable: ties keep canvas order
    let y = 0;
    for (const n of col) {
      pos.set(n.id, { x: d * (NODE_WIDTH + COL_GAP), y });
      rowOf.set(n.id, y + heightOf(n, specs) / 2);
      y += heightOf(n, specs) + ROW_GAP;
    }
  }
  return nodes.map((n) => ({ ...n, position: pos.get(n.id) }));
}

/**
 * Place only the nodes in `newIds`, leaving the user's own arrangement alone:
 * one column right of their furthest parent, below whatever already sits in that column.
 */
export function placeNew(nodes, edges, newIds, specs) {
  const placed = new Map(nodes.filter((n) => !newIds.has(n.id)).map((n) => [n.id, n]));
  const depth = depths(nodes, edges);
  const left = placed.size ? Math.min(...[...placed.values()].map((n) => n.position.x)) : 0;
  const colW = NODE_WIDTH + COL_GAP;
  const todo = nodes.filter((n) => newIds.has(n.id)).sort((a, b) => depth.get(a.id) - depth.get(b.id));
  for (const n of todo) {
    const parents = edges.filter((e) => e.target === n.id && placed.has(e.source)).map((e) => placed.get(e.source));
    const x = parents.length ? Math.max(...parents.map((p) => p.position.x)) + colW : left;
    // anything that horizontally overlaps this box, even a hand-placed node that is not in a tidy column
    const sameCol = [...placed.values()].filter((p) => Math.abs(p.position.x - x) < NODE_WIDTH);
    const y = sameCol.length ? Math.max(...sameCol.map((p) => p.position.y + heightOf(p, specs))) + ROW_GAP : 0;
    placed.set(n.id, { ...n, position: { x, y } });
  }
  return nodes.map((n) => placed.get(n.id));
}

// ---------- applying agent actions ----------

/**
 * Apply the (already server-validated) actions to the canvas.
 * Actions that refer to nodes that no longer exist (the user may have edited while waiting) are skipped.
 * Returns { nodes, edges, summary } where summary counts what changed.
 */
export function applyActions(nodes, edges, actions, specs) {
  const summary = { added: 0, connected: 0, configured: 0, removed: 0, arranged: false, cleared: false };
  const added = new Set();
  let shouldArrange = false;
  const find = (id) => nodes.find((n) => n.id === id);

  for (const a of actions) {
    switch (a.op) {
      case 'add_node': {
        const spec = specs[a.type];
        if (!spec || find(a.id)) break;
        nodes = [...nodes, makeNode(spec, { x: 0, y: 0 }, { id: a.id, params: a.params })];
        added.add(a.id);
        summary.added++;
        break;
      }
      case 'connect': {
        const src = find(a.source);
        const dst = find(a.target);
        if (!src || !dst) break;
        const type = portType(specs, src, a.sourceHandle, 'outputs');
        if (!type || type !== portType(specs, dst, a.targetHandle, 'inputs')) break;
        edges = withEdge(edges, makeEdge(a, type));
        summary.connected++;
        break;
      }
      case 'disconnect':
        edges = edges.filter((e) => !(e.source === a.source && e.sourceHandle === a.sourceHandle && e.target === a.target && e.targetHandle === a.targetHandle));
        break;
      case 'set_params':
        if (!find(a.id)) break;
        nodes = nodes.map((n) => (n.id === a.id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...a.params } } } : n));
        summary.configured++;
        break;
      case 'remove_node':
        if (!find(a.id)) break;
        nodes = nodes.filter((n) => n.id !== a.id);
        edges = edges.filter((e) => e.source !== a.id && e.target !== a.id);
        added.delete(a.id);
        summary.removed++;
        break;
      case 'clear':
        nodes = [];
        edges = [];
        added.clear();
        summary.cleared = true;
        break;
      case 'arrange':
        shouldArrange = true;
        break;
      default:
    }
  }
  if (shouldArrange) {
    nodes = arrange(nodes, edges, specs);
    summary.arranged = true;
  } else if (added.size) {
    nodes = placeNew(nodes, edges, added, specs);
  }
  return { nodes, edges, summary };
}

export function describeSummary(s) {
  const bits = [];
  if (s.cleared) bits.push('cleared the canvas');
  if (s.added) bits.push(`added ${s.added} node${s.added > 1 ? 's' : ''}`);
  if (s.connected) bits.push(`${s.connected} connection${s.connected > 1 ? 's' : ''}`);
  if (s.configured) bits.push(`configured ${s.configured}`);
  if (s.removed) bits.push(`removed ${s.removed}`);
  if (s.arranged) bits.push('arranged');
  return bits.join(' · ');
}
