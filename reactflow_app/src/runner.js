import { useCallback } from 'react';
import { useReactFlow } from '@xyflow/react';
import { runNodeApi } from './api';

export function useRunner() {
  const { getNode, getNodes, getEdges, updateNodeData } = useReactFlow();

  // `fresh` holds results produced earlier in the same run: the store's node lookup
  // only reflects updateNodeData after a render, so it can't be read back synchronously.
  const runOne = useCallback(async (id, fresh = new Map()) => {
    const node = getNode(id);
    if (!node) return false;
    const inputs = {};
    for (const e of getEdges()) {
      if (e.target !== id) continue;
      const upstream = fresh.get(e.source) ?? getNode(e.source)?.data.result;
      const value = upstream?.[e.sourceHandle];
      if (value !== undefined) inputs[e.targetHandle] = value;
    }
    updateNodeData(id, { status: 'running', error: '' });
    try {
      const result = await runNodeApi(node.data.type, node.data.params, inputs);
      fresh.set(id, result);
      updateNodeData(id, { status: 'done', error: '', result });
      return true;
    } catch (err) {
      updateNodeData(id, { status: 'error', error: err.message || String(err) });
      return false;
    }
  }, [getNode, getEdges, updateNodeData]);

  const runAll = useCallback(async () => {
    const nodes = getNodes();
    const edges = getEdges();
    const indeg = new Map(nodes.map((n) => [n.id, 0]));
    for (const e of edges) {
      if (indeg.has(e.target) && indeg.has(e.source)) indeg.set(e.target, indeg.get(e.target) + 1);
    }
    const queue = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const e of edges) {
        if (e.source !== id || !indeg.has(e.target)) continue;
        indeg.set(e.target, indeg.get(e.target) - 1);
        if (indeg.get(e.target) === 0) queue.push(e.target);
      }
    }
    if (order.length < nodes.length) return false; // cycle
    const fresh = new Map();
    for (const id of order) {
      if (!(await runOne(id, fresh))) return false;
    }
    return true;
  }, [getNodes, getEdges, runOne]);

  return { runOne, runAll };
}
