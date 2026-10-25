export const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? 'http://localhost:7860' : '');
export const API_BASE = API_URL;

const detailText = (d) => (typeof d === 'string' ? d : JSON.stringify(d));

async function request(path, init) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error('Cannot reach the backend. Run `pixi run dev` in gradio_app.');
  }
  if (!res.ok) {
    let msg = `Server returned ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail !== undefined) msg = detailText(body.detail);
    } catch { /* non-JSON error body */ }
    throw new Error(msg);
  }
  return res.json();
}

export const fetchSpecs = () => request('/api/nodes');

export function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  return request('/api/upload', { method: 'POST', body: form });
}

export async function runNodeApi(type, params, inputs) {
  const json = await request('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type, params, inputs }),
  });
  return json.outputs ?? {};
}
