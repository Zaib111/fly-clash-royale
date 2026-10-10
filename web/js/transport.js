// Where the match runs: the Python server (play3d.py) if there is one, otherwise right here
// in the browser (engine/worker.js). Both deliver the same init info and snapshots.

const $ = (id) => document.getElementById(id);
const PLAYERS = ['human', 'fly', 'scripted', 'random'];

export async function connect() {
  const server = await fetch('api/init').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return server ? serverTransport(server) : browserTransport();
}

function serverTransport(info) {
  const post = (url, body) => fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then((r) => r.json());
  return {
    info,
    onSnapshot(cb, onError) {
      const stream = new EventSource('api/stream');
      stream.onmessage = (e) => cb(JSON.parse(e.data));
      stream.onerror = onError;
    },
    play: (slot, x, y) => post('api/play', { slot, x, y }),
    control: (cmd, value) => post('api/control', { cmd, value }),
    brainBuffer: async (seat) => (await fetch(`api/brain/${seat}`)).arrayBuffer(),
  };
}

// Ask which match to play (unless the URL already says, e.g. ?p1=human&p2=fly).
function chooseMatch() {
  const q = new URLSearchParams(location.search);
  const p1 = q.get('p1'), p2 = q.get('p2');
  if (PLAYERS.includes(p1) && PLAYERS.includes(p2)) return Promise.resolve([p1, p2]);
  $('menu').classList.remove('hidden');
  return new Promise((resolve) => {
    $('menu').querySelectorAll('button[data-p1]').forEach((b) => b.addEventListener('click', () => {
      const pick = [b.dataset.p1, b.dataset.p2];
      history.replaceState(null, '', `?p1=${pick[0]}&p2=${pick[1]}`);
      $('menu').classList.add('hidden');
      resolve(pick);
    }));
  });
}

async function browserTransport() {
  const players = await chooseMatch();
  const loading = $('loading');
  loading.classList.remove('hidden');
  const worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' });
  let snapCb = null;
  let brainBuf = null;
  const pending = new Map();
  let nextId = 1;
  const info = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'progress') $('loading-text').textContent = msg.text;
      else if (msg.type === 'init') resolve(msg.info);
      else if (msg.type === 'brainviz') brainBuf = msg.buffer;
      else if (msg.type === 'snap') snapCb && snapCb(msg.data);
      else if (msg.type === 'played') { pending.get(msg.id)?.({ ok: msg.ok }); pending.delete(msg.id); }
      else if (msg.type === 'error') { $('loading-text').textContent = `Something went wrong: ${msg.text}`; reject(new Error(msg.text)); }
    };
    worker.postMessage({ cmd: 'init', players, model: new URL('model', location.href).href });
  });
  loading.classList.add('hidden');
  return {
    info,
    onSnapshot(cb) { snapCb = cb; },
    play: (slot, x, y) => new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      worker.postMessage({ cmd: 'play', id, slot, x, y });
    }),
    control: (cmd, value) => { worker.postMessage({ cmd: 'control', what: cmd, value }); return Promise.resolve({ ok: true }); },
    brainBuffer: async () => {
      while (!brainBuf) await new Promise((r) => setTimeout(r, 20));
      return brainBuf;
    },
  };
}
