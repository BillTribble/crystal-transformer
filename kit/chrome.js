// The buttons around the sketch: settings cog, keyboard shortcuts, and UI visibility state.

const UI_KEY = 'crystals:ui';

function load(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function save(key, value) {
  try { localStorage.setItem(key, value); } catch { /* ignore */ }
}

export function setupChrome(persist, onPane, shortcuts = {}) {
  const body = document.body;
  const cog = document.getElementById('cog');
  const panel = document.getElementById('panel');
  const settings = persist && location.hash === '#settings';
  let ui = persist && (settings || load(UI_KEY) !== 'off') ? 'on' : 'off';
  let pane = settings ? 'open' : 'closed';

  function apply() {
    body.dataset.ui = ui;
    body.dataset.pane = pane;
    const open = ui === 'on' && pane === 'open';
    panel.setAttribute('aria-hidden', String(!open));
    panel.inert = !open;
    cog.setAttribute('aria-expanded', String(open));
  }
  function setUi(v) { ui = v; if (persist) save(UI_KEY, v); apply(); }
  function setPane(v) { pane = v; apply(); onPane?.(v === 'open'); }

  cog.addEventListener('click', () => setPane(pane === 'open' ? 'closed' : 'open'));

  window.addEventListener('keydown', (e) => {
    const t = e.target;
    const typing = !!t && (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' ||
      (t.tagName === 'INPUT' && t.type !== 'range' && t.type !== 'checkbox'));
    if (e.key === 'Escape') { if (pane === 'open') setPane('closed'); return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === ' ') {
      e.preventDefault();
      setUi(ui === 'on' ? 'off' : 'on');
    } else if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      shortcuts.onRandomize?.();
    } else if (e.key === 'g' || e.key === 'G') {
      e.preventDefault();
      shortcuts.onGrow?.();
    } else if (e.key === 'm' || e.key === 'M') {
      e.preventDefault();
      shortcuts.onSound?.();
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      setPane(pane === 'open' ? 'closed' : 'open');
    }
  });

  apply();
  return {
    get paneOpen() { return ui === 'on' && pane === 'open'; },
    openPane() { setUi('on'); setPane('open'); },
    closePane() { setPane('closed'); },
    togglePane() { setUi('on'); setPane(pane === 'open' ? 'closed' : 'open'); },
  };
}
