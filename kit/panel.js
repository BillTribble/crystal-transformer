// The settings pane: preset selector + custom preset saving/deletion + randomization +
// one row per param generated from params.js, plus Share URL, JSON Export/Import, Reset and Copy.

import { createColourPicker } from './colour-picker.js';

export const STATE_TAG = 'crystals/state@1';

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function noSuggestions(input) {
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.enterKeyHint = 'done';
  input.setAttribute('autocorrect', 'off');
  input.setAttribute('autocapitalize', 'off');
}

export function buildPanel({ slug, store, onEdited, presets, onPreset, onRandomize, onSkipTrack }) {
  const controls = document.getElementById('controls');
  const copyBtn = document.getElementById('copy');
  const shareBtn = document.getElementById('share-url');
  const exportBtn = document.getElementById('export-json');
  const importBtn = document.getElementById('import-json');
  const importFile = document.getElementById('import-file');
  const resetBtn = document.getElementById('reset');
  const statusEl = document.getElementById('status');
  const showBox = document.getElementById('show-changes');
  const rows = [];
  const picker = createColourPicker();

  function buildRow(p) {
    const wrap = el('div', 'row row-' + p.type);
    const label = el('label', 'lbl', '- '.repeat(p.depth) + p.label);
    const id = 'p-' + p.key;
    label.htmlFor = id;
    label.title = (p.desc ? p.desc + '\n' : '') + 'key: ' + p.key + ' · default: ' + p.default;
    label.addEventListener('click', (e) => e.preventDefault());
    label.addEventListener('dblclick', () => { store.setValue(p, p.base); sync(); onEdited(p.key); });
    wrap.appendChild(label);
    const mark = () => wrap.classList.toggle('changed', store.isChanged(p));

    let sync;
    if (p.type === 'bool') {
      const box = el('input');
      box.type = 'checkbox';
      box.id = id;
      box.addEventListener('change', () => { store.setValue(p, box.checked); sync(); onEdited(p.key); });
      wrap.appendChild(box);
      sync = () => { box.checked = !!p.value; mark(); };
    } else if (p.type === 'select') {
      const menu = el('select');
      menu.id = id;
      for (const o of p.options) menu.appendChild(el('option', null, o)).value = o;
      menu.addEventListener('change', () => { store.setValue(p, menu.value); sync(); onEdited(p.key); });
      wrap.appendChild(menu);
      sync = () => { menu.value = String(p.value); mark(); };
    } else if (p.type === 'color') {
      const pick = el('button', 'swatch');
      pick.type = 'button';
      pick.id = id;
      pick.setAttribute('aria-haspopup', 'dialog');
      pick.setAttribute('aria-expanded', 'false');
      const hex = el('input', 'num');
      hex.type = 'text';
      hex.maxLength = 7;
      noSuggestions(hex);
      pick.addEventListener('click', () => {
        if (picker.anchor === pick) { picker.close(); return; }
        picker.open(pick, String(p.value), (v) => {
          store.setValue(p, v);
          pick.style.background = hex.value = String(p.value);
          mark();
          onEdited(p.key);
        });
      });
      hex.addEventListener('change', () => { store.setValue(p, hex.value.trim().replace(/^#?/, '#')); sync(); onEdited(p.key); });
      wrap.append(pick, hex);
      sync = () => {
        pick.style.background = hex.value = String(p.value);
        if (picker.anchor === pick) picker.follow(String(p.value));
        mark();
      };
    } else {
      const track = el('span', 'track');
      const range = el('input');
      range.type = 'range';
      range.id = id;
      range.step = String(p.step);
      const home = el('i', 'home');
      home.setAttribute('aria-hidden', 'true');
      const rerange = el('button', 'rerange', '↔');
      rerange.type = 'button';
      rerange.title = 'Re-centre this slider\'s range on its value.';
      const fixedEnds = p.codeMin >= 0 && (p.codeMax === 1 || p.codeMax === 100);
      rerange.addEventListener('click', () => {
        const w = p.max - p.min, v = Number(p.value);
        let lo = v - w / 2, hi = v + w / 2;
        if (p.codeMin >= 0 && lo < 0) { lo = 0; hi = w; }
        store.setRange(p, Number(lo.toFixed(p.decimals)), Number(hi.toFixed(p.decimals)));
        sync();
        onEdited(p.key);
      });
      track.append(range, home, rerange);
      const place = () => {
        range.min = String(p.min);
        range.max = String(p.max);
        const lo = p.min, hi = p.max, b = Number(p.base), v = Number(p.value);
        home.hidden = !store.isChanged(p) || b < lo || b > hi;
        home.style.left = 'calc(5px + (100% - 10px) * ' + ((b - lo) / (hi - lo)) + ')';
        const atMin = v <= lo && lo !== 0, atMax = v >= hi;
        rerange.hidden = fixedEnds || store.bypassed || !(atMin || atMax);
        rerange.classList.toggle('at-min', atMin);
      };
      const num = el('input', 'num');
      num.type = 'number';
      num.step = String(p.step);
      if (p.min != null && p.min >= 0) num.inputMode = Number.isInteger(p.step) ? 'numeric' : 'decimal';
      noSuggestions(num);
      range.addEventListener('input', () => { store.setValue(p, range.value); num.value = String(Number(Number(p.value).toFixed(p.decimals))); mark(); place(); onEdited(p.key); });
      num.addEventListener('change', () => { store.setValue(p, num.value); sync(); onEdited(p.key); });
      wrap.append(track, num);
      sync = () => {
        place();
        if (document.activeElement !== range) range.value = String(p.value);
        if (document.activeElement !== num) num.value = String(Number(Number(p.value).toFixed(p.decimals)));
        mark();
      };
    }
    wrap.title = p.desc || '';
    sync();
    rows.push({ p, wrap, sync });
    return wrap;
  }

  // ---- Credit attribution at the top of Settings (post-014072)
  const creditBox = el('div', 'credit-section credit-top');
  const creditText = el('span', 'credit-label', 'Credit: ');
  const creditLink = el('a', 'credit-link', 'Jack Bush (@jackbush)');
  creditLink.id = 'jack-bush-link';
  creditLink.href = 'https://github.com/jackbush';
  creditLink.target = '_blank';
  creditLink.rel = 'noopener noreferrer';
  const creditSep = el('span', 'credit-label', ' · ');
  const repoLink = el('a', 'credit-link', 'jackbush/glimmers');
  repoLink.href = 'https://github.com/jackbush/glimmers';
  repoLink.target = '_blank';
  repoLink.rel = 'noopener noreferrer';
  creditBox.append(creditText, creditLink, creditSep, repoLink);
  controls.appendChild(creditBox);

  // ---- Preset & Randomize section at top of controls
  const presetBox = el('div', 'preset-section');
  const presetRow = el('div', 'row row-select row-preset');
  const presetLabel = el('label', 'lbl', 'Preset');
  presetLabel.htmlFor = 'p--preset';
  const presetMenu = el('select');
  presetMenu.id = 'p--preset';
  presetRow.append(presetLabel, presetMenu);

  const actionRow = el('div', 'preset-actions');
  const randBtn = el('button', 'mini-btn', 'Randomize');
  randBtn.type = 'button';
  randBtn.id = 'randomize-btn';
  randBtn.title = 'Generate a random crystal habit, colour, and optical setup (R)';

  const openSaveBtn = el('button', 'mini-btn', '+ Save preset');
  openSaveBtn.type = 'button';
  openSaveBtn.id = 'open-save-preset';
  openSaveBtn.title = 'Save current settings as a named preset in this browser';

  const delPresetBtn = el('button', 'mini-btn mini-btn-danger', 'Delete');
  delPresetBtn.type = 'button';
  delPresetBtn.id = 'delete-preset-btn';
  delPresetBtn.hidden = true;
  delPresetBtn.title = 'Delete this saved custom preset';

  actionRow.append(randBtn, openSaveBtn, delPresetBtn);

  // Inline Save Preset form (no window.prompt!)
  const saveForm = el('div', 'save-preset-form');
  saveForm.id = 'save-preset-form';
  saveForm.hidden = true;
  const nameInput = el('input', 'num save-preset-input');
  nameInput.type = 'text';
  nameInput.id = 'preset-name-input';
  nameInput.placeholder = 'Preset name…';
  noSuggestions(nameInput);
  const confirmSaveBtn = el('button', 'mini-btn mini-btn-primary', 'Save');
  confirmSaveBtn.type = 'button';
  confirmSaveBtn.id = 'confirm-save-preset';
  const cancelSaveBtn = el('button', 'mini-btn', 'Cancel');
  cancelSaveBtn.type = 'button';
  cancelSaveBtn.id = 'cancel-save-preset';
  saveForm.append(nameInput, confirmSaveBtn, cancelSaveBtn);

  presetBox.append(presetRow, actionRow, saveForm);
  controls.appendChild(presetBox);

  // ---- Settings Search Filter Box (post-014048)
  const searchSection = el('div', 'search-section');
  const searchWrap = el('div', 'settings-search-wrap');
  const searchInput = el('input', 'num settings-search-input');
  searchInput.type = 'search';
  searchInput.id = 'settings-search';
  searchInput.placeholder = 'Filter settings...';
  searchInput.setAttribute('aria-label', 'Filter settings');
  noSuggestions(searchInput);
  const searchClearBtn = el('button', 'settings-search-clear', '×');
  searchClearBtn.type = 'button';
  searchClearBtn.id = 'settings-search-clear';
  searchClearBtn.title = 'Clear filter (Esc)';
  searchClearBtn.setAttribute('aria-label', 'Clear filter');
  searchClearBtn.hidden = true;
  searchWrap.append(searchInput, searchClearBtn);
  searchSection.appendChild(searchWrap);
  controls.appendChild(searchSection);

  let lastMenuKey = '';
  function rebuildPresetMenu(force = false) {
    const hudPresetName = document.getElementById('preset-name') || document.getElementById('hud-preset-name');
    if (hudPresetName) {
      hudPresetName.textContent = store.preset || 'Amethyst Cathedral';
    }
    delPresetBtn.hidden = !(store.preset && store.customPresets && store.customPresets[store.preset]);
    if (!force && document.activeElement === presetMenu) return;
    const customNames = Object.keys(store.customPresets || {});
    const menuKey = `${store.preset}|${store.changedCount()}|${customNames.join(',')}`;
    if (!force && menuKey === lastMenuKey) return;
    lastMenuKey = menuKey;

    presetMenu.innerHTML = '';
    const placeholder = el('option', null, store.preset ? `${store.preset}${store.changedCount() ? ' *' : ''}` : 'Custom / Default');
    placeholder.value = '';
    presetMenu.appendChild(placeholder);

    const curatedNames = presets ? Object.keys(presets) : [];
    if (curatedNames.length) {
      const grp = el('optgroup');
      grp.label = 'Curated Gems';
      for (const n of curatedNames) {
        const opt = el('option', null, n);
        opt.value = n;
        if (store.preset === n && store.changedCount() === 0) opt.selected = true;
        grp.appendChild(opt);
      }
      presetMenu.appendChild(grp);
    }

    if (customNames.length) {
      const grp = el('optgroup');
      grp.label = 'Saved Presets';
      for (const n of customNames) {
        const opt = el('option', null, n);
        opt.value = n;
        if (store.preset === n && store.changedCount() === 0) opt.selected = true;
        grp.appendChild(opt);
      }
      presetMenu.appendChild(grp);
    }
  }

  presetMenu.addEventListener('change', () => {
    if (presetMenu.value && onPreset) onPreset(presetMenu.value);
    rebuildPresetMenu();
  });

  randBtn.addEventListener('click', () => {
    onRandomize?.();
  });

  openSaveBtn.addEventListener('click', () => {
    saveForm.hidden = !saveForm.hidden;
    if (!saveForm.hidden) {
      nameInput.value = store.preset && store.customPresets?.[store.preset] ? store.preset : '';
      nameInput.focus();
    }
  });

  cancelSaveBtn.addEventListener('click', () => {
    saveForm.hidden = true;
  });

  function commitSavePreset() {
    const raw = nameInput.value.trim() || ('Custom Gem ' + (Object.keys(store.customPresets).length + 1));
    const savedName = store.saveCustomPreset(raw);
    if (savedName) {
      saveForm.hidden = true;
      rebuildPresetMenu();
      panel.refreshStatus();
    }
  }

  confirmSaveBtn.addEventListener('click', commitSavePreset);
  nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commitSavePreset(); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); saveForm.hidden = true; }
  });

  delPresetBtn.addEventListener('click', () => {
    if (store.preset && store.customPresets?.[store.preset]) {
      store.deleteCustomPreset(store.preset);
      rebuildPresetMenu();
      panel.refreshStatus();
    }
  });

  const groupEntries = [];
  for (const group of store.groups) {
    const det = el('details', 'group');
    det.open = true;
    det.appendChild(el('summary', null, group.group));
    const groupRows = [];
    for (const p of group.params) {
      const rowWrap = buildRow(p);
      det.appendChild(rowWrap);
      groupRows.push(rows[rows.length - 1]);
      if (p.key === 'soundTrack') {
        const nowRow = el('div', 'row row-track-now');
        nowRow.id = 'current-track-row';
        const nowLbl = el('span', 'lbl', 'Current track');
        const nowName = el('span', 'current-track-name', String(p.value || ''));
        nowName.id = 'current-track-name';
        nowName.title = String(p.value || '');
        const skipBtn = el('button', 'mini-btn skip-track-btn', 'Next track →');
        skipBtn.type = 'button';
        skipBtn.id = 'skip-track-btn';
        skipBtn.title = 'Skip forward to the next Enso ambient track';
        skipBtn.addEventListener('click', () => {
          onSkipTrack?.();
          nowName.textContent = String(p.value || '');
          nowName.title = String(p.value || '');
        });
        nowRow.append(nowLbl, nowName, skipBtn);
        det.appendChild(nowRow);
        const origSync = rows[rows.length - 1].sync;
        rows[rows.length - 1].sync = () => {
          origSync();
          nowName.textContent = String(p.value || '');
          nowName.title = String(p.value || '');
        };
      }
    }
    controls.appendChild(det);
    groupEntries.push({ group, det, rows: groupRows });
  }

  const emptySearchEl = el('div', 'settings-search-empty', 'No matching settings');
  emptySearchEl.id = 'settings-search-empty';
  emptySearchEl.hidden = true;
  controls.appendChild(emptySearchEl);

  function applySearchFilter() {
    const q = searchInput.value.trim().toLowerCase();
    searchClearBtn.hidden = q.length === 0;
    if (!q) {
      for (const g of groupEntries) {
        g.det.hidden = false;
        for (const r of g.rows) {
          r.wrap.hidden = !store.isShown(r.p);
        }
      }
      emptySearchEl.hidden = true;
      return;
    }

    let totalMatches = 0;
    for (const g of groupEntries) {
      const groupMatch = g.group.group.toLowerCase().includes(q);
      let groupMatches = 0;
      for (const r of g.rows) {
        const p = r.p;
        const match = groupMatch ||
          p.label.toLowerCase().includes(q) ||
          p.key.toLowerCase().includes(q) ||
          (p.desc && p.desc.toLowerCase().includes(q));
        r.wrap.hidden = !match;
        if (match) groupMatches++;
      }
      g.det.hidden = groupMatches === 0;
      if (groupMatches > 0) {
        g.det.open = true;
        totalMatches += groupMatches;
      }
    }
    emptySearchEl.hidden = totalMatches > 0;
  }

  searchInput.addEventListener('input', applySearchFilter);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (searchInput.value) {
        e.preventDefault();
        e.stopPropagation();
        searchInput.value = '';
        applySearchFilter();
      }
    } else {
      e.stopPropagation();
    }
  });
  searchClearBtn.addEventListener('click', () => {
    searchInput.value = '';
    applySearchFilter();
    searchInput.focus();
  });

  const panel = {
    syncAll() {
      for (const r of rows) r.sync();
      rebuildPresetMenu();
    },
    refreshStatus() {
      applySearchFilter();
      const n = store.changedCount();
      statusEl.textContent = n === 0 ? 'No changes' : n + (n === 1 ? ' change' : ' changes');
      resetBtn.hidden = n === 0 && !store.preset;
      showBox.hidden = n === 0;
      showBox.checked = !store.bypassed;
      controls.classList.toggle('bypassed', store.bypassed);
      rebuildPresetMenu();
    },
  };

  showBox.addEventListener('change', () => {
    store.setBypass(!showBox.checked);
    panel.syncAll();
    onEdited();
  });

  function buildState() {
    const changed = {};
    for (const p of store.params) {
      if (!store.isChanged(p) || !store.isShown(p)) continue;
      changed[p.key] = { from: p.base, to: store.edited(p) };
      if (p.min !== p.codeMin || p.max !== p.codeMax) changed[p.key].range = [p.min, p.max];
    }
    return JSON.stringify({ glimmers: STATE_TAG, sketch: slug, ...(store.preset ? { preset: store.preset } : {}), values: store.exportSnapshot(), changed }, null, 2);
  }

  async function writeClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      const ta = el('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none';
      document.body.appendChild(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch { /* ignore */ }
      ta.remove();
      return ok;
    }
  }

  let copyTimer = 0;
  copyBtn?.addEventListener('click', async () => {
    const text = buildState();
    const ok = await writeClipboard(text);
    if (!ok) console.log(text);
    copyBtn.textContent = ok ? 'Copied JSON' : 'Logged to console';
    clearTimeout(copyTimer);
    copyTimer = window.setTimeout(() => { copyBtn.textContent = 'Copy JSON'; }, 1800);
  });

  let shareTimer = 0;
  shareBtn?.addEventListener('click', async () => {
    const url = store.encodeShareUrl();
    const ok = await writeClipboard(url);
    shareBtn.textContent = ok ? 'Link copied!' : 'URL updated';
    clearTimeout(shareTimer);
    shareTimer = window.setTimeout(() => { shareBtn.textContent = 'Share URL'; }, 1800);
  });

  exportBtn?.addEventListener('click', () => {
    const text = buildState();
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    const slugName = (store.preset || 'crystal-preset').toLowerCase().replace(/[^a-z0-9]+/g, '-');
    a.download = `${slugName}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  importBtn?.addEventListener('click', () => {
    importFile?.click();
  });

  importFile?.addEventListener('change', async () => {
    const file = importFile.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const vals = data.values || {};
      if (data.changed) {
        for (const k in data.changed) vals[k] = data.changed[k].to ?? data.changed[k];
      }
      store.applyValues(vals, 0.5, data.preset || file.name.replace(/\.json$/i, ''));
      panel.syncAll();
      onEdited();
    } catch (err) {
      console.error('Failed to import preset JSON:', err);
    }
    importFile.value = '';
  });

  resetBtn.addEventListener('click', () => {
    store.resetAll();
    store.clearSaved();
    try { localStorage.removeItem('crystals_default_zoom'); } catch { /* ignore */ }
    const url = new URL(window.location.href);
    url.search = '';
    window.history.replaceState(null, '', url.toString());
    panel.syncAll();
    onEdited('reset');
  });

  rebuildPresetMenu();
  return panel;
}
