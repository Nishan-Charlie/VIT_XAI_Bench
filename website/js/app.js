/* ═══════════════════════════════════════════════════════════════════════
   VIT-XAI-BENCH — the whole page is rendered from benchmark.json.

   There are no benchmark numbers in this file. Every value shown comes from
   the exported data, so rebuilding the results updates the site.
   ═══════════════════════════════════════════════════════════════════════ */
'use strict';

const DATA_URL = 'public/data/benchmark.json';
const META_URL = 'public/data/meta.json';

const state = { data: null, meta: null, arch: null };

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ─────────────────────────── data helpers ─────────────────────────── */

const metricSpec = (key) => state.data.metrics.find((m) => m.key === key);
const modelSpec = (key) => state.data.models.find((m) => m.key === key);
const methodSpec = (key) => state.data.methods.find((m) => m.key === key);
const measuredMetrics = () => state.data.metrics.filter((m) => m.measured);

function cellValue(model, method, metric) {
  const rec = state.data.records.find((r) => r.model === model && r.method === method);
  if (!rec) return null;
  const mv = rec.metrics[metric];
  return mv ? mv.mean : null;
}

function cellRecord(model, method) {
  return state.data.records.find((r) => r.model === model && r.method === method) || null;
}

/** Models present in the results, ordered so families stay contiguous. */
function activeModels(family = 'all') {
  return state.data.models.filter(
    (m) => m.in_results && (family === 'all' || m.family === family)
  );
}

function activeMethods(family = 'all') {
  return state.data.methods.filter(
    (m) => m.in_results && (family === 'all' || m.family === family)
  );
}

/** Format a measurement. Keeps a stable width so columns line up. */
const fmt = (v, digits = 3) =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits);

/* ───────────────────────────── tooltip ────────────────────────────── */

const tip = $('#tooltip');
function showTip(html, evt) {
  tip.innerHTML = html;
  tip.classList.add('on');
  tip.style.left = `${evt.clientX}px`;
  tip.style.top = `${evt.clientY}px`;
}
function hideTip() { tip.classList.remove('on'); }

function attachTip(node, htmlFn) {
  node.addEventListener('mouseenter', (e) => showTip(htmlFn(), e));
  node.addEventListener('mousemove', (e) => showTip(htmlFn(), e));
  node.addEventListener('mouseleave', hideTip);
  // Keyboard parity: focusing a cell shows the same information.
  node.addEventListener('focus', (e) => {
    const r = node.getBoundingClientRect();
    showTip(htmlFn(), { clientX: r.left + r.width / 2, clientY: r.top });
  });
  node.addEventListener('blur', hideTip);
}

/* ───────────────────── colour scale (sequential) ──────────────────── */

function rampColor(t) {
  const steps = ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5'].map(css);
  if (!Number.isFinite(t)) return css('--surface-2');
  const clamped = Math.min(1, Math.max(0, t));
  const idx = Math.min(steps.length - 1, Math.floor(clamped * (steps.length - 1) + 0.5));
  return steps[idx];
}

/** Normalise a value to 0..1 within [lo,hi], flipping when lower is better. */
function normalise(v, lo, hi, higherIsBetter) {
  if (v == null || hi === lo) return 0.5;
  const t = (v - lo) / (hi - lo);
  return higherIsBetter ? t : 1 - t;
}

/* ════════════════════════════ scope readout ═══════════════════════ */

function renderScope() {
  const host = $('#scope-readout');
  host.innerHTML = '';
  const dims = state.data.dimensions;
  const measured = dims.filter((d) => d.status === 'measured').length;

  const stats = [
    [String(state.meta.n_methods_in_results), 'Attribution methods'],
    [String(state.meta.n_models_in_results), 'Backbones'],
    [String(state.data.arch_families.length), 'Architecture families'],
    [`${measured}/${dims.length}`, 'Dimensions with data'],
    [String(state.data.records.length), 'Measured cells'],
  ];
  stats.forEach(([value, label], i) => {
    const stat = el('div', 'stat');
    const v = el('div', 'stat-value', value);
    // The dimensions tile is the one that can under-deliver; mark it plainly.
    if (i === 3 && measured < dims.length) v.classList.add('muted');
    stat.append(v, el('div', 'stat-label', label));
    host.append(stat);
  });
}

/* ═════════════════════ signature: transfer ribbon ═════════════════ */

function renderRibbon() {
  const metric = measuredMetrics().some((m) => m.key === 'pointing_game')
    ? 'pointing_game'
    : measuredMetrics()[0].key;
  const rows = state.data.transfer[metric] || [];
  $('#ribbon-metric-name').textContent = metricSpec(metric).label;

  const svg = $('#ribbon');
  svg.innerHTML = '';
  if (!rows.length) {
    $('#ribbon-verdict').textContent = 'No paired CNN / transformer results';
    return;
  }

  const moved = rows.filter((r) => r.rank_delta !== 0).length;
  $('#ribbon-verdict').textContent =
    `${moved} of ${rows.length} methods change rank`;

  // Height follows the row count so no line is ever clipped, and the SVG is
  // given a matching aspect-ratio so CSS scales it without cropping.
  const W = 900, padY = 34, step = 30;
  const H = padY * 2 + Math.max(1, rows.length - 1) * step;
  const leftX = 250, rightX = 650;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.aspectRatio = `${W} / ${H}`;

  const NS = 'http://www.w3.org/2000/svg';
  const mk = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  };
  const yFor = (rank) => padY + (rank - 1) * step;

  // column headers
  [['Rank on CNNs', leftX], ['Rank on transformers', rightX]].forEach(([label, x]) => {
    const t = mk('text', {
      x, y: 14, 'text-anchor': 'middle', fill: css('--ink-3'),
      'font-family': css('--font-data'), 'font-size': 10,
      'letter-spacing': '1.2', 'text-transform': 'uppercase',
    });
    t.textContent = label.toUpperCase();
    svg.append(t);
  });

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  rows.forEach((row, i) => {
    const y0 = yFor(row.cnn_rank);
    const y1 = yFor(row.transformer_rank);
    // Colour encodes direction of movement, not identity — this chart is about change.
    const color = row.rank_delta === 0 ? css('--neutral-mark')
      : row.rank_delta > 0 ? css('--s2') : css('--s3');

    const mid = (leftX + rightX) / 2;
    const path = mk('path', {
      d: `M ${leftX} ${y0} C ${mid} ${y0}, ${mid} ${y1}, ${rightX} ${y1}`,
      fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round',
      opacity: 0.9,
    });
    if (!reduce) {
      const len = 460;
      path.style.strokeDasharray = len;
      path.style.strokeDashoffset = len;
      path.style.animation = `ribbon-draw .8s ease ${i * 0.05}s forwards`;
    }
    svg.append(path);

    [[leftX, y0], [rightX, y1]].forEach(([cx, cy]) => {
      svg.append(mk('circle', {
        cx, cy, r: 4.5, fill: color, stroke: css('--surface-1'), 'stroke-width': 2,
      }));
    });

    const mkLabel = (x, y, anchor, text) => {
      const t = mk('text', {
        x, y: y + 3.5, 'text-anchor': anchor, fill: css('--ink-2'),
        'font-family': css('--font-prose'), 'font-size': 12,
      });
      t.textContent = text;
      return t;
    };
    const name = methodSpec(row.method).label;
    svg.append(mkLabel(leftX - 12, y0, 'end', name));
    svg.append(mkLabel(rightX + 12, y1, 'start', name));
  });

  // accessible table equivalent
  const table = $('#ribbon-table');
  table.innerHTML = '';
  const thead = el('thead');
  thead.innerHTML =
    '<tr><th>Method</th><th>Rank on CNNs</th><th>Rank on transformers</th><th>Change</th></tr>';
  const tbody = el('tbody');
  rows.forEach((r) => {
    const tr = el('tr');
    tr.innerHTML =
      `<td>${methodSpec(r.method).label}</td><td>${r.cnn_rank}</td>` +
      `<td>${r.transformer_rank}</td><td>${r.rank_delta > 0 ? '+' : ''}${r.rank_delta}</td>`;
    tbody.append(tr);
  });
  table.append(thead, tbody);

  $('#question-note').textContent =
    `In the published results, ${moved} of ${rows.length} methods occupy a different ` +
    `rank on transformer backbones than on CNNs when scored by ` +
    `${metricSpec(metric).label.toLowerCase()}.`;
}

/* ═══════════════════════════ result matrix ════════════════════════ */

function renderMatrix() {
  const metric = $('#matrix-metric').value;
  const family = $('#matrix-family').value;
  const methodFamily = $('#matrix-method-family').value;
  const sortBy = $('#matrix-sort').value;
  const spec = metricSpec(metric);

  $('#matrix-metric-note').textContent =
    `${spec.description} ${spec.higher_is_better ? 'Higher is better.' : 'Lower is better.'}`;

  const models = activeModels(family);
  let methods = activeMethods(methodFamily);

  const meanFor = (methodKey) => {
    const vals = models.map((m) => cellValue(m.key, methodKey, metric)).filter((v) => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  if (sortBy === 'score') {
    methods = methods.slice().sort((a, b) => {
      const va = meanFor(a.key), vb = meanFor(b.key);
      if (va == null) return 1;
      if (vb == null) return -1;
      return spec.higher_is_better ? vb - va : va - vb;
    });
  } else if (sortBy === 'name') {
    methods = methods.slice().sort((a, b) => a.label.localeCompare(b.label));
  } else {
    methods = methods.slice().sort(
      (a, b) => a.family.localeCompare(b.family) || a.label.localeCompare(b.label)
    );
  }

  const all = [];
  models.forEach((mo) => methods.forEach((me) => {
    const v = cellValue(mo.key, me.key, metric);
    if (v != null) all.push(v);
  }));
  const lo = Math.min(...all), hi = Math.max(...all);

  const host = $('#matrix-grid');
  host.innerHTML = '';
  if (!all.length) {
    host.append(el('p', 'note', 'No results for this combination of filters.'));
    return;
  }

  const table = el('table', 'matrix');

  // family band row
  const famRow = el('tr');
  famRow.append(el('th', '', ''));
  let i = 0;
  while (i < models.length) {
    let j = i;
    while (j + 1 < models.length && models[j + 1].family === models[i].family) j++;
    const th = el('th', 'fam-head', models[i].family_label);
    th.colSpan = j - i + 1;
    famRow.append(th);
    i = j + 1;
  }

  const headRow = el('tr');
  headRow.append(el('th', '', 'Method'));
  models.forEach((m) => {
    const th = el('th', 'model-head');
    th.append(el('span', '', m.label));
    headRow.append(th);
  });

  const thead = el('thead');
  thead.append(famRow, headRow);
  table.append(thead);

  const tbody = el('tbody');
  methods.forEach((me) => {
    const tr = el('tr');
    const nameCell = el('td', 'method-name');
    nameCell.append(document.createTextNode(me.label));
    nameCell.append(el('span', 'method-family-tag', me.family_label));
    tr.append(nameCell);

    models.forEach((mo) => {
      const v = cellValue(mo.key, me.key, metric);
      const td = el('td', 'cell');
      td.tabIndex = 0;
      if (v == null) {
        td.classList.add('empty');
        td.textContent = '·';
        const reason = me.vit_only && mo.family !== 'isotropic_vit'
          ? 'Not applicable: this method needs global CLS-token attention.'
          : 'Not run.';
        attachTip(td, () =>
          `<strong>${me.label} · ${mo.label}</strong><br>${reason}`);
      } else {
        const t = normalise(v, lo, hi, spec.higher_is_better);
        td.style.background = rampColor(t);
        td.style.color = t > 0.55 ? '#fff' : css('--ink');
        td.textContent = fmt(v, 2);
        const rec = cellRecord(mo.key, me.key);
        const mv = rec.metrics[metric];
        attachTip(td, () => {
          const n = mv.count != null ? `${mv.count} images` : 'n not recorded';
          const sd = mv.std != null ? ` ± ${fmt(mv.std, 3)}` : '';
          const run = rec.provenance.source_run || 'run not recorded';
          return `<strong>${me.label} · ${mo.label}</strong><br>` +
            `${spec.label}: ${fmt(mv.mean, 4)}${sd}<br>` +
            `${n}<br><span style="opacity:.7">source: ${run}</span>`;
        });
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  host.append(table);

  // legend
  const legend = $('#matrix-legend');
  legend.innerHTML = '';
  legend.append(el('span', '', spec.higher_is_better ? 'worse' : 'better'));
  const scale = el('div', 'legend-scale');
  for (let k = 0; k <= 5; k++) {
    const sw = el('div', 'legend-swatch');
    sw.style.background = rampColor(k / 5);
    scale.append(sw);
  }
  legend.append(scale);
  legend.append(el('span', '', spec.higher_is_better ? 'better' : 'worse'));
  legend.append(el('span', '', `· range ${fmt(lo, 2)} – ${fmt(hi, 2)} · “·” = not measured`));
}

/* ═════════════════════════ transfer panel ═════════════════════════ */

function renderTransfer() {
  const metric = $('#transfer-metric').value;
  const rows = state.data.transfer[metric] || [];
  const host = $('#transfer-panel');
  host.innerHTML = '';

  if (!rows.length) {
    host.append(el('p', 'note',
      'No method has results on both CNN and transformer backbones for this metric.'));
    return;
  }

  rows.forEach((r) => {
    const spec = methodSpec(r.method);
    const card = el('div', 'transfer-card');
    card.append(el('h3', '', spec.label));
    card.append(el('div', 'tc-family', spec.family_label));

    const ranks = el('div', 'tc-ranks');
    ranks.append(el('span', 'tc-rank', `#${r.cnn_rank}`));
    ranks.append(el('span', 'tc-arrow', '→'));
    ranks.append(el('span', 'tc-rank', `#${r.transformer_rank}`));
    const d = r.rank_delta;
    const delta = el('span',
      `tc-delta ${d === 0 ? 'same' : d > 0 ? 'down' : 'up'}`,
      d === 0 ? 'holds' : `${d > 0 ? '↓' : '↑'} ${Math.abs(d)}`);
    ranks.append(delta);
    card.append(ranks);

    const vals = el('div', 'tc-values');
    vals.append(el('span', '', `CNN ${fmt(r.cnn_value, 3)}`));
    vals.append(el('span', '', `TFM ${fmt(r.transformer_value, 3)}`));
    card.append(vals);
    host.append(card);
  });
}

/* ═════════════════════════ ranking explorer ═══════════════════════ */

function renderRanking() {
  const metric = $('#rank-metric').value;
  const scope = $('#rank-scope').value;
  const rows = (state.data.rankings[metric] || {})[scope] || [];
  const spec = metricSpec(metric);
  const host = $('#ranking-list');
  host.innerHTML = '';

  if (!rows.length) {
    host.append(el('p', 'note', 'No results in this scope.'));
    return;
  }

  const values = rows.map((r) => r.value);
  const lo = Math.min(...values), hi = Math.max(...values);

  rows.forEach((r) => {
    const ms = methodSpec(r.method);
    const row = el('div', 'rank-row');
    row.append(el('div', 'rank-num', String(r.rank)));

    const name = el('div', 'rank-name');
    name.append(document.createTextNode(ms.label));
    name.append(el('span', 'method-family-tag', ms.family_label));
    row.append(name);

    const track = el('div', 'rank-bar-track');
    const bar = el('div', 'rank-bar');
    const t = normalise(r.value, lo, hi, spec.higher_is_better);
    bar.style.width = `${Math.max(3, t * 100)}%`;
    bar.style.background = rampColor(t);
    track.append(bar);
    row.append(track);

    const val = el('div', 'rank-value');
    val.innerHTML = `${fmt(r.value, 3)} <small>n=${r.n_models}</small>`;
    row.append(val);
    host.append(row);
  });
}

/* ═══════════════════════ architecture explorer ════════════════════ */

function renderArchNav() {
  const nav = $('#arch-nav');
  nav.innerHTML = '';
  activeModels().forEach((m) => {
    const b = el('button', '', '');
    b.type = 'button';
    b.append(document.createTextNode(m.label));
    b.append(el('span', 'an-family', m.family_label));
    b.setAttribute('aria-current', String(m.key === state.arch));
    b.addEventListener('click', () => { state.arch = m.key; renderArchNav(); renderArchDetail(); });
    nav.append(b);
  });
}

function renderArchDetail() {
  const host = $('#arch-detail');
  host.innerHTML = '';
  const m = modelSpec(state.arch);
  if (!m) return;

  const head = el('div', 'arch-detail-head');
  head.append(el('h3', '', m.label));
  head.append(el('div', 'arch-meta',
    `${m.family_label}  ·  attention: ${m.attention}`));
  host.append(head);

  const metric = $('#matrix-metric') ? $('#matrix-metric').value : measuredMetrics()[0].key;
  const spec = metricSpec(metric);

  const rows = activeMethods()
    .map((me) => {
      const v = cellValue(m.key, me.key, metric);
      if (v == null) return null;
      // benchmark-wide mean for this method, for the delta
      const others = activeModels()
        .map((mo) => cellValue(mo.key, me.key, metric))
        .filter((x) => x != null);
      const mean = others.reduce((a, b) => a + b, 0) / others.length;
      return { me, v, delta: v - mean };
    })
    .filter(Boolean)
    .sort((a, b) => (spec.higher_is_better ? b.v - a.v : a.v - b.v));

  if (!rows.length) {
    host.append(el('p', 'note', 'No measured methods for this backbone.'));
    return;
  }

  const vals = rows.map((r) => r.v);
  const lo = Math.min(...vals), hi = Math.max(...vals);

  const list = el('div', 'ranking-list');
  rows.forEach((r, i) => {
    const row = el('div', 'rank-row');
    row.append(el('div', 'rank-num', String(i + 1)));
    const name = el('div', 'rank-name');
    name.append(document.createTextNode(r.me.label));
    name.append(el('span', 'method-family-tag', r.me.family_label));
    row.append(name);

    const track = el('div', 'rank-bar-track');
    const bar = el('div', 'rank-bar');
    const t = normalise(r.v, lo, hi, spec.higher_is_better);
    bar.style.width = `${Math.max(3, t * 100)}%`;
    bar.style.background = rampColor(t);
    track.append(bar);
    row.append(track);

    const better = spec.higher_is_better ? r.delta > 0 : r.delta < 0;
    const val = el('div', 'rank-value');
    val.innerHTML =
      `${fmt(r.v, 3)} <small>${better ? '+' : ''}${fmt(r.delta, 3)} vs mean</small>`;
    row.append(val);
    list.append(row);
  });
  host.append(list);
  host.append(el('p', 'note',
    `Scored by ${spec.label}. “vs mean” compares with this method's average over all backbones.`));
}

/* ═══════════════════════ dimension coverage ═══════════════════════ */

function renderDimensions() {
  const host = $('#dimension-grid');
  host.innerHTML = '';
  state.data.dimensions.forEach((d) => {
    const ok = d.status === 'measured';
    const card = el('div', `dim-card${ok ? '' : ' missing'}`);
    card.append(el('span', `dim-status ${ok ? 'ok' : 'no'}`,
      ok ? 'measured' : 'no data'));
    card.append(el('h3', '', d.label));
    if (ok) {
      const n = state.data.records.filter(
        (r) => d.metrics.some((k) => k in r.metrics)
      ).length;
      card.append(el('p', '', `${n} of ${state.data.records.length} cells carry a value for this dimension.`));
      card.append(el('div', 'dim-metrics',
        d.metrics.map((k) => metricSpec(k).label).join(' · ')));
    } else {
      card.append(el('p', '', d.reason));
    }
    host.append(card);
  });

  const gaps = state.meta.provenance_gaps || {};
  const gapNames = Object.keys(gaps);
  const total = state.data.records.length;
  $('#provenance-note').textContent = gapNames.length
    ? `Each record carries the run it came from. Fields not recorded by the ` +
      `archived runs: ${gapNames.join(', ')} (missing on up to ${Math.max(...Object.values(gaps))} ` +
      `of ${total} records). Runs produced with the current runner record all of them.`
    : `Every record carries complete provenance metadata.`;
}

/* ═════════════════════════════ controls ═══════════════════════════ */

function fillSelect(node, options, selected) {
  node.innerHTML = '';
  options.forEach(([value, label]) => {
    const o = el('option', '', label);
    o.value = value;
    if (value === selected) o.selected = true;
    node.append(o);
  });
}

function wireControls() {
  const metricOpts = measuredMetrics().map((m) => [m.key, m.label]);
  const defaultMetric = metricOpts.some(([k]) => k === 'pointing_game')
    ? 'pointing_game' : metricOpts[0][0];

  fillSelect($('#matrix-metric'), metricOpts, defaultMetric);
  fillSelect($('#rank-metric'), metricOpts, defaultMetric);
  fillSelect($('#transfer-metric'), metricOpts, defaultMetric);

  const famOpts = [['all', 'All families']].concat(
    state.data.arch_families
      .filter((f) => activeModels(f.key).length)
      .map((f) => [f.key, f.label])
  );
  fillSelect($('#matrix-family'), famOpts, 'all');

  const mfamOpts = [['all', 'All families']].concat(
    state.data.method_families
      .filter((f) => activeMethods(f.key).length)
      .map((f) => [f.key, f.label])
  );
  fillSelect($('#matrix-method-family'), mfamOpts, 'all');

  const scopeOpts = [['all', 'All backbones'], ['cnn_only', 'CNNs only'],
    ['transformer_only', 'Transformers only']].concat(
    state.data.arch_families
      .filter((f) => (state.data.rankings[defaultMetric] || {})[f.key])
      .map((f) => [f.key, f.label])
  );
  fillSelect($('#rank-scope'), scopeOpts, 'all');

  ['#matrix-metric', '#matrix-family', '#matrix-method-family', '#matrix-sort']
    .forEach((s) => $(s).addEventListener('change', () => { renderMatrix(); renderArchDetail(); }));
  ['#rank-metric', '#rank-scope'].forEach((s) =>
    $(s).addEventListener('change', renderRanking));
  $('#transfer-metric').addEventListener('change', renderTransfer);

  document.querySelectorAll('[data-reveal]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.reveal);
      const hidden = target.classList.toggle('visually-hidden');
      btn.textContent = hidden ? 'Show as table' : 'Hide table';
    });
  });
}


/* ══════════════════════════════ boot ══════════════════════════════ */

async function main() {
  try {
    const [data, meta] = await Promise.all([
      fetch(DATA_URL).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch(META_URL).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]);
    state.data = data;
    state.meta = meta;
  } catch (err) {
    document.querySelector('main').prepend(Object.assign(
      el('div', 'container'),
      { innerHTML:
        `<p class="note">Could not load the benchmark data (${err.message}). ` +
        `Run <code>python scripts/export_web_data.py</code>, then serve this ` +
        `directory over HTTP — opening index.html from the filesystem blocks fetch.</p>` }
    ));
    return;
  }

  if (!measuredMetrics().length) {
    $('#main').prepend(Object.assign(el('div', 'container'),
      { innerHTML: '<p class="note">The exported results contain no measured metrics.</p>' }));
    return;
  }

  state.arch = activeModels()[0]?.key ?? null;

  wireControls();
  renderScope();
  renderRibbon();
  renderTransfer();
  renderMatrix();
  renderRanking();
  renderArchNav();
  renderArchDetail();
  renderDimensions();

  const commit = state.meta.git_commit ? state.meta.git_commit.slice(0, 8) : 'unknown';
  $('#footer-meta').textContent =
    `${state.meta.n_records} records · schema v${state.meta.schema_version} · ` +
    `built ${state.meta.generated_utc} · commit ${commit}`;
}

// Stroke-draw keyframes injected here so the CSS file stays declarative.
const style = document.createElement('style');
style.textContent = '@keyframes ribbon-draw { to { stroke-dashoffset: 0; } }';
document.head.append(style);

main();
