"""Vision 4 — Deck Map.

The board IS the deck. One column per rank, one slot per copy of that rank.
A claim like "three 4s" is literally the bottom three slots of column 4, so
you bid a rank claim by clicking the slot at that height. Your cards fill
their columns from the bottom; the live claim's required slots glow; every
open slot is shaded by how likely that claim is. Combination claims (two
pair, full house) sit in a side list and preview themselves on the map on
hover. At showdown the bot's cards drop into the map, so you see exactly why
the call won or lost.

Run:  streamlit run visuals/arena_4_deck_map.py
Keys: C call · Space min-raise · N new · R peek · H odds · L load
"""

import streamlit as st

st.set_page_config(page_title="Liar's Poker · Deck Map", layout="wide", initial_sidebar_state="collapsed")

import arena_backend as ab  # noqa: E402

CSS = r"""
:host { --bg:#0a0f1e; --panel:#111a30; --panel2:#16213d; --line:#243152; --ink:#e8eeff; --soft:#8392b8; --you:#34e0ff; --bot:#ff4fd8; --heat:255,183,77;
  font-family:'Space Grotesk','Inter',system-ui,sans-serif; }
.lp-root { height:100vh; box-sizing:border-box; display:grid; grid-template-rows:auto 1fr; background:var(--bg); color:var(--ink); overflow:hidden;
  background-image:linear-gradient(rgba(52,224,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(52,224,255,.035) 1px, transparent 1px); background-size:32px 32px; }
button { font-family:inherit; cursor:pointer; color:inherit; }
.mono { font-family:'JetBrains Mono',monospace; }
.top { display:flex; align-items:center; gap:10px; padding:10px 18px; border-bottom:1px solid var(--line); background:rgba(10,15,30,.85); }
.brand { font-weight:700; font-size:20px; letter-spacing:.02em; } .brand span { color:var(--you); }
.chip { border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:7px 12px; font-size:13px; font-weight:600; }
.chip.on { border-color:var(--you); color:var(--you); }
.chip small { color:var(--soft); }
.grow { flex:1; }
.score { font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:700; } .score .y { color:var(--you); } .score .b { color:var(--bot); } .score small { font-size:12px; color:var(--soft); margin:0 6px; }
.main { display:grid; grid-template-columns:1fr minmax(340px,29vw); min-height:0; }
/* ---------- map ---------- */
.mapwrap { display:flex; flex-direction:column; padding:14px 20px 12px; min-height:0; gap:10px; }
.maphead { display:flex; align-items:flex-end; gap:20px; }
.maphead h2 { margin:0; font-size:15px; letter-spacing:.2em; text-transform:uppercase; color:var(--soft); font-weight:600; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--soft); margin-left:auto; }
.legend i { display:inline-block; width:14px; height:14px; border-radius:4px; vertical-align:-2px; margin-right:5px; }
.map { flex:1; display:grid; gap:12px; min-height:0; }
.col { display:grid; gap:8px; min-height:0; }
.colhead { text-align:center; font-family:'JetBrains Mono',monospace; font-size:clamp(26px,2.6vw,40px); font-weight:700; line-height:1; padding-top:4px; }
.colhead small { display:block; font-size:12px; color:var(--soft); font-family:'Space Grotesk',sans-serif; font-weight:500; margin-top:3px; }
.slot { position:relative; border-radius:12px; border:1.5px dashed #2c3a60; background:rgba(var(--heat), calc(var(--o,0) * .55)); display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:0; padding:0; container-type:size; transition:transform .1s, box-shadow .1s; }
.slot .kn { font-size:clamp(10px, 13cqh, 15px); letter-spacing:.14em; text-transform:uppercase; color:rgba(232,238,255,.7); font-weight:600; }
.slot .pc { font-family:'JetBrains Mono',monospace; font-size:clamp(14px, 30cqh, 34px); font-weight:700; }
.slot.legal:hover { transform:scale(1.04); box-shadow:0 0 0 2px var(--you), 0 10px 26px rgba(0,0,0,.5); z-index:2; }
.slot.gone { opacity:.28; cursor:default; }
.slot.gone.need { opacity:1; background:rgba(255,79,216,.08); }
.slot.gone.need .kn, .slot.gone.need .pc { color:var(--bot); }
.slot.gone.need.byyou { background:rgba(52,224,255,.08); } .slot.gone.need.byyou .kn, .slot.gone.need.byyou .pc { color:var(--you); }
.slot.nokind { background:transparent; cursor:default; }
.slot.mine, .slot.botc { border-style:solid; background:#f4f6ff; color:#0a0f1e; opacity:1; }
.slot.mine { border:3px solid var(--you); box-shadow:0 0 18px rgba(52,224,255,.35); }
.slot.botc { border:3px solid var(--bot); box-shadow:0 0 18px rgba(255,79,216,.35); }
.slot.mine .pc, .slot.botc .pc { font-size:clamp(20px, 46cqh, 54px); }
.slot.mine .kn { color:#0a7d93; } .slot.botc .kn { color:#b0149a; }
.slot.need::after { content:''; position:absolute; inset:-5px; border-radius:15px; border:3px solid var(--bot); pointer-events:none; animation:glow 1.6s ease-in-out infinite; }
.slot.need.byyou::after { border-color:var(--you); }
@keyframes glow { 50% { opacity:.45; } }
.slot.pv::before { content:''; position:absolute; inset:-5px; border-radius:15px; border:3px dashed var(--you); pointer-events:none; }
.slot .tag { position:absolute; top:5px; right:7px; font-size:10px; font-weight:700; letter-spacing:.08em; padding:2px 6px; border-radius:6px; }
.slot .tag.you { background:var(--you); color:#04222a; } .slot .tag.bot { background:var(--bot); color:#2c0626; }
/* ---------- side ---------- */
.side { border-left:1px solid var(--line); background:rgba(17,26,48,.92); display:flex; flex-direction:column; min-height:0; }
.sec { padding:14px 18px; border-bottom:1px solid var(--line); }
.sec h3 { margin:0 0 8px; font-size:12px; letter-spacing:.2em; text-transform:uppercase; color:var(--soft); font-weight:600; }
.cur .who { font-size:13px; color:var(--soft); }
.cur .name { font-size:clamp(26px,2.4vw,38px); font-weight:700; line-height:1.1; margin:2px 0 4px; }
.cur .name.you { color:var(--you); } .cur .name.bot { color:var(--bot); }
.cur .meta { font-size:14px; color:var(--soft); } .cur .meta b { color:var(--ink); }
.btns { display:flex; gap:10px; margin-top:12px; }
.call { flex:1.3; border:0; border-radius:12px; padding:14px; font-size:26px; font-weight:700; letter-spacing:.1em; background:var(--bot); color:#2c0626; }
.call:disabled { background:var(--panel2); color:#3f4c70; cursor:default; }
.minr { flex:1; border:1px solid var(--line); background:var(--panel2); border-radius:12px; padding:8px 10px; text-align:left; font-size:14px; font-weight:600; }
.minr small { display:block; font-size:11px; color:var(--soft); letter-spacing:.1em; }
.minr:disabled { opacity:.35; cursor:default; }
.cards { display:flex; gap:6px; }
.mini { width:34px; height:46px; border-radius:6px; background:#f4f6ff; color:#0a0f1e; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:20px; }
.mini.back { background:repeating-linear-gradient(45deg,#3a1846 0 5px,#5a2168 5px 10px); border:2px solid var(--bot); box-sizing:border-box; }
.status { font-size:14px; color:var(--you); margin-left:auto; } .status.bot { color:var(--bot); }
.combos { flex:1; overflow:auto; }
.cgrp { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--soft); margin:10px 0 6px; }
.clist { display:flex; flex-wrap:wrap; gap:6px; }
.cb { border:1px solid var(--line); background:var(--panel2); border-radius:8px; padding:6px 10px; font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:700; position:relative; }
.cb small { font-family:'Space Grotesk',sans-serif; color:var(--soft); font-weight:500; font-size:11px; margin-left:4px; }
.cb:hover { border-color:var(--you); color:var(--you); }
.none { color:var(--soft); font-size:14px; }
.hist { display:flex; flex-wrap:wrap; gap:6px; }
.hist span { font-size:14px; font-weight:600; padding:4px 9px; border-radius:7px; border:1px solid; }
.hist .you { border-color:var(--you); color:var(--you); } .hist .bot { border-color:var(--bot); color:var(--bot); }
.hist .fresh { animation:pop .4s ease-out; } @keyframes pop { from { transform:scale(.5); opacity:0; } }
/* result */
.res .v { font-size:44px; font-weight:700; line-height:1; } .res .v.you { color:var(--you); } .res .v.bot { color:var(--bot); }
.res p { font-size:16px; margin:8px 0; color:#c4cdea; }
.again { width:100%; border:0; border-radius:12px; padding:14px; font-size:20px; font-weight:700; background:var(--you); color:#04222a; margin-top:6px; }
.lp-review { display:flex; flex-direction:column; gap:8px; margin-top:10px; }
.lp-review-move { background:var(--panel2); border-radius:10px; padding:8px 10px; }
.lp-review-head { font-size:12px; color:var(--soft); margin-bottom:4px; } .lp-review-head b { color:var(--ink); }
.lp-review-row { display:grid; grid-template-columns:1fr 60px 36px; gap:8px; align-items:center; font-size:13px; padding:1px 0; }
.lp-review-row i { height:5px; border-radius:3px; background:linear-gradient(90deg,var(--bot) calc(var(--p)*100%), #2a3658 0); }
.lp-review-row em { font-style:normal; text-align:right; color:var(--soft); }
.lp-review-row.chosen span { color:var(--bot); font-weight:700; }
/* loader */
.lp-modal { position:fixed; inset:0; background:rgba(4,7,15,.75); display:flex; align-items:center; justify-content:center; z-index:50; }
.lp-modal-card { width:min(820px,92vw); max-height:84vh; display:flex; flex-direction:column; background:var(--panel); border:1px solid var(--you); border-radius:16px; padding:18px; }
.lp-modal-head { display:flex; align-items:center; } .lp-modal-title { font-size:24px; font-weight:700; flex:1; }
.lp-x { background:none; border:0; font-size:28px; }
.lp-pathrow { display:flex; gap:8px; margin:10px 0; }
.lp-path { flex:1; background:var(--bg); border:1px solid var(--line); color:var(--ink); border-radius:10px; padding:10px 12px; font-size:14px; font-family:'JetBrains Mono',monospace; }
.lp-go { background:var(--you); color:#04222a; border:0; border-radius:10px; padding:0 20px; font-weight:700; }
.lp-err { color:var(--bot); margin-bottom:8px; }
.lp-presets { overflow-y:auto; display:flex; flex-direction:column; gap:4px; }
.lp-group { margin:10px 0 4px; font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); }
.lp-preset { display:flex; justify-content:space-between; gap:12px; text-align:left; background:var(--panel2); border:1px solid transparent; border-radius:8px; padding:8px 10px; }
.lp-preset:hover { border-color:var(--you); } .lp-preset.current { border-color:var(--bot); }
.lp-preset-game { font-weight:600; font-size:14px; white-space:nowrap; } .lp-preset-rel { color:var(--soft); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:'JetBrains Mono',monospace; }
"""

JS = r"""
export default function (c) {
  LP.font('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
  const d = c.data;
  const root = LP.root(c);
  const ui = LP.ui(c, {loader: false, lastPly: -1, lastHand: -1});
  const send = o => LP.send(c, o);
  const LEVEL_KIND = {1: 'RankHigh', 2: 'Pair', 3: 'Trips', 4: 'Quads'};
  const KIND_WORD = {RankHigh: 'one', Pair: 'pair', Trips: 'trips', Quads: 'quads'};

  function draw() {
    if (!d.loaded) {
      root.innerHTML = `<div class="top"><div class="brand">DECK<span>MAP</span></div></div><div style="padding:40px;color:var(--soft)">Load an opponent to begin.</div>` + LP.loaderHTML(d);
      LP.bindLoader(root, c, ui, draw); return;
    }
    const S = d.spec, legal = new Set(d.legal);
    const mine = Object.fromEntries(d.mine);
    const botc = {};
    if (d.bot_hand) for (const x of d.bot_hand) botc[x.rank] = (botc[x.rank] || 0) + 1;
    const last = LP.claimOf(d, d.last_claim);
    const lastBid = d.history.filter(h => h.idx >= 0).slice(-1)[0];
    const need = last ? Object.fromEntries(last.need) : {};
    const single = {};
    for (const cl of d.claims) if (LEVEL_KIND[cl.cards] === cl.kind) single[cl.ranks[0] + ':' + cl.cards] = cl;

    // ---- the map: columns = ranks, rows = copies (level 1 at the bottom) ----
    // rows: highest level that has a claim, or that cards actually reach
    let rows = Math.max(...d.claims.filter(cl => LEVEL_KIND[cl.cards] === cl.kind).map(cl => cl.cards));
    for (let r = 1; r <= S.ranks; r++) rows = Math.max(rows, (mine[r] || 0) + (botc[r] || 0));
    rows = Math.min(rows, S.suits);
    let map = `<div class="map" style="grid-template-columns:repeat(${S.ranks},1fr)">`;
    for (let r = 1; r <= S.ranks; r++) {
      map += `<div class="col" style="grid-template-rows:auto repeat(${rows},1fr)"><div class="colhead">${r}<small>${mine[r] ? `you hold ${mine[r]}` : '&nbsp;'}</small></div>`;
      for (let k = rows; k >= 1; k--) {
        const cl = single[r + ':' + k];
        const isMine = k <= (mine[r] || 0);
        const isBot = !isMine && k <= (mine[r] || 0) + (botc[r] || 0);
        const cls = ['slot'];
        let inner = '';
        if (isMine || isBot) {
          cls.push(isMine ? 'mine' : 'botc');
          inner = `<span class="pc">${r}</span><span class="kn">${isMine ? 'your card' : "bot's card"}</span>`;
        } else if (cl) {
          inner = `<span class="kn">${KIND_WORD[cl.kind]}</span>${d.hints ? `<span class="pc">${LP.pct(d.odds[cl.idx])}</span>` : `<span class="pc">${r}${'·'.repeat(0)}</span>`}`;
        }
        if (!cl) cls.push('nokind');
        else if (legal.has(cl.idx)) cls.push('legal');
        else if (!isMine && !isBot) cls.push('gone');
        if (k <= (need[r] || 0)) { cls.push('need'); if (lastBid && lastBid.who === 'you') cls.push('byyou'); }
        const o = cl && !isMine && !isBot && d.hints ? d.odds[cl.idx] : 0;
        const bid = cl ? d.history.find(h => h.idx === cl.idx) : null;
        const tag = bid ? `<span class="tag ${bid.who}">${bid.who === 'you' ? 'YOU' : 'BOT'} #${bid.ply + 1}</span>` : '';
        map += `<button class="${cls.join(' ')}" data-r="${r}" data-k="${k}" ${cl ? `data-idx="${cl.idx}"` : ''} style="--o:${o}" title="${cl ? LP.esc(cl.label) : ''}">${tag}${inner}</button>`;
      }
      map += `</div>`;
    }
    map += `</div>`;

    // ---- side panel ----
    const botCards = d.bot_hand ? d.bot_hand.map(x => `<div class="mini">${x.rank}</div>`).join('') : Array(d.bot_cards).fill('<div class="mini back"></div>').join('');
    const statusTxt = d.turn === 'bot' ? `<span class="status bot">thinking…</span>` : d.turn === 'you' ? `<span class="status">your move</span>` : '';
    let curSec;
    if (d.turn === 'over') {
      const R = d.result;
      const story = R.caller === 'you'
        ? `You called the bot's <b>${LP.esc(R.claim)}</b> — ${R.was_true ? 'it was there.' : 'a bluff.'}`
        : `The bot called your <b>${LP.esc(R.claim)}</b> — ${R.was_true ? 'it was there.' : 'it was not there.'}`;
      curSec = `<div class="sec res"><div class="v ${R.winner}">${R.winner === 'you' ? 'YOU WIN' : 'BOT WINS'}</div><p>${story}<br>The map now shows both hands; the glowing slots were needed.</p>
        <button class="again" data-act="new">Deal again · N</button>${LP.reviewHTML(d)}</div>`;
    } else {
      const min = d.legal.length ? d.claims[d.legal[0]] : null;
      curSec = `<div class="sec cur"><h3>On the table</h3>
        ${last ? `<div class="who">${lastBid.who === 'you' ? 'You claimed' : 'Bot claims'}</div><div class="name ${lastBid.who}">${LP.esc(last.label)}</div>
          <div class="meta">Glowing slots on the map${d.hints ? ` · true vs a random hand <b>${LP.pct(d.odds[last.idx])}</b>` : ''}</div>`
              : `<div class="name">${d.turn === 'you' ? 'You open.' : 'Bot opens…'}</div><div class="meta">Click a slot: the k-th slot of a column claims k of that rank.</div>`}
        <div class="btns"><button class="call" data-act="call" ${d.can_call ? '' : 'disabled'}>CALL</button>
          <button class="minr" data-act="minraise" ${min && d.turn === 'you' ? '' : 'disabled'}><small>MIN RAISE · SPACE</small>${min ? LP.esc(min.label) : '—'}</button></div></div>`;
    }
    let combos = '';
    const comboKinds = [['TwoPair', 'Two pair'], ['FullHouse', 'Full house']].filter(([k]) => S.claim_kinds.includes(k));
    for (const [k, name] of comboKinds) {
      const list = d.claims.filter(cl => cl.kind === k && legal.has(cl.idx));
      combos += `<div class="cgrp">${name}</div><div class="clist">`;
      combos += list.length ? list.map(cl => `<button class="cb" data-idx="${cl.idx}" title="${LP.esc(cl.label)}">${cl.need.map(([r, n]) => String(r).repeat(n)).join('·')}${d.hints ? `<small>${LP.pct(d.odds[cl.idx])}</small>` : ''}</button>`).join('')
                            : `<span class="none">${d.turn === 'you' ? 'none left' : '—'}</span>`;
      combos += `</div>`;
    }
    const hist = d.history.length ? d.history.map(h => `<span class="${h.who} ${(ui.lastHand === d.hand_no && h.ply > ui.lastPly) ? 'fresh' : ''}">${LP.esc(h.label)}</span>`).join('') : '<span class="none" style="border:0">No bids yet.</span>';
    const seatNow = d.pending_seat || (d.you_open ? 'first' : 'second');

    root.innerHTML = `
      <div class="top"><div class="brand">DECK<span>MAP</span></div>
        <button class="chip" data-act="loader">${S.ranks} ranks × ${S.suits} copies · ${S.hand_size} cards each · ${S.n_claims} claims <small>▾ ${LP.esc(d.policy_kind)}</small></button>
        <div class="grow"></div><div class="score"><span class="y">${d.score.you}</span><small>YOU · BOT</small><span class="b">${d.score.bot}</span></div><div class="grow"></div>
        <button class="chip ${seatNow === 'first' ? 'on' : ''}" data-seat="first">You open</button><button class="chip ${seatNow === 'second' ? 'on' : ''}" data-seat="second">Bot opens</button>
        <button class="chip ${d.hints ? 'on' : ''}" data-act="hints">Odds · H</button><button class="chip ${d.reveal ? 'on' : ''}" data-act="reveal">Peek · R</button>
        <button class="chip" data-act="new">New · N</button><button class="chip" data-act="reset_score">Reset score</button></div>
      <div class="main">
        <div class="mapwrap"><div class="maphead"><h2>The deck · ${S.ranks * S.suits} cards</h2>
          <div class="legend"><span><i style="background:#f4f6ff;box-shadow:inset 0 0 0 2px var(--you)"></i>your card</span><span><i style="box-shadow:inset 0 0 0 2px var(--bot)"></i>needed by current claim</span>${d.hints ? `<span><i style="background:rgba(var(--heat),.5)"></i>chance it's there</span>` : ''}</div></div>
          ${map}</div>
        <div class="side">
          <div class="sec"><h3 style="display:flex">Bot's hand ${statusTxt}</h3><div class="cards">${botCards}</div></div>
          ${curSec}
          <div class="sec combos"><h3>Combination claims · hover to preview</h3>${combos || '<span class="none">This game has none.</span>'}</div>
          <div class="sec"><h3>Bidding · hand ${d.hand_no}</h3><div class="hist">${hist}</div></div>
        </div>
      </div>
      ${ui.loader ? LP.loaderHTML(d) : ''}`;

    root.querySelectorAll('.slot.legal, .cb[data-idx]').forEach(b => {
      b.addEventListener('click', () => send({type: 'claim', idx: +b.dataset.idx}));
      b.addEventListener('mouseenter', () => preview(+b.dataset.idx));
      b.addEventListener('mouseleave', () => preview(null));
    });
    root.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => act(b.dataset.act)));
    root.querySelectorAll('[data-seat]').forEach(b => b.addEventListener('click', () => send({type: 'seat', value: b.dataset.seat})));
    if (ui.loader) LP.bindLoader(root, c, ui, draw);
    ui.lastPly = d.history.length - 1; ui.lastHand = d.hand_no;
  }

  // dashed outline on the slots a hovered claim would need
  function preview(idx) {
    root.querySelectorAll('.slot.pv').forEach(s => s.classList.remove('pv'));
    if (idx == null || Number.isNaN(idx)) return;
    for (const [r, n] of d.claims[idx].need)
      for (let k = 1; k <= n; k++) root.querySelector(`.slot[data-r="${r}"][data-k="${k}"]`)?.classList.add('pv');
  }

  function act(a) {
    if (a === 'loader') { ui.loader = true; draw(); return; }
    if (a === 'minraise') { if (d.legal.length && d.turn === 'you') send({type: 'claim', idx: d.legal[0]}); return; }
    send({type: a});
  }

  LP.keys(c, e => {
    const k = e.key.toLowerCase();
    if (!d.loaded) { if (k === 'escape') { ui.loader = false; draw(); return true; } return false; }
    if (ui.loader) { if (k === 'escape') { ui.loader = false; draw(); return true; } return false; }
    if (k === ' ') { act('minraise'); return true; }
    if (k === 'c') { if (d.can_call) act('call'); return true; }
    if (k === 'n' || (k === 'enter' && d.turn === 'over')) { act('new'); return true; }
    if (k === 'r') { act('reveal'); return true; }
    if (k === 'h') { act('hints'); return true; }
    if (k === 'l') { act('loader'); return true; }
    return false;
  });
  draw();
}
"""

board = ab.make_component("arena_deck_map", css=CSS, js=JS)
ab.run(board, key="deckmap_board", background="#0a0f1e")
