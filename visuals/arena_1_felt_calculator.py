"""Vision 1 — Felt Calculator.

A card table with a compact claim calculator. Both hands sit side by side in
the top strip, next to the live claim and the CALL button; the calculator
fills the table below and the bidding history runs down the right. The
calculator has three layouts (Grid · Table · Rows), switchable from its
header or with G.

Run:  streamlit run visuals/arena_1_felt_calculator.py
Keys: C call · Space min-raise · N new hand · G button layout · R peek · H odds · L load
"""

import streamlit as st

st.set_page_config(page_title="Liar's Poker · Felt", layout="wide", initial_sidebar_state="collapsed")

import arena_backend as ab  # noqa: E402

CSS = r"""
:host { --felt:#0e3b2c; --felt2:#145238; --ink:#f3efe3; --muted:#9fc1ae; --you:#f5c542; --bot:#ff6f5b;
  --card:#fbf8ef; --line:rgba(255,255,255,.12); font-family:'Inter',system-ui,sans-serif;
  --cc-bg:rgba(255,255,255,.93); --cc-fg:#14261d; --cc-border:rgba(255,255,255,.2); --cc-radius:7px; --cc-font:'Barlow Condensed',sans-serif;
  --cc-fs:18px; --cc-h:36px; --cc-w:52px;
  --cc-hover-bg:#fff; --cc-dead-bg:rgba(0,0,0,.18); --cc-dead-fg:rgba(255,255,255,.26); --cc-cur-bg:#08251a; --cc-cur-fg:#fff; --cc-cur-ring:var(--ink);
  --cc-you:var(--you); --cc-bot:var(--bot); --cc-bar:linear-gradient(90deg,#2fb57a,#8ee06a); --cc-odds-fg:#3b7a5c; --cc-hdr:var(--muted);
  --cc-sw-border:var(--line); --cc-sw-on-bg:var(--ink); --cc-sw-on-fg:#12291f; --cc-read-strong:var(--ink); }
.lp-root { height:100vh; box-sizing:border-box; color:var(--ink); display:grid; grid-template-rows:auto auto 1fr;
  background:radial-gradient(120% 90% at 40% 45%, var(--felt2) 0%, var(--felt) 60%, #082419 100%); overflow:hidden; }
button { font-family:inherit; cursor:pointer; }
/* ---------- top bar ---------- */
.top { display:flex; align-items:center; gap:12px; padding:9px 18px; background:rgba(0,0,0,.28); border-bottom:1px solid var(--line); flex-wrap:wrap; }
.brand { font-family:'Barlow Condensed',sans-serif; font-weight:800; font-size:25px; letter-spacing:.06em; text-transform:uppercase; }
.brand span { color:var(--you); }
.opp { display:flex; flex-direction:column; line-height:1.15; padding:4px 12px; border-radius:10px; background:rgba(255,255,255,.06); border:1px solid var(--line); text-align:left; color:var(--ink); }
.opp small { color:var(--muted); font-size:12px; } .opp b { font-size:14px; }
.spacer { flex:1; }
.score { display:flex; align-items:center; gap:9px; font-family:'Barlow Condensed',sans-serif; font-size:28px; font-weight:800; }
.score .lbl { font-size:12px; letter-spacing:.12em; color:var(--muted); font-weight:600; }
.score .y { color:var(--you); } .score .b { color:var(--bot); }
.seg { display:flex; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.seg button { background:transparent; color:var(--muted); border:0; padding:7px 11px; font-size:13px; font-weight:600; }
.seg button.on { background:var(--ink); color:#12291f; }
.tbtn { background:rgba(255,255,255,.08); color:var(--ink); border:1px solid var(--line); border-radius:10px; padding:7px 12px; font-size:13px; font-weight:600; }
.tbtn.on { background:rgba(245,197,66,.18); border-color:var(--you); color:var(--you); }
.tbtn kbd { font-family:inherit; font-size:11px; opacity:.55; margin-left:5px; }
/* ---------- strip: hands · claim · actions ---------- */
.strip { display:flex; align-items:center; gap:22px; padding:12px 18px; margin:12px 16px 0; border-radius:16px; background:rgba(0,0,0,.25); border:1px solid var(--line); flex-wrap:wrap; }
.hands { display:flex; gap:20px; align-items:flex-end; }
.seat .who { font-family:'Barlow Condensed',sans-serif; font-size:14px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; display:flex; gap:8px; align-items:center; white-space:nowrap; }
.seat.you .who { color:var(--you); }
.cards { display:flex; gap:7px; }
.card { width:clamp(40px,3.6vw,60px); aspect-ratio:5/7; border-radius:8px; background:var(--card); color:#1d1d1d; display:flex; align-items:center; justify-content:center; position:relative;
  font-family:'Barlow Condensed',sans-serif; font-weight:800; font-size:clamp(26px,2.5vw,42px); box-shadow:0 4px 0 rgba(0,0,0,.25), 0 8px 16px rgba(0,0,0,.3); }
.card .c { position:absolute; top:3px; left:6px; font-size:.36em; }
.card.back { background:repeating-linear-gradient(45deg,#8a2c22 0 7px,#a8392c 7px 14px); border:3px solid var(--card); box-sizing:border-box; }
.card.hit { box-shadow:0 0 0 3px var(--you), 0 8px 16px rgba(0,0,0,.3); }
.vs { align-self:center; font-family:'Barlow Condensed',sans-serif; font-weight:700; color:var(--muted); font-size:14px; letter-spacing:.1em; padding-top:18px; }
.status { font-size:12px; letter-spacing:.04em; text-transform:none; color:var(--muted); }
.status.turn-you { color:var(--you); font-weight:700; }
.dots i { display:inline-block; width:6px; height:6px; margin:0 1px; border-radius:50%; background:var(--bot); animation:blink 1s infinite; }
.dots i:nth-child(2){animation-delay:.2s} .dots i:nth-child(3){animation-delay:.4s}
@keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
.now { flex:1; min-width:180px; padding-left:22px; border-left:1px solid var(--line); }
.now small { display:block; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); }
.now b { font-family:'Barlow Condensed',sans-serif; font-size:clamp(26px,3vw,36px); font-weight:800; line-height:1.05; display:block; }
.now b.by-you { color:var(--you); } .now b.by-bot { color:var(--bot); }
.now b.fresh { animation:claimIn .3s ease-out .25s backwards; }
@keyframes claimIn { from { opacity:0; transform:translateY(6px); } }
.now .odds { font-size:13px; color:var(--muted); margin-top:2px; } .now .odds b { display:inline; font-family:inherit; font-size:inherit; color:var(--ink); }
.acts { display:flex; gap:10px; align-items:stretch; }
.minr { background:rgba(255,255,255,.08); color:var(--ink); border:1px solid var(--line); border-radius:12px; padding:6px 14px; text-align:left; font-size:15px; font-weight:700; }
.minr small { display:block; font-size:10px; letter-spacing:.14em; color:var(--muted); font-weight:600; }
.minr:disabled { opacity:.35; cursor:default; }
.call { background:var(--bot); color:#fff; border:0; border-radius:12px; padding:10px 30px; font-family:'Barlow Condensed',sans-serif; font-size:30px; font-weight:800; letter-spacing:.08em; box-shadow:0 5px 0 #a5382b; }
.call small { font-size:14px; opacity:.6; margin-left:4px; }
.call:disabled { background:#3d5a4d; color:#7f9b8d; box-shadow:none; cursor:default; }
.call:not(:disabled):hover { filter:brightness(1.08); }
/* ---------- main ---------- */
.main { display:grid; grid-template-columns:1fr minmax(240px, 21vw); min-height:0; gap:0; }
.work { padding:14px 18px 16px; min-height:0; display:flex; flex-direction:column; }
.work .calc-frame { flex:1; }
.side { margin:12px 16px 16px 0; border-radius:16px; background:rgba(0,0,0,.2); border:1px solid var(--line); display:flex; flex-direction:column; min-height:0; }
.side h3 { margin:0; padding:12px 16px 8px; font-family:'Barlow Condensed',sans-serif; font-size:16px; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.hist { flex:1; overflow-y:auto; padding:0 12px 12px; display:flex; flex-direction:column; gap:6px; }
.h { display:grid; grid-template-columns:22px 1fr; align-items:center; gap:8px; padding:7px 10px; border-radius:10px; background:rgba(255,255,255,.05); }
.h .n { font-size:11px; color:var(--muted); font-weight:700; }
.h .t { font-size:clamp(15px,1.15vw,19px); font-weight:700; }
.h .w { font-size:10px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.h.you { border-left:4px solid var(--you); } .h.you .w { color:var(--you); }
.h.bot { border-left:4px solid var(--bot); } .h.bot .w { color:var(--bot); }
.h.call { background:rgba(255,111,91,.18); }
.h.fresh { animation:pop .3s ease-out; } .h.bot.fresh { animation:pop .3s ease-out .25s backwards; }
@keyframes pop { from { transform:translateX(24px); opacity:0 } to { transform:none; opacity:1 } }
.hist .none { color:var(--muted); font-size:14px; padding:6px 4px; }
.lp-pending .cc, .lp-pending .call, .lp-pending .minr { pointer-events:none; }
/* ---------- result ---------- */
.result { display:flex; flex-direction:column; gap:12px; justify-content:center; align-items:center; text-align:center; border-radius:18px; background:rgba(0,0,0,.3); border:1px solid var(--line); padding:18px; flex:1; min-height:0; overflow:auto; }
.result .big { font-family:'Barlow Condensed',sans-serif; font-size:60px; font-weight:800; line-height:1; letter-spacing:.04em; }
.result .big.you { color:var(--you); } .result .big.bot { color:var(--bot); }
.result .why { font-size:19px; max-width:760px; }
.counts { display:flex; gap:8px; }
.cnt { width:58px; padding:6px 0; border-radius:10px; background:rgba(255,255,255,.08); font-family:'Barlow Condensed',sans-serif; }
.cnt b { display:block; font-size:28px; } .cnt small { font-size:12px; color:var(--muted); display:block; }
.cnt.need { background:rgba(245,197,66,.2); outline:2px solid var(--you); }
.again { background:var(--you); color:#1d1a0f; border:0; border-radius:12px; padding:11px 28px; font-size:20px; font-weight:800; box-shadow:0 5px 0 #b38a1d; }
.lp-review { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
.lp-review-move { background:rgba(255,255,255,.06); border-radius:12px; padding:10px 12px; min-width:230px; text-align:left; }
.lp-review-head { font-size:13px; color:var(--muted); margin-bottom:6px; } .lp-review-head b { color:var(--ink); }
.lp-review-row { display:grid; grid-template-columns:1fr 70px 40px; gap:8px; align-items:center; font-size:13px; padding:2px 0; }
.lp-review-row i { height:6px; border-radius:3px; background:linear-gradient(90deg,var(--bot) calc(var(--p)*100%), rgba(255,255,255,.1) 0); }
.lp-review-row em { font-style:normal; text-align:right; color:var(--muted); }
.lp-review-row.chosen span { color:var(--bot); font-weight:800; }
/* ---------- loader modal ---------- */
.lp-modal { position:fixed; inset:0; background:rgba(0,0,0,.6); display:flex; align-items:center; justify-content:center; z-index:50; }
.lp-modal-card { width:min(820px,92vw); max-height:84vh; display:flex; flex-direction:column; background:#10271e; border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:0 30px 80px rgba(0,0,0,.6); }
.lp-modal-head { display:flex; align-items:center; } .lp-modal-title { font-family:'Barlow Condensed',sans-serif; font-size:28px; font-weight:800; flex:1; }
.lp-x { background:none; border:0; color:var(--ink); font-size:30px; }
.lp-pathrow { display:flex; gap:8px; margin:10px 0; }
.lp-path { flex:1; background:rgba(255,255,255,.08); border:1px solid var(--line); color:var(--ink); border-radius:10px; padding:10px 12px; font-size:14px; }
.lp-go { background:var(--you); border:0; border-radius:10px; padding:0 20px; font-weight:800; }
.lp-err { color:var(--bot); margin-bottom:8px; }
.lp-presets { overflow-y:auto; display:flex; flex-direction:column; gap:4px; }
.lp-group { margin:10px 0 2px; font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.lp-preset { display:flex; justify-content:space-between; gap:12px; text-align:left; background:rgba(255,255,255,.04); border:1px solid transparent; color:var(--ink); border-radius:8px; padding:8px 10px; }
.lp-preset:hover { border-color:var(--you); } .lp-preset.current { background:rgba(245,197,66,.14); }
.lp-preset-game { font-weight:700; font-size:14px; white-space:nowrap; } .lp-preset-rel { color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.welcome { display:flex; align-items:center; justify-content:center; font-size:22px; color:var(--muted); }
/* ---------- other screen shapes ---------- */
@media (orientation: portrait) {
  .top .spacer { display:none; } .top .score { margin-left:auto; }
  .strip { margin:10px 12px 0; gap:14px 18px; }
  .now { border-left:0; padding-left:0; flex-basis:100%; order:3; }
  .acts { margin-left:auto; }
  .main { grid-template-columns:1fr; grid-template-rows:1fr auto; }
  .work { padding:12px; }
  .side { margin:0 12px 12px; height:min(22vh, 240px); }
}
@media (orientation: landscape) and (max-width: 1150px) {
  .top .spacer { display:none; } .top .score { margin-left:auto; }
  .main { grid-template-columns:1fr 220px; }
}
@media (min-width: 2200px) and (min-height: 1250px) { .lp-root { zoom:1.35; height:calc(100vh / 1.35); } }
@media (min-width: 3200px) and (min-height: 1800px) { .lp-root { zoom:1.8; height:calc(100vh / 1.8); } }
"""

JS = r"""
export default function (c) {
  LP.font('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Inter:wght@400;600;700;800&display=swap');
  let d = c.data;
  const root = LP.root(c, 'felt');
  const ui = LP.ui(c, {loader: false, lastPly: -1, lastHand: -1, layout: LP.calcLayout('felt', 'grid')});
  const send = o => LP.send(c, o);
  const words = ['', 'one', 'two', 'three', 'four', 'five', 'six'];

  const card = (cd, cls = '') => `<div class="card ${cls}"><span class="c">${cd.rank}${cd.suit == null ? '' : '♠♥♦♣'[cd.suit]}</span>${cd.rank}</div>`;

  function bid(idx) {
    if (d.turn !== 'you' || !d.legal.includes(idx)) return;
    d = LP.optimistic(d, idx);
    draw();
    send({type: 'claim', idx});
  }
  function setLayout(v) { ui.layout = v; LP.setCalcLayout('felt', v); draw(); }

  function draw() {
    if (!d.loaded) {
      root.innerHTML = `<div class="top"><div class="brand">Liar's <span>Poker</span></div></div><div></div><div class="welcome">Load an opponent to begin.</div>` + LP.loaderHTML(d);
      LP.bindLoader(root, c, ui, draw); return;
    }
    const S = d.spec;
    const last = LP.claimOf(d, d.last_claim);
    const lastBy = d.history.filter(h => h.idx >= 0).slice(-1)[0];
    const need = last ? new Set(last.need.map(x => x[0])) : new Set();
    const claimFresh = lastBy && lastBy.who === 'bot' && !(ui.lastHand === d.hand_no && lastBy.ply <= ui.lastPly);

    // ---- work area: calculator, or the result once the hand is over ----
    let work;
    if (d.turn === 'over') {
      const R = d.result;
      const why = R.caller === 'you'
        ? `You called the bot's <b>${LP.esc(R.claim)}</b> — it was <b>${R.was_true ? 'true' : 'a bluff'}</b>.`
        : `The bot called your <b>${LP.esc(R.claim)}</b> — it was <b>${R.was_true ? 'true' : 'false'}</b>.`;
      let counts = '<div class="counts">';
      for (const [r, n] of R.counts) {
        const nd = R.need.find(x => x[0] === r);
        counts += `<div class="cnt ${nd ? 'need' : ''}"><small>${r}s</small><b>${n}</b><small>${nd ? 'need ' + nd[1] : '&nbsp;'}</small></div>`;
      }
      counts += '</div>';
      work = `<div class="result"><div class="big ${R.winner}">${R.winner === 'you' ? 'You win' : 'Bot wins'}</div>
        <div class="why">${why}</div>${counts}<div style="color:var(--muted);font-size:13px">How many of each rank were out, across both hands</div>
        <button class="again" data-act="new">Deal again <span style="opacity:.6;font-size:13px">N</span></button>${LP.reviewHTML(d)}</div>`;
    } else {
      work = LP.calcFrame(d, ui.layout);
    }

    const status = d.turn === 'bot' ? `<span class="status">thinking <span class="dots"><i></i><i></i><i></i></span></span>`
      : d.turn === 'you' ? `<span class="status turn-you">your move</span>` : '';
    const botCards = d.bot_hand ? d.bot_hand.map(x => card(x, need.has(x.rank) ? 'hit' : '')).join('') : Array(d.bot_cards).fill('<div class="card back"></div>').join('');
    const hold = d.mine.filter(x => x[1]).reverse().map(x => x[1] === 1 ? `one ${x[0]}` : `${words[x[1]] || x[1]} ${x[0]}s`).join(', ');
    const min = d.turn === 'you' && d.legal.length ? d.claims[d.legal[0]] : null;
    const nowTxt = last ? `<b class="by-${lastBy.who}${claimFresh ? ' fresh' : ''}">${LP.esc(last.label)}</b>` : `<b>No claim yet</b>`;
    const nowSub = last ? `Current claim · ${lastBy.who === 'you' ? 'yours' : "the bot's"}` : (d.turn === 'you' ? 'You open the bidding' : 'Bot opens');
    const oddsTxt = last && d.hints && d.turn !== 'over' ? `<div class="odds">If the bot's cards were random, true <b>${LP.pct(d.odds[last.idx])}</b> of the time</div>` : '';

    let hist = d.history.length ? '' : `<div class="none">No bids yet.</div>`;
    for (const h of d.history) {
      const fresh = (ui.lastHand === d.hand_no && h.ply > ui.lastPly) ? ' fresh' : '';
      hist += `<div class="h ${h.who}${h.idx < 0 ? ' call' : ''}${fresh}"><div class="n">${h.ply + 1}</div><div><div class="w">${h.who === 'you' ? 'You' : 'Bot'}</div><div class="t">${LP.esc(h.label)}</div></div></div>`;
    }
    const seatNow = d.pending_seat || (d.you_open ? 'first' : 'second');

    root.innerHTML = `
      <div class="top">
        <div class="brand">Liar's <span>Poker</span></div>
        <button class="opp" data-act="loader"><small>Opponent · ${LP.esc(d.policy_kind)} ▾</small><b>${S.ranks} ranks · ${S.suits} suits · ${S.hand_size} cards · ${S.n_claims} claims</b></button>
        <div class="spacer"></div>
        <div class="score"><span class="lbl">YOU</span><span class="y">${d.score.you}</span><span style="opacity:.4">–</span><span class="b">${d.score.bot}</span><span class="lbl">BOT</span></div>
        <div class="spacer"></div>
        <div class="seg" title="Who opens the bidding${d.pending_seat ? ' (applies next hand)' : ''}"><button data-seat="first" class="${seatNow === 'first' ? 'on' : ''}">You open</button><button data-seat="second" class="${seatNow === 'second' ? 'on' : ''}">Bot opens</button></div>
        <button class="tbtn ${d.hints ? 'on' : ''}" data-act="hints" title="Show the chance each claim is true if the bot held random cards">Odds %<kbd>H</kbd></button>
        <button class="tbtn ${d.reveal ? 'on' : ''}" data-act="reveal">Peek<kbd>R</kbd></button>
        <button class="tbtn" data-act="new">New hand<kbd>N</kbd></button>
        <button class="tbtn" data-act="reset_score">Reset score</button>
      </div>
      <div class="strip">
        <div class="hands">
          <div class="seat you"><div class="who">You</div><div class="cards">${d.hand.map(x => card(x, need.has(x.rank) ? 'hit' : '')).join('')}</div></div>
          <div class="vs">VS</div>
          <div class="seat bot"><div class="who">Bot ${status}</div><div class="cards">${botCards}</div></div>
        </div>
        <div class="now"><small>${nowSub}</small>${nowTxt}${oddsTxt}</div>
        <div class="acts">
          <button class="minr" data-act="minraise" ${min ? '' : 'disabled'}><small>MIN RAISE · SPACE</small>${min ? LP.esc(min.label) : '—'}</button>
          <button class="call" data-act="call" ${d.can_call ? '' : 'disabled'}>CALL<small>C</small></button>
        </div>
      </div>
      <div class="main">
        <div class="work">${work}</div>
        <div class="side"><h3>Bidding · hand ${d.hand_no}</h3><div class="hist">${hist}</div>
          <div style="padding:0 16px 12px;font-size:12px;color:var(--muted)">You hold ${hold}. The deck has ${S.suits} of each rank.</div></div>
      </div>
      ${ui.loader ? LP.loaderHTML(d) : ''}`;

    if (d.turn !== 'over') { LP.bindCalc(root, d, bid, setLayout); LP.autoFit(c, root, 1.5); }
    root.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => act(b.dataset.act)));
    root.querySelectorAll('[data-seat]').forEach(b => b.addEventListener('click', () => send({type: 'seat', value: b.dataset.seat})));
    if (ui.loader) LP.bindLoader(root, c, ui, draw);
    const hs = root.querySelector('.hist'); hs.scrollTop = hs.scrollHeight;
    ui.lastPly = d.history.length - 1; ui.lastHand = d.hand_no;
  }

  function act(a) {
    if (a === 'loader') { ui.loader = true; draw(); return; }
    if (a === 'minraise') { if (d.turn === 'you' && d.legal.length) bid(d.legal[0]); return; }
    send({type: a});
  }

  LP.keys(c, e => {
    const k = e.key.toLowerCase();
    if (!d.loaded || ui.loader) { if (k === 'escape') { ui.loader = false; draw(); return true; } return false; }
    if (k === 'c') { if (d.can_call) act('call'); return true; }
    if (k === ' ') { act('minraise'); return true; }
    if (k === 'n' || (k === 'enter' && d.turn === 'over')) { act('new'); return true; }
    if (k === 'g') { setLayout(LP.nextCalcLayout(ui.layout)); return true; }
    if (k === 'r') { act('reveal'); return true; }
    if (k === 'h') { act('hints'); return true; }
    if (k === 'l') { act('loader'); return true; }
    return false;
  });
  draw();
}
"""

board = ab.make_component("arena_felt_calculator", css=CSS, js=JS)
ab.run(board, key="felt_board", background="#0e3b2c")
