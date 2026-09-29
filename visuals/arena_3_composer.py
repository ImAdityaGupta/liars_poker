"""Vision 3 — Composer (purple).

A card-game-app look. The top row holds both hands side by side, the live
bid drawn as the cards it promises, and CALL. The rest of the screen is the
claim calculator (Table layout by default; Grid and Rows switchable from its
header or with G). The bidding so far runs as a chip track above it.

Run:  streamlit run visuals/arena_3_composer.py
Keys: C call · Space min-raise · N new hand · G button layout · R peek · H odds · L load
"""

import streamlit as st

st.set_page_config(page_title="Liar's Poker · Composer", layout="wide", initial_sidebar_state="collapsed")

import arena_backend as ab  # noqa: E402

CSS = r"""
:host { --bg1:#170d33; --bg2:#2b1760; --glass:rgba(255,255,255,.07); --line:rgba(255,255,255,.14); --ink:#f4f1ff; --soft:#a99ed0;
  --you:#c6f432; --bot:#ff4f8b; --card:#fffdf6; font-family:'Outfit','Inter',system-ui,sans-serif;
  --cc-bg:var(--card); --cc-fg:#1d1633; --cc-border:transparent; --cc-radius:11px; --cc-font:'Outfit',sans-serif; --cc-weight:800;
  --cc-fs:18px; --cc-h:38px; --cc-w:54px; --cc-shadow:0 4px 0 #bcb3dd; --cc-hover-bg:#fff; --cc-hover-shadow:0 6px 0 #bcb3dd;
  --cc-dead-bg:rgba(255,255,255,.06); --cc-dead-fg:rgba(255,255,255,.24); --cc-cur-bg:var(--bot); --cc-cur-fg:#fff; --cc-cur-ring:#fff;
  --cc-you:var(--you); --cc-bot:var(--bot); --cc-bar:#29c27a; --cc-odds-fg:#8a80b5; --cc-hdr:var(--soft);
  --cc-sw-border:var(--line); --cc-sw-on-bg:var(--ink); --cc-sw-on-fg:var(--bg1); --cc-read-strong:var(--ink); }
.lp-root { height:100vh; box-sizing:border-box; display:grid; grid-template-rows:auto auto auto 1fr; color:var(--ink); overflow:hidden;
  background:radial-gradient(90% 70% at 50% 35%, #3a2280 0%, var(--bg2) 45%, var(--bg1) 100%); }
button { font-family:inherit; cursor:pointer; color:inherit; }
/* top */
.top { display:flex; align-items:center; gap:10px; padding:12px 20px 4px; flex-wrap:wrap; }
.brand { font-size:22px; font-weight:800; letter-spacing:-.01em; margin-right:6px; } .brand b { color:var(--bot); }
.pill { background:var(--glass); border:1px solid var(--line); border-radius:999px; padding:7px 14px; font-size:14px; font-weight:600; }
.pill.on { background:var(--ink); color:var(--bg1); }
.pill small { color:var(--soft); font-weight:500; }
.grow { flex:1; }
.scorepill { display:flex; gap:14px; align-items:center; font-size:15px; font-weight:700; background:var(--glass); border:1px solid var(--line); border-radius:999px; padding:4px 18px; }
.scorepill b { font-size:24px; } .scorepill .y { color:var(--you); } .scorepill .b { color:var(--bot); }
/* table row: hands · bid · actions */
.tablerow { display:grid; grid-template-columns:auto 1fr auto; gap:28px; align-items:center; margin:10px 20px 0; padding:14px 22px; border-radius:24px;
  background:radial-gradient(closest-side at 50% 50%, rgba(198,244,50,.05), rgba(0,0,0,0)), rgba(255,255,255,.04); border:1px solid var(--line); }
.hands { display:flex; gap:26px; align-items:center; }
.seat { display:flex; align-items:center; gap:12px; }
.avatar { width:48px; height:48px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:13px; letter-spacing:.06em; flex-shrink:0; }
.avatar.bot { background:linear-gradient(135deg,var(--bot),#ff9a5a); color:#2a0716; }
.avatar.you { background:linear-gradient(135deg,var(--you),#6ef0b4); color:#1b2a00; }
.avatar.turn { box-shadow:0 0 0 4px rgba(255,255,255,.18), 0 0 26px rgba(255,255,255,.35); }
.fan { display:flex; }
.fan .card { margin-left:-10px; } .fan .card:first-child { margin-left:0; }
.card { width:clamp(46px,4vw,70px); aspect-ratio:5/7; border-radius:11px; background:var(--card); color:#1d1633; display:flex; align-items:center; justify-content:center; position:relative;
  font-weight:800; font-size:clamp(28px,2.7vw,46px); box-shadow:0 6px 16px rgba(0,0,0,.35); border:1px solid rgba(0,0,0,.08); }
.card .c { position:absolute; top:4px; left:7px; font-size:.3em; }
.card.back { background:linear-gradient(135deg,#ff4f8b,#7b3cff); border:3px solid var(--card); box-sizing:border-box; }
.youfan .card { transform:rotate(calc((var(--i) - 1.5) * 3deg)); }
.botfan .card { transform:rotate(calc((var(--i) - 1.5) * -3deg)); }
.card.hit { box-shadow:0 0 0 3px var(--you), 0 6px 16px rgba(0,0,0,.35); }
.vs { font-weight:800; color:var(--soft); font-size:13px; letter-spacing:.14em; }
.dots i { display:inline-block; width:7px; height:7px; margin:0 2px; border-radius:50%; background:var(--bot); animation:blink 1s infinite; }
.dots i:nth-child(2){animation-delay:.2s} .dots i:nth-child(3){animation-delay:.4s}
@keyframes blink { 0%,100%{opacity:.25} 50%{opacity:1} }
.bid { display:flex; flex-direction:column; align-items:center; gap:6px; text-align:center; min-width:0; }
.bid .lbl { font-size:12px; letter-spacing:.2em; text-transform:uppercase; color:var(--soft); font-weight:700; }
.bid .name { font-size:clamp(24px,2.4vw,38px); font-weight:800; line-height:1.05; }
.bid .name.you { color:var(--you); } .bid .name.bot { color:var(--bot); }
.bid .name.fresh { animation:pop .35s cubic-bezier(.2,1.6,.4,1) .25s backwards; }
@keyframes pop { from { transform:scale(.6); opacity:0; } }
.bid .odds { font-size:13px; color:var(--soft); } .bid .odds b { color:var(--ink); }
.promise { display:flex; gap:5px; }
.promise .card { width:30px; font-size:18px; border-radius:7px; box-shadow:0 3px 8px rgba(0,0,0,.3); }
.promise .card .c { display:none; }
.promise .card.mine { box-shadow:0 0 0 2px var(--you); }
.promise .card.unk { background:rgba(255,255,255,.1); color:var(--ink); border:2px dashed rgba(255,255,255,.4); box-shadow:none; }
.acts { display:flex; flex-direction:column; gap:8px; min-width:200px; }
.callbtn { border:0; border-radius:16px; background:var(--bot); color:#fff; font-size:32px; font-weight:900; letter-spacing:.06em; padding:10px 18px; box-shadow:0 6px 0 #b3285c; }
.callbtn small { font-size:13px; opacity:.6; margin-left:6px; }
.callbtn:disabled { background:rgba(255,255,255,.08); color:rgba(255,255,255,.3); box-shadow:none; cursor:default; }
.minbtn { border:1px solid var(--line); background:var(--glass); border-radius:12px; padding:7px 12px; font-size:15px; font-weight:700; text-align:left; }
.minbtn small { display:block; color:var(--soft); font-size:11px; font-weight:600; }
.minbtn:disabled { opacity:.35; cursor:default; }
/* history track */
.track { display:flex; gap:6px; flex-wrap:wrap; align-items:center; padding:10px 24px 0; min-height:30px; }
.track .t0 { font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); font-weight:700; margin-right:6px; }
.track span.b { font-size:15px; font-weight:700; padding:4px 12px; border-radius:999px; }
.track .you { background:rgba(198,244,50,.16); color:var(--you); } .track .bot { background:rgba(255,79,139,.16); color:var(--bot); }
.track .arrow { color:var(--soft); }
.track .fresh { animation:pop .3s ease-out; } .track .bot.fresh { animation:pop .3s ease-out .25s backwards; }
/* calculator panel */
.board { position:relative; margin:12px 20px 18px; padding:14px 18px; border-radius:22px; background:rgba(10,6,26,.45); border:1px solid var(--line); min-height:0; display:flex; flex-direction:column; }
.board .calc-frame { flex:1; }
.lp-pending .cc, .lp-pending .callbtn, .lp-pending .minbtn { pointer-events:none; }
/* result */
.res { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; overflow:auto; }
.res .v { font-size:60px; font-weight:900; line-height:1; } .res .v.you { color:var(--you); } .res .v.bot { color:var(--bot); }
.res p { font-size:19px; margin:10px 0 14px; color:#ddd6ff; }
.again { border:0; border-radius:16px; background:var(--you); color:#1b2a00; font-size:21px; font-weight:800; padding:12px 32px; box-shadow:0 6px 0 #8aad14; }
.lp-review { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top:16px; text-align:left; }
.lp-review-move { background:var(--glass); border-radius:14px; padding:10px 12px; min-width:240px; }
.lp-review-head { font-size:13px; color:var(--soft); margin-bottom:6px; } .lp-review-head b { color:var(--ink); }
.lp-review-row { display:grid; grid-template-columns:1fr 70px 40px; gap:8px; align-items:center; font-size:14px; padding:2px 0; }
.lp-review-row i { height:6px; border-radius:3px; background:linear-gradient(90deg,var(--bot) calc(var(--p)*100%), rgba(255,255,255,.1) 0); }
.lp-review-row em { font-style:normal; text-align:right; color:var(--soft); }
.lp-review-row.chosen span { color:var(--bot); font-weight:800; }
/* loader */
.lp-modal { position:fixed; inset:0; background:rgba(10,5,25,.7); display:flex; align-items:center; justify-content:center; z-index:50; }
.lp-modal-card { width:min(820px,92vw); max-height:84vh; display:flex; flex-direction:column; background:#231451; border:1px solid var(--line); border-radius:22px; padding:20px; }
.lp-modal-head { display:flex; align-items:center; } .lp-modal-title { font-size:28px; font-weight:800; flex:1; }
.lp-x { background:none; border:0; font-size:30px; }
.lp-pathrow { display:flex; gap:8px; margin:12px 0; }
.lp-path { flex:1; background:var(--glass); border:1px solid var(--line); color:var(--ink); border-radius:12px; padding:10px 12px; font-size:14px; }
.lp-go { background:var(--you); color:#1b2a00; border:0; border-radius:12px; padding:0 22px; font-weight:800; }
.lp-err { color:var(--bot); margin-bottom:8px; }
.lp-presets { overflow-y:auto; display:flex; flex-direction:column; gap:4px; }
.lp-group { margin:12px 0 4px; font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); font-weight:700; }
.lp-preset { display:flex; justify-content:space-between; gap:12px; text-align:left; background:var(--glass); border:1px solid transparent; border-radius:10px; padding:8px 12px; }
.lp-preset:hover { border-color:var(--you); } .lp-preset.current { border-color:var(--bot); }
.lp-preset-game { font-weight:700; font-size:14px; white-space:nowrap; } .lp-preset-rel { color:var(--soft); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.waiting { display:flex; align-items:center; justify-content:center; font-size:22px; color:var(--soft); }
/* other screen shapes */
@media (orientation: portrait), (max-width: 1150px) {
  .top .grow { display:none; } .scorepill { margin-left:auto; }
  .tablerow { grid-template-columns:1fr auto; gap:14px 18px; margin:10px 12px 0; }
  .hands { grid-column:1 / -1; justify-content:center; flex-wrap:wrap; }
  .board { margin:10px 12px 12px; }
}
@media (min-width: 2200px) and (min-height: 1250px) { .lp-root { zoom:1.35; height:calc(100vh / 1.35); } }
@media (min-width: 3200px) and (min-height: 1800px) { .lp-root { zoom:1.8; height:calc(100vh / 1.8); } }
"""

JS = r"""
export default function (c) {
  LP.font('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&display=swap');
  let d = c.data;
  const root = LP.root(c);
  const ui = LP.ui(c, {loader: false, lastPly: -1, lastHand: -1, layout: LP.calcLayout('composer', 'table')});
  const send = o => LP.send(c, o);

  const card = (rank, suit, cls = '', i = 0) =>
    `<div class="card ${cls}" style="--i:${i}"><span class="c">${rank}${suit == null ? '' : '♠♥♦♣'[suit]}</span>${rank}</div>`;

  function bid(idx) {
    if (d.turn !== 'you' || !d.legal.includes(idx)) return;
    d = LP.optimistic(d, idx);
    draw();
    send({type: 'claim', idx});
  }
  function setLayout(v) { ui.layout = v; LP.setCalcLayout('composer', v); draw(); }

  function draw() {
    if (!d.loaded) {
      root.innerHTML = `<div class="top"><div class="brand">Liar's <b>Poker</b></div></div><div></div><div></div><div class="waiting">Load an opponent to begin.</div>` + LP.loaderHTML(d);
      LP.bindLoader(root, c, ui, draw); return;
    }
    const S = d.spec;
    const mine = Object.fromEntries(d.mine);
    const last = LP.claimOf(d, d.last_claim);
    const lastBid = d.history.filter(h => h.idx >= 0).slice(-1)[0];
    const needRanks = last ? new Set(last.need.map(x => x[0])) : new Set();
    const fresh = h => !(ui.lastHand === d.hand_no && h.ply <= ui.lastPly);

    // ---- hands ----
    const youCards = d.hand.map((x, i) => card(x.rank, x.suit, needRanks.has(x.rank) ? 'hit' : '', i)).join('');
    const botCards = d.bot_hand ? d.bot_hand.map((x, i) => card(x.rank, x.suit, needRanks.has(x.rank) ? 'hit' : '', i)).join('')
                                : Array.from({length: d.bot_cards}, (_, i) => `<div class="card back" style="--i:${i}"></div>`).join('');

    // ---- the live bid, drawn as the cards it promises ----
    let bidHTML;
    if (last) {
      let prom = '';
      for (const [r, n] of last.need) {
        const have = Math.min(mine[r] || 0, n);
        for (let j = 0; j < n; j++) prom += j < have ? card(r, null, 'mine') : `<div class="card unk">${r}</div>`;
      }
      bidHTML = `<div class="lbl">${lastBid.who === 'you' ? 'You claimed' : 'Bot claims'}</div>
        <div class="name ${lastBid.who} ${lastBid.who === 'bot' && fresh(lastBid) ? 'fresh' : ''}">${LP.esc(last.label)}</div>
        <div class="promise" title="Green: cards you hold. Dashed: must be in the bot's hand">${prom}</div>
        ${d.hints && d.turn !== 'over' ? `<div class="odds">If the bot's cards were random, true <b>${LP.pct(d.odds[last.idx])}</b> of the time</div>` : ''}`;
    } else {
      bidHTML = `<div class="lbl">No bid yet</div><div class="name">${d.turn === 'you' ? 'You open' : 'Bot opens…'}</div>`;
    }
    if (d.turn === 'bot') bidHTML += `<div class="odds">Bot is thinking <span class="dots"><i></i><i></i><i></i></span></div>`;

    const min = d.turn === 'you' && d.legal.length ? d.claims[d.legal[0]] : null;
    const track = `<span class="t0">Bidding</span>` + (d.history.length
      ? d.history.map((h, i) => `${i ? '<span class="arrow">›</span>' : ''}<span class="b ${h.who} ${fresh(h) ? 'fresh' : ''}">${LP.esc(h.label)}</span>`).join('')
      : `<span style="color:var(--soft);font-size:14px">nothing yet</span>`);

    // ---- board: calculator, or the showdown ----
    let board;
    if (d.turn === 'over') {
      const R = d.result;
      const story = R.caller === 'you'
        ? `You called the bot's <b>${LP.esc(R.claim)}</b>. ${R.was_true ? 'It was true.' : 'It was a bluff!'}`
        : `The bot called your <b>${LP.esc(R.claim)}</b>. ${R.was_true ? 'It was true.' : 'It was false.'}`;
      const actual = R.actual.map(([r, n]) => `${n} × ${r}`).join(', ');
      const needTxt = R.need.map(([r, n]) => `${n} × ${r}`).join(', ');
      board = `<div class="res"><div class="v ${R.winner}">${R.winner === 'you' ? 'YOU WIN' : 'BOT WINS'}</div>
        <p>${story}<br><span style="color:var(--soft)">Needed ${needTxt} · there were ${actual}</span></p>
        <button class="again" data-act="new">Deal again · N</button>${LP.reviewHTML(d)}</div>`;
    } else {
      board = LP.calcFrame(d, ui.layout);
    }

    const seatNow = d.pending_seat || (d.you_open ? 'first' : 'second');
    root.innerHTML = `
      <div class="top">
        <div class="brand">Liar's <b>Poker</b></div>
        <button class="pill" data-act="loader">${S.ranks}×${S.suits} deck · ${S.hand_size} cards · ${S.n_claims} claims <small>· ${LP.esc(d.policy_kind)} ▾</small></button>
        <div class="grow"></div>
        <div class="scorepill"><span>You</span><b class="y">${d.score.you}</b><b class="b">${d.score.bot}</b><span>Bot</span></div>
        <div class="grow"></div>
        <button class="pill ${seatNow === 'first' ? 'on' : ''}" data-seat="first">You open</button>
        <button class="pill ${seatNow === 'second' ? 'on' : ''}" data-seat="second">Bot opens</button>
        <button class="pill ${d.hints ? 'on' : ''}" data-act="hints" title="Chance each claim is true if the bot held random cards">Odds %</button>
        <button class="pill ${d.reveal ? 'on' : ''}" data-act="reveal">Peek</button>
        <button class="pill" data-act="new">New hand</button>
        <button class="pill" data-act="reset_score">Reset</button>
      </div>
      <div class="tablerow">
        <div class="hands">
          <div class="seat"><div class="avatar you ${d.turn === 'you' ? 'turn' : ''}">YOU</div><div class="fan youfan">${youCards}</div></div>
          <div class="vs">VS</div>
          <div class="seat"><div class="fan botfan">${botCards}</div><div class="avatar bot ${d.turn === 'bot' ? 'turn' : ''}">BOT</div></div>
        </div>
        <div class="bid">${bidHTML}</div>
        <div class="acts">
          <button class="callbtn" data-act="call" ${d.can_call ? '' : 'disabled'}>CALL<small>C</small></button>
          <button class="minbtn" data-act="minraise" ${min ? '' : 'disabled'}><small>Min raise · Space</small>${min ? LP.esc(min.label) : '—'}</button>
        </div>
      </div>
      <div class="track">${track}</div>
      <div class="board">${board}</div>
      ${ui.loader ? LP.loaderHTML(d) : ''}`;

    if (d.turn !== 'over') { LP.bindCalc(root, d, bid, setLayout); LP.autoFit(c, root, 1.6); }
    root.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => act(b.dataset.act)));
    root.querySelectorAll('[data-seat]').forEach(b => b.addEventListener('click', () => send({type: 'seat', value: b.dataset.seat})));
    if (ui.loader) LP.bindLoader(root, c, ui, draw);
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

board = ab.make_component("arena_composer", css=CSS, js=JS)
ab.run(board, key="composer_board", background="#170d33")
