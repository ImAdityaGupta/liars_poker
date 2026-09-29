"""Vision 2 — Paper (formerly the Ladder).

Editorial paper-and-ink styling with the claim calculator. The top strip
holds both hands side by side, the live claim in large serif type, and the
Call / Min-raise buttons. The calculator (Rows layout by default; Grid and
Table switchable from its header or with G) fills the page, with the
bidding log on the right.

Run:  streamlit run visuals/arena_2_ladder.py
Keys: C call · Space min-raise · N new hand · G button layout · R peek · H odds · L load
"""

import streamlit as st

st.set_page_config(page_title="Liar's Poker · Paper", layout="wide", initial_sidebar_state="collapsed")

import arena_backend as ab  # noqa: E402

CSS = r"""
:host { --paper:#f3ede2; --paper2:#e9e1d2; --ink:#1d1a16; --soft:#6f665a; --line:#d6cbb8; --you:#1f5fbf; --bot:#c8412b; --gold:#b8862b;
  font-family:'Instrument Sans','Inter',system-ui,sans-serif;
  --cc-bg:#fffdf8; --cc-fg:var(--ink); --cc-border:var(--ink); --cc-radius:8px; --cc-font:'Fraunces',Georgia,serif; --cc-weight:700;
  --cc-fs:18px; --cc-h:36px; --cc-w:50px; --cc-shadow:3px 3px 0 var(--ink); --cc-hover-bg:#fff6dc; --cc-hover-shadow:4px 4px 0 var(--ink);
  --cc-dead-bg:transparent; --cc-dead-fg:#b9ad99; --cc-cur-bg:var(--ink); --cc-cur-fg:var(--paper); --cc-cur-ring:var(--ink);
  --cc-you:var(--you); --cc-bot:var(--bot); --cc-bar:var(--gold); --cc-odds-fg:var(--soft); --cc-odds-font:'Instrument Sans',sans-serif; --cc-hdr:var(--soft);
  --cc-sw-border:var(--line); --cc-sw-on-bg:var(--ink); --cc-sw-on-fg:var(--paper); --cc-read-strong:var(--ink); }
.lp-root { height:100vh; box-sizing:border-box; display:grid; grid-template-rows:auto auto 1fr; background:var(--paper); color:var(--ink); overflow:hidden; }
button { font-family:inherit; cursor:pointer; color:inherit; }
.cc.dead:not(.cur) { border:1px dashed var(--line) !important; }
.cc.cur { box-shadow:3px 3px 0 var(--bot) !important; }
/* top */
.top { display:flex; align-items:center; gap:12px; padding:12px 22px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
.brand { font-family:'Fraunces',Georgia,serif; font-size:28px; font-weight:800; letter-spacing:-.01em; line-height:1; margin-right:4px; }
.brand i { font-weight:400; color:var(--bot); }
.opp { text-align:left; background:#fffdf8; border:1px solid var(--line); border-radius:10px; padding:5px 12px; }
.opp small { display:block; color:var(--soft); font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
.opp b { font-size:14px; }
.grow { flex:1; }
.score { display:flex; align-items:baseline; gap:10px; font-family:'Fraunces',Georgia,serif; }
.score b { font-size:34px; line-height:1; } .score .y { color:var(--you); } .score .b { color:var(--bot); }
.score span { color:var(--soft); font-size:13px; font-family:'Instrument Sans',sans-serif; }
.seg { display:flex; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.seg button { background:#fffdf8; border:0; padding:7px 11px; font-size:13px; font-weight:600; color:var(--soft); }
.seg button.on { background:var(--ink); color:var(--paper); }
.ctl { background:#fffdf8; border:1px solid var(--line); border-radius:10px; padding:7px 11px; font-size:13px; font-weight:600; }
.ctl kbd { font-family:inherit; font-size:11px; color:var(--soft); margin-left:4px; }
.ctl.on { border-color:var(--ink); box-shadow:inset 0 0 0 1px var(--ink); }
/* strip */
.strip { display:grid; grid-template-columns:auto 1fr auto; gap:30px; align-items:center; padding:16px 22px; border-bottom:1px solid var(--line); background:var(--paper2); }
.hands { display:flex; gap:24px; align-items:flex-end; }
.label { font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); font-weight:700; margin-bottom:7px; display:flex; gap:8px; align-items:center; }
.label.you { color:var(--you); } .label.bot { color:var(--bot); }
.cards { display:flex; gap:8px; }
.card { width:clamp(42px,3.6vw,62px); aspect-ratio:5/7; border-radius:8px; background:#fffdf8; border:1.5px solid var(--ink); display:flex; align-items:center; justify-content:center; position:relative;
  font-family:'Fraunces',Georgia,serif; font-weight:800; font-size:clamp(26px,2.4vw,40px); box-shadow:3px 3px 0 var(--ink); }
.card .c { position:absolute; top:3px; left:5px; font-size:.32em; }
.card.back { background:repeating-linear-gradient(-45deg,var(--bot) 0 5px,#e7765f 5px 10px); }
.card.hit { box-shadow:3px 3px 0 var(--you); border-color:var(--you); }
.thinking i { display:inline-block; width:6px; height:6px; margin-left:3px; border-radius:50%; background:var(--bot); animation:blink 1s infinite; }
.thinking i:nth-child(2){animation-delay:.2s} .thinking i:nth-child(3){animation-delay:.4s}
@keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
.spot { padding-left:28px; border-left:1px solid var(--line); min-width:0; }
.spot small { display:block; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); font-weight:700; }
.spot .claim { font-family:'Fraunces',Georgia,serif; font-weight:800; font-size:clamp(28px,3vw,46px); line-height:1.05; margin:4px 0 2px; }
.spot .claim.you { color:var(--you); } .spot .claim.bot { color:var(--bot); }
.spot .claim.fresh { animation:inkIn .35s ease-out .25s backwards; }
@keyframes inkIn { from { opacity:0; transform:translateY(5px); } }
.spot .meta { font-size:14px; color:var(--soft); } .spot .meta b { color:var(--ink); }
.acts { display:flex; flex-direction:column; gap:8px; min-width:210px; }
.big { border:2px solid var(--ink); border-radius:12px; padding:10px 14px; font-size:17px; font-weight:800; text-align:left; display:flex; justify-content:space-between; align-items:center; gap:12px; box-shadow:4px 4px 0 var(--ink); background:#fffdf8; }
.big small { font-size:11px; font-weight:700; opacity:.7; letter-spacing:.1em; }
.big:active { transform:translate(2px,2px); box-shadow:2px 2px 0 var(--ink); }
.big:disabled { opacity:.35; cursor:default; box-shadow:none; }
.callb { background:var(--bot); color:#fff; font-size:26px; font-family:'Fraunces',Georgia,serif; }
.big em { font-style:normal; font-family:'Fraunces',Georgia,serif; }
/* main */
.main { display:grid; grid-template-columns:1fr minmax(240px,22vw); min-height:0; }
.work { padding:16px 22px; display:flex; flex-direction:column; min-height:0; }
.work .calc-frame { flex:1; }
.side { border-left:1px solid var(--line); padding:16px 20px; display:flex; flex-direction:column; min-height:0; }
.log { flex:1; overflow:auto; margin:0; padding:0; list-style:none; display:flex; flex-direction:column; gap:6px; }
.log li { display:grid; grid-template-columns:22px 34px 1fr; gap:6px; align-items:baseline; font-size:17px; padding-bottom:6px; border-bottom:1px solid var(--line); }
.log li .n { font-size:11px; color:var(--soft); }
.log li b { font-family:'Fraunces',Georgia,serif; }
.log li .w { font-size:10px; font-weight:800; letter-spacing:.1em; }
.log li.you .w { color:var(--you); } .log li.bot .w { color:var(--bot); }
.log li.fresh { animation:inkIn .3s ease-out; } .log li.bot.fresh { animation:inkIn .3s ease-out .25s backwards; }
.holding { font-size:13px; color:var(--soft); margin-top:10px; line-height:1.45; } .holding b { color:var(--ink); }
.lp-pending .cc, .lp-pending .big { pointer-events:none; }
/* result */
.result { flex:1; overflow:auto; display:flex; flex-direction:column; gap:10px; max-width:900px; }
.result .verdict { font-family:'Fraunces',Georgia,serif; font-size:54px; font-weight:800; line-height:1; }
.result .verdict.you { color:var(--you); } .result .verdict.bot { color:var(--bot); }
.result p { font-size:18px; line-height:1.45; margin:4px 0; }
.tally { display:flex; gap:6px; flex-wrap:wrap; }
.tally div { border:1px solid var(--line); background:#fffdf8; border-radius:8px; padding:4px 10px; text-align:center; min-width:40px; }
.tally div.need { border:2px solid var(--ink); }
.tally b { display:block; font-family:'Fraunces',Georgia,serif; font-size:24px; }
.tally small { font-size:11px; color:var(--soft); }
.result .big { align-self:flex-start; }
.lp-review { display:flex; gap:10px; flex-wrap:wrap; }
.lp-review-move { background:#fffdf8; border:1px solid var(--line); border-radius:10px; padding:10px 12px; min-width:240px; }
.lp-review-head { font-size:13px; color:var(--soft); margin-bottom:6px; } .lp-review-head b { color:var(--ink); }
.lp-review-row { display:grid; grid-template-columns:1fr 80px 40px; gap:8px; align-items:center; font-size:14px; padding:2px 0; }
.lp-review-row i { height:6px; border-radius:3px; background:linear-gradient(90deg,var(--bot) calc(var(--p)*100%), var(--paper2) 0); }
.lp-review-row em { font-style:normal; text-align:right; color:var(--soft); }
.lp-review-row.chosen span { color:var(--bot); font-weight:800; }
/* loader */
.lp-modal { position:fixed; inset:0; background:rgba(29,26,22,.45); display:flex; align-items:center; justify-content:center; z-index:50; }
.lp-modal-card { width:min(820px,92vw); max-height:84vh; display:flex; flex-direction:column; background:var(--paper); border:2px solid var(--ink); border-radius:16px; padding:20px; box-shadow:8px 8px 0 var(--ink); }
.lp-modal-head { display:flex; align-items:center; } .lp-modal-title { font-family:'Fraunces',Georgia,serif; font-size:30px; font-weight:800; flex:1; }
.lp-x { background:none; border:0; font-size:30px; }
.lp-pathrow { display:flex; gap:8px; margin:12px 0; }
.lp-path { flex:1; background:#fffdf8; border:1px solid var(--line); border-radius:10px; padding:10px 12px; font-size:14px; }
.lp-go { background:var(--ink); color:var(--paper); border:0; border-radius:10px; padding:0 22px; font-weight:800; }
.lp-err { color:var(--bot); margin-bottom:8px; }
.lp-presets { overflow-y:auto; display:flex; flex-direction:column; gap:3px; }
.lp-group { margin:12px 0 4px; font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--soft); font-weight:700; }
.lp-preset { display:flex; justify-content:space-between; gap:12px; text-align:left; background:#fffdf8; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
.lp-preset:hover { border-color:var(--ink); } .lp-preset.current { border:2px solid var(--you); }
.lp-preset-game { font-weight:700; font-size:14px; white-space:nowrap; } .lp-preset-rel { color:var(--soft); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
/* other screen shapes */
@media (orientation: portrait), (max-width: 1150px) {
  .top .grow { display:none; } .score { margin-left:auto; }
  .strip { grid-template-columns:1fr auto; gap:14px 20px; }
  .hands { grid-column:1 / -1; }
  .spot { border-left:0; padding-left:0; }
}
@media (orientation: portrait) {
  .main { grid-template-columns:1fr; grid-template-rows:1fr auto; }
  .side { border-left:0; border-top:1px solid var(--line); height:min(22vh,230px); }
}
@media (min-width: 2200px) and (min-height: 1250px) { .lp-root { zoom:1.35; height:calc(100vh / 1.35); } }
@media (min-width: 3200px) and (min-height: 1800px) { .lp-root { zoom:1.8; height:calc(100vh / 1.8); } }
"""

JS = r"""
export default function (c) {
  LP.font('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700;9..144,800&family=Instrument+Sans:wght@400;600;700;800&display=swap');
  let d = c.data;
  const root = LP.root(c);
  const ui = LP.ui(c, {loader: false, lastPly: -1, lastHand: -1, layout: LP.calcLayout('paper', 'rows')});
  const send = o => LP.send(c, o);
  const words = ['', 'one', 'two', 'three', 'four', 'five', 'six'];
  const card = (cd, cls = '') => `<div class="card ${cls}"><span class="c">${cd.rank}${cd.suit == null ? '' : '♠♥♦♣'[cd.suit]}</span>${cd.rank}</div>`;

  function bid(idx) {
    if (d.turn !== 'you' || !d.legal.includes(idx)) return;
    d = LP.optimistic(d, idx);
    draw();
    send({type: 'claim', idx});
  }
  function setLayout(v) { ui.layout = v; LP.setCalcLayout('paper', v); draw(); }

  function draw() {
    if (!d.loaded) {
      root.innerHTML = `<div class="top"><div class="brand">Liar's <i>Poker</i></div></div><div></div><p style="padding:30px">Load an opponent to begin.</p>` + LP.loaderHTML(d);
      LP.bindLoader(root, c, ui, draw); return;
    }
    const S = d.spec;
    const last = LP.claimOf(d, d.last_claim);
    const lastBy = d.history.filter(h => h.idx >= 0).slice(-1)[0];
    const need = last ? new Set(last.need.map(x => x[0])) : new Set();
    const fresh = h => !(ui.lastHand === d.hand_no && h.ply <= ui.lastPly);

    let work;
    if (d.turn === 'over') {
      const R = d.result;
      let tally = '<div class="tally">';
      for (const [r, n] of R.counts) {
        const nd = R.need.find(x => x[0] === r);
        tally += `<div class="${nd ? 'need' : ''}"><b>${n}</b><small>${r}s${nd ? ' · need ' + nd[1] : ''}</small></div>`;
      }
      tally += '</div>';
      const story = R.caller === 'you'
        ? `You called the bot's <b>${LP.esc(R.claim)}</b>. It was ${R.was_true ? '<b>true</b> — the bot had it.' : '<b>a bluff</b>.'}`
        : `The bot called your <b>${LP.esc(R.claim)}</b>. It was ${R.was_true ? '<b>true</b>.' : '<b>false</b>.'}`;
      work = `<div class="result"><div class="label">Hand ${d.hand_no} · showdown</div><div class="verdict ${R.winner}">${R.winner === 'you' ? 'You win.' : 'Bot wins.'}</div>
        <p>${story}</p><div class="label">Cards out, by rank</div>${tally}
        <button class="big" data-act="new"><span>Deal the next hand</span><small>N / ENTER</small></button>
        <div class="label" style="margin-top:6px">What the bot was weighing</div>${LP.reviewHTML(d)}</div>`;
    } else {
      work = LP.calcFrame(d, ui.layout);
    }

    const botCards = d.bot_hand ? d.bot_hand.map(x => card(x, need.has(x.rank) ? 'hit' : '')).join('') : Array(d.bot_cards).fill('<div class="card back"></div>').join('');
    const hold = d.mine.filter(x => x[1]).reverse().map(x => x[1] === 1 ? `one ${x[0]}` : `${words[x[1]] || x[1]} ${x[0]}s`).join(', ');
    const min = d.turn === 'you' && d.legal.length ? d.claims[d.legal[0]] : null;
    const spot = last
      ? `<small>${lastBy.who === 'you' ? 'You claimed' : 'The bot claims'}</small><div class="claim ${lastBy.who} ${lastBy.who === 'bot' && fresh(lastBy) ? 'fresh' : ''}">${LP.esc(last.label)}</div>
         <div class="meta">${d.hints && d.turn !== 'over' ? `If the bot's cards were random, this is true <b>${LP.pct(d.odds[last.idx])}</b> of the time.` : ''}</div>`
      : `<small>Opening bid</small><div class="claim">${d.turn === 'you' ? 'You open.' : 'Bot opens…'}</div>`;
    const log = d.history.map(h => `<li class="${h.who} ${fresh(h) ? 'fresh' : ''}"><span class="n">${h.ply + 1}</span><span class="w">${h.who === 'you' ? 'YOU' : 'BOT'}</span><b>${LP.esc(h.label)}</b></li>`).join('')
      || '<li style="border:0;color:var(--soft);font-size:15px;display:block">No bids yet.</li>';
    const seatNow = d.pending_seat || (d.you_open ? 'first' : 'second');

    root.innerHTML = `
      <div class="top">
        <div class="brand">Liar's <i>Poker</i></div>
        <button class="opp" data-act="loader"><small>Opponent · ${LP.esc(d.policy_kind)} · change</small><b>${S.ranks} ranks · ${S.suits} suits · ${S.hand_size} cards · ${S.n_claims} claims</b></button>
        <div class="grow"></div>
        <div class="score"><b class="y">${d.score.you}</b><span>you</span><b class="b">${d.score.bot}</b><span>bot</span></div>
        <div class="grow"></div>
        <div class="seg"><button data-seat="first" class="${seatNow === 'first' ? 'on' : ''}">You open</button><button data-seat="second" class="${seatNow === 'second' ? 'on' : ''}">Bot opens</button></div>
        <button class="ctl ${d.hints ? 'on' : ''}" data-act="hints" title="Chance each claim is true if the bot held random cards">Odds %<kbd>H</kbd></button>
        <button class="ctl ${d.reveal ? 'on' : ''}" data-act="reveal">Peek<kbd>R</kbd></button>
        <button class="ctl" data-act="new">New hand<kbd>N</kbd></button>
        <button class="ctl" data-act="reset_score">Reset score</button>
      </div>
      <div class="strip">
        <div class="hands">
          <div><div class="label you">Your hand</div><div class="cards">${d.hand.map(x => card(x, need.has(x.rank) ? 'hit' : '')).join('')}</div></div>
          <div><div class="label bot">Bot's hand ${d.turn === 'bot' ? '<span class="thinking"><i></i><i></i><i></i></span>' : ''}</div><div class="cards">${botCards}</div></div>
        </div>
        <div class="spot">${spot}</div>
        <div class="acts">
          <button class="big callb" data-act="call" ${d.can_call ? '' : 'disabled'}><span>Call it</span><small>C</small></button>
          <button class="big" data-act="minraise" ${min ? '' : 'disabled'}><span>Min raise · <em>${min ? LP.esc(min.label) : '—'}</em></span><small>SPACE</small></button>
        </div>
      </div>
      <div class="main">
        <div class="work">${work}</div>
        <div class="side"><div class="label">Bidding · hand ${d.hand_no}</div><ol class="log">${log}</ol>
          <div class="holding">You hold <b>${hold}</b>. There are ${S.suits} of each rank in the deck.</div></div>
      </div>
      ${ui.loader ? LP.loaderHTML(d) : ''}`;

    if (d.turn !== 'over') { LP.bindCalc(root, d, bid, setLayout); LP.autoFit(c, root, 1.5); }
    root.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => act(b.dataset.act)));
    root.querySelectorAll('[data-seat]').forEach(b => b.addEventListener('click', () => send({type: 'seat', value: b.dataset.seat})));
    if (ui.loader) LP.bindLoader(root, c, ui, draw);
    const lg = root.querySelector('.log'); if (lg) lg.scrollTop = lg.scrollHeight;
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

board = ab.make_component("arena_paper", css=CSS, js=JS)
ab.run(board, key="paper_board", background="#f3ede2")
