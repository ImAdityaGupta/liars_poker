"""Vision 5 — Terminal.

Keyboard-first, huge type, zero chrome. You bid by typing the cards you are
claiming: `4` = one 4, `44` = pair of 4s, `444` = three 4s, `4422` = two
pair, `44422` = full house. Autocomplete shows the matching legal bids (with
odds) as you type; Tab cycles, Enter bids. Everything is also clickable.

Run:  streamlit run visuals/arena_5_terminal.py
Commands: c call · n new · peek · odds · first / second (who opens) · load · reset · help
"""

import streamlit as st

st.set_page_config(page_title="Liar's Poker · Terminal", layout="wide", initial_sidebar_state="collapsed")

import arena_backend as ab  # noqa: E402

CSS = r"""
:host { --bg:#0c0a06; --amber:#ffb454; --dim:#8a6a3a; --faint:#3b2e1a; --you:#ffd99a; --bot:#ff6b4a; --ok:#9be37a;
  font-family:'VT323','IBM Plex Mono',monospace; }
.lp-root { height:100vh; box-sizing:border-box; background:radial-gradient(120% 100% at 50% 45%, #1a140b 0%, var(--bg) 70%); color:var(--amber);
  display:grid; grid-template-rows:auto 1fr; overflow:hidden; position:relative; text-shadow:0 0 6px rgba(255,180,84,.45); }
.lp-root::after { content:''; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(0deg, rgba(0,0,0,.18) 0 1px, transparent 1px 3px); z-index:20; }
button { font-family:inherit; font-size:inherit; cursor:pointer; background:none; border:0; color:inherit; padding:0; text-shadow:inherit; }
.bar { display:flex; gap:26px; align-items:baseline; padding:10px 26px 8px; border-bottom:2px solid var(--faint); font-size:clamp(22px,2vw,30px); white-space:nowrap; overflow:hidden; }
.bar .t { color:var(--bg); background:var(--amber); padding:0 10px; text-shadow:none; }
.bar .opp { color:var(--dim); font-size:clamp(18px,1.6vw,26px); overflow:hidden; text-overflow:ellipsis; min-width:0; flex-shrink:1; } .bar .opp:hover { color:var(--amber); }
.bar .sc b { font-size:38px; } .bar .sc .y { color:var(--you); } .bar .sc .b { color:var(--bot); }
.bar .flags { margin-left:auto; color:var(--dim); font-size:clamp(18px,1.6vw,26px); flex-shrink:0; } .bar .flags .on { color:var(--amber); }
.bar .flags button { margin-left:14px; }
.main { display:grid; grid-template-columns:1fr minmax(360px,34vw); min-height:0; }
/* transcript + prompt */
.term { display:flex; flex-direction:column; min-height:0; padding:12px 26px 18px; }
.log { flex:1; overflow-y:auto; font-size:clamp(30px,2.6vw,40px); line-height:1.2; scrollbar-width:thin; scrollbar-color:var(--faint) transparent; }
.log .ln { white-space:pre-wrap; }
.log .sys { color:var(--dim); }
.log .you { color:var(--you); } .log .bot { color:var(--bot); }
.log .win { color:var(--ok); } .log .lose { color:var(--bot); }
.log .code { color:var(--dim); }
.log .fresh { animation:type .5s steps(12); } @keyframes type { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
.prompt { display:flex; align-items:center; gap:12px; font-size:clamp(36px,3.2vw,50px); border-top:2px solid var(--faint); padding-top:10px; }
.prompt .ps { color:var(--you); }
.prompt input { flex:1; background:transparent; border:0; outline:0; color:var(--you); font-family:inherit; font-size:inherit; caret-color:var(--amber); text-shadow:inherit; }
.prompt .err { color:var(--bot); font-size:28px; }
.sugg { display:flex; flex-wrap:wrap; gap:8px 10px; margin-top:10px; min-height:46px; font-size:28px; }
.s { border:1px solid var(--faint); padding:2px 10px; color:var(--amber); }
.s b { color:var(--you); font-weight:400; margin-right:8px; }
.s em { font-style:normal; color:var(--dim); margin-left:8px; }
.s.hi { background:var(--amber); color:var(--bg); text-shadow:none; } .s.hi b, .s.hi em { color:var(--bg); }
.s.call { border-color:var(--bot); color:var(--bot); } .s.call.hi { background:var(--bot); color:var(--bg); }
/* status pane */
.pane { border-left:2px solid var(--faint); padding:14px 22px; display:flex; flex-direction:column; gap:14px; min-height:0; overflow:auto; font-size:28px; }
.h { color:var(--dim); font-size:26px; letter-spacing:.1em; }
.cards { display:flex; gap:12px; margin-top:6px; }
.cd { width:clamp(52px,4.4vw,72px); aspect-ratio:5/7; border:3px double currentColor; display:flex; align-items:center; justify-content:center; font-size:clamp(44px,4vw,64px); box-shadow:0 0 12px rgba(255,180,84,.25), inset 0 0 12px rgba(255,180,84,.12); }
.cards.you { color:var(--you); } .cards.bot { color:var(--bot); }
.cd.back { background:repeating-linear-gradient(45deg, rgba(255,107,74,.35) 0 3px, transparent 3px 7px); }
.table { font-size:clamp(42px,3.8vw,60px); line-height:1.0; }
.table.you { color:var(--you); } .table.bot { color:var(--bot); }
.sub { color:var(--dim); font-size:26px; }
.sub b { color:var(--amber); font-weight:400; }
.keys { margin-top:auto; color:var(--dim); font-size:24px; line-height:1.3; }
.keys b { color:var(--amber); font-weight:400; }
.blink { animation:blink 1s steps(2) infinite; } @keyframes blink { 50% { opacity:0; } }
.lp-review { display:flex; flex-direction:column; gap:10px; }
.lp-review-head { color:var(--dim); font-size:24px; } .lp-review-head b { color:var(--amber); font-weight:400; }
.lp-review-row { display:grid; grid-template-columns:1fr 90px 50px; gap:10px; align-items:center; font-size:25px; }
.lp-review-row i { height:10px; background:linear-gradient(90deg,var(--bot) calc(var(--p)*100%), var(--faint) 0); }
.lp-review-row em { font-style:normal; text-align:right; color:var(--dim); }
.lp-review-row.chosen span { color:var(--bot); }
/* loader */
.lp-modal { position:absolute; inset:0; background:rgba(12,10,6,.92); display:flex; align-items:center; justify-content:center; z-index:30; }
.lp-modal-card { width:min(980px,94vw); max-height:86vh; display:flex; flex-direction:column; border:2px solid var(--amber); padding:16px 20px; background:var(--bg); }
.lp-modal-head { display:flex; font-size:30px; } .lp-modal-title { flex:1; } .lp-modal-title::before { content:'> '; }
.lp-x { font-size:30px; }
.lp-pathrow { display:flex; gap:10px; margin:10px 0; font-size:22px; }
.lp-path { flex:1; background:transparent; border:1px solid var(--faint); color:var(--you); font-family:inherit; font-size:22px; padding:4px 8px; outline:0; }
.lp-go { border:1px solid var(--amber); padding:0 16px; font-size:22px; }
.lp-err { color:var(--bot); font-size:22px; }
.lp-presets { overflow-y:auto; font-size:22px; }
.lp-group { color:var(--dim); margin-top:10px; } .lp-group::before { content:'# '; }
.lp-preset { display:flex; gap:20px; width:100%; text-align:left; padding:1px 6px; }
.lp-preset:hover, .lp-preset.current { background:var(--amber); color:var(--bg); text-shadow:none; }
.lp-preset-rel { color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .lp-preset:hover .lp-preset-rel { color:var(--bg); }
"""

JS = r"""
export default function (c) {
  LP.font('https://fonts.googleapis.com/css2?family=VT323&display=swap');
  const d = c.data;
  const root = LP.root(c);
  const ui = LP.ui(c, {loader: false, built: false, text: '', hi: 0, err: '', past: [], seenHand: 0, lastPly: -1, lastHand: -1, prevResult: null});
  const send = o => LP.send(c, o);

  // remember one-line summaries of finished hands so the log reads as a session
  if (d.loaded && ui.seenHand && ui.seenHand !== d.hand_no && ui.prevResult) ui.past.push(ui.prevResult);
  if (d.loaded) {
    ui.seenHand = d.hand_no;
    ui.prevResult = d.result ? `hand ${d.hand_no}: ${d.result.winner === 'you' ? 'WON' : 'LOST'} · ${d.result.caller === 'you' ? 'you called' : 'bot called'} ${d.result.claim.toLowerCase()} (${d.result.was_true ? 'true' : 'false'})` : null;
  }

  function cards(ranks, back, cls) {
    return `<div class="cards ${cls}">${ranks.map(r => `<div class="cd ${back ? 'back' : ''}">${back ? '' : r}</div>`).join('')}</div>`;
  }

  function options(text) {
    const t = text.trim().toLowerCase().replace(/\s+/g, '');
    if (!d.loaded || d.turn !== 'you') return [];
    const out = [];
    const legal = d.legal.map(i => d.claims[i]);
    if (d.can_call && ('call'.startsWith(t) || t === '')) out.push({call: true});
    const digits = /^\d+$/.test(t);
    for (const cl of legal) {
      const code = (cl.ranks[0] + cl.code).toLowerCase();
      if (t === '' || (digits && cl.short.startsWith(t)) || code.startsWith(t) || cl.label.toLowerCase().replace(/\s+/g, '').startsWith(t)) out.push(cl);
    }
    // exact card-string match first
    out.sort((a, b) => (b.short === t) - (a.short === t));
    return out.slice(0, t === '' ? 7 : 12);
  }

  function run(cmd) {
    const t = cmd.trim().toLowerCase();
    ui.err = '';
    if (!t) { const o = options(''); if (o[ui.hi]) pick(o[ui.hi]); return; }
    if (['n', 'new', 'deal'].includes(t)) return send({type: 'new'});
    if (['peek', 'r'].includes(t)) return send({type: 'reveal'});
    if (['odds', 'h'].includes(t)) return send({type: 'hints'});
    if (['first', 'second'].includes(t)) return send({type: 'seat', value: t});
    if (t === 'reset') return send({type: 'reset_score'});
    if (t === 'load') { ui.loader = true; return draw(); }
    if (t.startsWith('load ')) return send({type: 'load', path: cmd.trim().slice(5)});
    if (t === 'help' || t === '?') { ui.err = 'type the cards you claim: 4 · 44 · 444 · 4422 · 44422 — or c, n, peek, odds, first, second, load, reset'; return draw(); }
    const o = options(cmd);
    if (o[ui.hi]) return pick(o[ui.hi]);
    if (d.turn !== 'you') { ui.err = d.turn === 'bot' ? 'wait — the bot is thinking' : 'hand is over — type n'; return draw(); }
    ui.err = `no legal bid matches "${cmd.trim()}" — must beat ${d.last_claim == null ? 'nothing' : d.claims[d.last_claim].label.toLowerCase()}`;
    draw();
  }
  function pick(o) { ui.text = ''; ui.hi = 0; if (o.call) send({type: 'call'}); else send({type: 'claim', idx: o.idx}); }

  function build() {
    root.innerHTML = `<div class="bar"></div><div class="main"><div class="term"><div class="log"></div>
      <div class="prompt"><span class="ps">you&gt;</span><input spellcheck="false" autocomplete="off" placeholder=""><span class="err"></span></div><div class="sugg"></div></div>
      <div class="pane"></div></div><div class="modal-slot"></div>`;
    const inp = root.querySelector('input');
    inp.addEventListener('input', () => { ui.text = inp.value; ui.hi = 0; ui.err = ''; ui.api.drawSugg(); });
    inp.addEventListener('keydown', e => {
      const {options, drawSugg, run} = ui.api;
      const o = options(ui.text);
      if (e.key === 'Tab') { e.preventDefault(); if (o.length) { ui.hi = (ui.hi + (e.shiftKey ? o.length - 1 : 1)) % o.length; drawSugg(); } }
      else if (e.key === 'ArrowRight' && inp.selectionStart === inp.value.length && o.length) { e.preventDefault(); ui.hi = (ui.hi + 1) % o.length; drawSugg(); }
      else if (e.key === 'ArrowLeft' && inp.selectionStart === inp.value.length && ui.hi > 0) { e.preventDefault(); ui.hi -= 1; drawSugg(); }
      else if (e.key === 'Enter') { e.preventDefault(); const v = inp.value; inp.value = ''; ui.text = ''; run(v); }
      else if (e.key === 'Escape') { inp.value = ''; ui.text = ''; ui.hi = 0; drawSugg(); }
      e.stopPropagation();
    });
    root.addEventListener('click', e => { if (!e.target.closest('button, input, .lp-modal')) inp.focus(); });
    ui.built = true;
  }

  function drawSugg() {
    const o = options(ui.text);
    if (ui.hi >= o.length) ui.hi = 0;
    const box = root.querySelector('.sugg');
    box.innerHTML = o.map((x, i) => x.call
      ? `<button class="s call ${i === ui.hi ? 'hi' : ''}" data-i="${i}"><b>c</b>CALL ${d.last_claim != null ? d.claims[d.last_claim].label.toLowerCase() : ''}</button>`
      : `<button class="s ${i === ui.hi ? 'hi' : ''}" data-i="${i}"><b>${x.short}</b>${LP.esc(x.label.toLowerCase())}${d.hints ? `<em>${LP.pct(d.odds[x.idx])}</em>` : ''}</button>`).join('')
      || (d.turn === 'you' ? '<span class="s" style="border:0;color:var(--bot)">no match</span>' : '');
    box.querySelectorAll('[data-i]').forEach(b => b.addEventListener('click', () => pick(o[+b.dataset.i])));
    root.querySelector('.err').textContent = ui.err;
  }

  function draw() {
    ui.api = {options, run, pick, drawSugg, draw};
    if (!ui.built) build();
    const inp = root.querySelector('input');
    const S = d.spec || {};
    const bar = root.querySelector('.bar');
    if (!d.loaded) {
      bar.innerHTML = `<span class="t">LIARS-POKER</span><span>no opponent loaded — type <b>load</b></span>`;
    } else {
      bar.innerHTML = `<span class="t">LIARS-POKER</span><button class="opp" data-cmd="load">[${S.ranks}×${S.suits} deck · ${S.hand_size} cards · ${S.n_claims} claims · ${LP.esc(d.policy_kind)}]</button>
        <span class="sc">YOU <b class="y">${d.score.you}</b> : <b class="b">${d.score.bot}</b> BOT</span>
        <span class="flags"><button data-cmd="${d.you_open ? 'second' : 'first'}">[${(d.pending_seat || (d.you_open ? 'first' : 'second')) === 'first' ? 'you open' : 'bot opens'}]</button><button class="${d.hints ? 'on' : ''}" data-cmd="odds">[odds ${d.hints ? 'on' : 'off'}]</button><button class="${d.reveal ? 'on' : ''}" data-cmd="peek">[peek ${d.reveal ? 'on' : 'off'}]</button><button data-cmd="reset">[reset]</button></span>`;
    }

    // ---- transcript ----
    const log = root.querySelector('.log');
    let lines = ui.past.map(p => `<div class="ln sys">  ${LP.esc(p)}</div>`);
    if (d.loaded) {
      const hold = d.hand.map(x => x.rank).join(' ');
      lines.push(`<div class="ln sys">── hand ${d.hand_no} ── you hold ${hold} ── ${d.you_open ? 'you open' : 'bot opens'} ──</div>`);
      for (const h of d.history) {
        const fresh = (ui.lastHand === d.hand_no && h.ply > ui.lastPly) ? ' fresh' : '';
        const txt = h.idx < 0 ? 'CALL!' : `${h.label.toLowerCase()} <span class="code">[${h.short}]</span>`;
        lines.push(`<div class="ln ${h.who}${fresh}">${h.who === 'you' ? '&gt; you:' : '&lt; bot:'} ${txt}</div>`);
      }
      if (d.turn === 'bot') lines.push(`<div class="ln bot">&lt; bot: <span class="blink">▮</span></div>`);
      if (d.result) {
        const R = d.result;
        const W = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'];
        const many = (n, r) => `${W[n] || n} ${r}${n === 1 ? '' : 's'}`;
        const counts = R.actual.map(([r, n]) => many(n, r)).join(' + ');
        const need = R.need.map(([r, n]) => many(n, r)).join(' + ');
        lines.push(`<div class="ln sys">  showdown · bot held ${d.bot_hand.map(x => x.rank).join(' ')} · needed ${need} · there were ${counts}</div>`);
        lines.push(`<div class="ln ${R.winner === 'you' ? 'win' : 'lose'}">  ${R.winner === 'you' ? '*** YOU WIN ***' : '*** BOT WINS ***'} ${R.caller === 'you' ? 'your call' : 'bot called'} · claim was ${R.was_true ? 'TRUE' : 'FALSE'}</div>`);
        lines.push(`<div class="ln sys">  type n for the next hand</div>`);
      }
    }
    log.innerHTML = lines.join('');
    log.scrollTop = log.scrollHeight;

    // ---- status pane ----
    const pane = root.querySelector('.pane');
    if (d.loaded) {
      const last = LP.claimOf(d, d.last_claim);
      const lastBid = d.history.filter(h => h.idx >= 0).slice(-1)[0];
      const botRanks = d.bot_hand ? d.bot_hand.map(x => x.rank) : Array(d.bot_cards).fill(0);
      pane.innerHTML = `
        <div><div class="h">YOUR HAND</div>${cards(d.hand.map(x => x.rank), false, 'you')}</div>
        <div><div class="h">BOT ${d.turn === 'bot' ? '<span class="blink">· thinking</span>' : ''}</div>${cards(botRanks, !d.bot_hand, 'bot')}</div>
        <div><div class="h">ON THE TABLE</div>${last ? `<div class="table ${lastBid.who}">${LP.esc(last.label.toLowerCase())}</div>
          <div class="sub">${lastBid.who === 'you' ? 'your claim' : 'bot claims'} · cards [<b>${last.short}</b>]${d.hints ? ` · ${LP.pct(d.odds[last.idx])} vs random hand` : ''}</div>` : `<div class="table">—</div><div class="sub">${d.turn === 'you' ? 'your opening bid' : ''}</div>`}</div>
        ${d.result ? `<div><div class="h">WHAT THE BOT WAS WEIGHING</div>${LP.reviewHTML(d)}</div>` : ''}
        <div class="keys"><b>44</b> pair of 4s · <b>444</b> three · <b>4422</b> two pair · <b>44422</b> full house<br><b>tab</b> next match · <b>enter</b> bid · <b>c</b> call · <b>n</b> new · <b>help</b></div>`;
    } else pane.innerHTML = '';
    root.querySelectorAll('[data-cmd]').forEach(b => b.onclick = () => run(b.dataset.cmd));

    // ---- loader ----
    const slot = root.querySelector('.modal-slot');
    slot.innerHTML = (ui.loader || !d.loaded) ? LP.loaderHTML(d) : '';
    if (ui.loader || !d.loaded) LP.bindLoader(slot, c, ui, draw);

    inp.value = ui.text;
    drawSugg();
    if (!ui.loader && d.loaded) inp.focus({preventScroll: true});
    ui.lastPly = d.history ? d.history.length - 1 : -1; ui.lastHand = d.hand_no;
  }

  // keys typed while the input is not focused still reach the prompt
  LP.keys(c, e => {
    if (ui.loader) { if (e.key === 'Escape') { ui.loader = false; draw(); return true; } return false; }
    const inp = root.querySelector('input');
    if (inp && e.key.length === 1) { inp.focus(); return false; }
    return false;
  });
  draw();
}
"""

board = ab.make_component("arena_terminal", css=CSS, js=JS)
ab.run(board, key="terminal_board", background="#0c0a06")
