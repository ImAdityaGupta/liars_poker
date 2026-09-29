"""Shared game backend for the arena_* play UIs.

Python owns the game (Env, policy, score); each arena_* file supplies only a
custom HTML/CSS/JS board built with ``st.components.v2``. The board receives a
JSON payload from :meth:`Arena.payload` and sends actions back as a trigger
value ``action`` = ``{"type": ..., ...}``.

Action types understood by :meth:`Arena.apply`:
    claim {idx}, call, new, reset_score, reveal, hints,
    seat {value: "first" | "second"}, load {path}
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import gc

import streamlit as st
from streamlit import config as _st_config

# Streamlit runs gc.collect(2) after every rerun. With torch and a large policy
# in memory that full collection takes 100ms+ and holds the GIL, delaying the
# board update on every click, so skip it; long-lived objects are frozen below.
_st_config.set_option("runner.postScriptGC", False)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from liars_poker.core import GameSpec, card_rank, generate_deck  # noqa: E402
from liars_poker.env import Env, rules_for_spec  # noqa: E402
from liars_poker.infoset import CALL  # noqa: E402
from liars_poker.serialization import load_policy  # noqa: E402

ARTIFACTS = os.path.join(ROOT, "artifacts")
DEFAULT_POLICY = os.path.join(
    ARTIFACTS, "benchmark_runs", "cfr_plus_runs", "r4_s4_h4_hp2pt_ss___20260108-192548", "policy"
)

KIND_NAMES = {
    "RankHigh": "High",
    "Pair": "Pair",
    "TwoPair": "Two Pair",
    "Trips": "Trips",
    "FullHouse": "Full House",
    "Quads": "Quads",
}
KIND_CODES = {"RankHigh": "H", "Pair": "P", "TwoPair": "2P", "Trips": "T", "FullHouse": "FH", "Quads": "Q"}
SINGLE_NEED = {"RankHigh": 1, "Pair": 2, "Trips": 3, "Quads": 4}


# --------------------------------------------------------------------------
# Static game description
# --------------------------------------------------------------------------

def _plural(rank: int) -> str:
    return f"{rank}s"


@lru_cache(maxsize=32)
def claim_catalog(spec: GameSpec) -> Tuple[Dict[str, Any], ...]:
    """One dict per claim index, in bidding order."""

    rules = rules_for_spec(spec)
    out = []
    for idx, (kind, value) in enumerate(rules.claims):
        if kind == "TwoPair":
            low, high = rules.two_pair_ranks[value]
            ranks = [high, low]
            need = {high: 2, low: 2}
            label = f"Two pair, {_plural(high)} & {_plural(low)}"
            short = f"{high}{high}{low}{low}"
        elif kind == "FullHouse":
            trip, pair = rules.full_house_ranks[value]
            ranks = [trip, pair]
            need = {trip: 3, pair: 2}
            label = f"Full house, {_plural(trip)} over {_plural(pair)}"
            short = f"{trip}{trip}{trip}{pair}{pair}"
        else:
            ranks = [value]
            n = SINGLE_NEED[kind]
            need = {value: n}
            label = {
                "RankHigh": f"One {value}",
                "Pair": f"Pair of {_plural(value)}",
                "Trips": f"Three {_plural(value)}",
                "Quads": f"Four {_plural(value)}",
            }[kind]
            short = str(value) * n
        out.append(
            {
                "idx": idx,
                "kind": kind,
                "kind_name": KIND_NAMES[kind],
                "code": KIND_CODES[kind],
                "ranks": ranks,
                "need": [[r, n] for r, n in need.items()],
                "label": label,
                "short": short,
                "cards": sum(need.values()),
            }
        )
    return tuple(out)


def _claim_true(need: List[List[int]], counts: Dict[int, int]) -> bool:
    return all(counts.get(r, 0) >= n for r, n in need)


def truth_odds(spec: GameSpec, hand: Tuple[int, ...]) -> List[float]:
    """P(claim true) for every claim, given `hand` and a uniformly random
    opponent hand from the remaining deck. A naive prior, not the bot's view."""

    ranks = list(range(1, spec.ranks + 1))
    mine = {r: 0 for r in ranks}
    for c in hand:
        mine[card_rank(c, spec)] += 1
    avail = {r: spec.suits - mine[r] for r in ranks}
    h = spec.hand_size
    remaining = sum(avail.values())
    total = math.comb(remaining, h)
    catalog = claim_catalog(spec)
    acc = [0.0] * len(catalog)

    def rec(i: int, left: int, weight: int, drawn: Dict[int, int]) -> None:
        if i == len(ranks):
            if left:
                return
            counts = {r: mine[r] + drawn.get(r, 0) for r in ranks}
            for j, cl in enumerate(catalog):
                if _claim_true(cl["need"], counts):
                    acc[j] += weight
            return
        r = ranks[i]
        for k in range(0, min(left, avail[r]) + 1):
            drawn[r] = k
            rec(i + 1, left - k, weight * math.comb(avail[r], k), drawn)
        drawn[r] = 0

    rec(0, h, 1, {})
    return [a / total for a in acc]


@st.cache_data(show_spinner=False)
def discover_policies() -> List[Dict[str, str]]:
    """Saved policies under artifacts/, skipping responders and snapshots."""

    skip = ("approx_br", "monitor_brs", "current", "snapshots", "checkpoint", "responder", "_smoke", "/test")
    found = []
    for dirpath, dirnames, filenames in os.walk(ARTIFACTS):
        rel = os.path.relpath(dirpath, ARTIFACTS).replace("\\", "/")
        if any(s in "/" + rel for s in skip):
            dirnames[:] = []
            continue
        if "metadata.json" in filenames:
            try:
                with open(os.path.join(dirpath, "metadata.json"), "r", encoding="utf-8") as fh:
                    spec = json.load(fh).get("spec", {})
                kinds = spec.get("claim_kinds", [])
                n_claims = len(claim_catalog(GameSpec(
                    ranks=spec["ranks"], suits=spec["suits"], hand_size=spec["hand_size"],
                    claim_kinds=tuple(kinds), suit_symmetry=spec.get("suit_symmetry", False),
                )))
                found.append({
                    "path": dirpath,
                    "rel": rel,
                    "game": f"{spec['ranks']} ranks · {spec['suits']} suits · {spec['hand_size']} cards · {n_claims} claims",
                    "claims": n_claims,
                })
            except Exception:
                pass
            dirnames[:] = []
    found.sort(key=lambda d: (d["claims"], d["rel"]))
    return found


@st.cache_resource(show_spinner="Loading policy…", max_entries=6)
def cached_policy(path: str):
    """Policies can take seconds to load; share them across sessions."""
    loaded = load_policy(path)
    # move torch modules and policy tables out of the collector's reach so
    # routine garbage collection stays cheap while playing
    gc.collect()
    gc.freeze()
    return loaded


# --------------------------------------------------------------------------
# Live game state
# --------------------------------------------------------------------------

class Arena:
    def __init__(self) -> None:
        self.policy = None
        self.spec: Optional[GameSpec] = None
        self.env: Optional[Env] = None
        self.policy_path = ""
        self.error = ""
        self.human_seat = 0  # 0: you open; 1: bot opens
        self.pending_seat: Optional[int] = None
        self.score = [0, 0]  # you, bot
        self.recorded = False
        self.reveal = False
        self.hints = True
        self.rng = random.Random(time.time())
        self.odds: List[float] = []
        self.bot_moves: List[Dict[str, Any]] = []
        self.last_nonce = None
        self.hand_no = 0

    # ---- lifecycle -------------------------------------------------------
    def load(self, path: str) -> None:
        path = path.strip().strip('"')
        try:
            policy, spec = cached_policy(os.path.normpath(path))
        except Exception as exc:  # surfaced in the UI
            self.error = f"Could not load policy: {exc}"
            return
        self.policy, self.spec, self.policy_path = policy, spec, path
        self.env = Env(spec)
        self.error = ""
        self.score = [0, 0]
        self.hand_no = 0
        self.new_game()

    def new_game(self) -> None:
        if self.env is None:
            return
        if self.pending_seat is not None:
            self.human_seat, self.pending_seat = self.pending_seat, None
        self.env.reset(seed=self.rng.randrange(2**31))
        self.policy.begin_episode(self.rng)
        self.recorded = False
        self.reveal = False
        self.bot_moves = []
        self.hand_no += 1
        self.odds = truth_odds(self.spec, self.my_hand)

    # ---- helpers ---------------------------------------------------------
    @property
    def my_label(self) -> str:
        return "P1" if self.human_seat == 0 else "P2"

    @property
    def bot_label(self) -> str:
        return "P2" if self.human_seat == 0 else "P1"

    @property
    def my_hand(self) -> Tuple[int, ...]:
        return self.env._p1_hand if self.human_seat == 0 else self.env._p2_hand

    @property
    def bot_hand(self) -> Tuple[int, ...]:
        return self.env._p2_hand if self.human_seat == 0 else self.env._p1_hand

    @property
    def over(self) -> bool:
        return self.env is not None and self.env._done

    @property
    def bot_to_move(self) -> bool:
        return self.env is not None and not self.over and self.env.current_player() == self.bot_label

    def _record(self) -> None:
        if self.over and not self.recorded:
            won = self.env._winner == self.human_seat
            self.score[0 if won else 1] += 1
            self.recorded = True

    # ---- actions ---------------------------------------------------------
    def apply(self, action: Dict[str, Any]) -> None:
        if not action:
            return
        nonce = action.get("nonce")
        if nonce is not None and nonce == self.last_nonce:
            return
        self.last_nonce = nonce
        kind = action.get("type")
        if kind == "load":
            self.load(str(action.get("path", "")))
            return
        if self.env is None:
            return
        if kind in ("claim", "call"):
            if self.over or self.bot_to_move:
                return
            idx = CALL if kind == "call" else int(action["idx"])
            if idx in self.env.legal_actions():
                self.env.step(idx)
                self._record()
        elif kind == "new":
            self.new_game()
        elif kind == "reset_score":
            self.score = [0, 0]
        elif kind == "reveal":
            self.reveal = not self.reveal
        elif kind == "hints":
            self.hints = not self.hints
        elif kind == "seat":
            seat = 0 if action.get("value") == "first" else 1
            if not self.env._history or self.over:
                self.human_seat = seat
                self.pending_seat = None
                self.new_game()
            else:
                self.pending_seat = seat

    def bot_step(self) -> None:
        if not self.bot_to_move:
            return
        iset = self.env.infoset_key(self.bot_label)
        try:
            dist = self.policy.prob_dist_at_infoset(iset)
        except Exception:
            dist = {}
        act = self.policy.sample(iset, self.rng)
        catalog = claim_catalog(self.spec)
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
        self.bot_moves.append({
            "ply": len(self.env._history),
            "idx": act,
            "top": [
                {"idx": a, "label": "Call" if a == CALL else catalog[a]["label"], "p": float(p)}
                for a, p in top if p > 0.0005
            ],
        })
        self.env.step(act)
        self._record()

    # ---- payload ---------------------------------------------------------
    def _cards(self, hand: Tuple[int, ...]) -> List[Dict[str, Any]]:
        out = []
        sym = self.spec.suit_symmetry or self.spec.suits <= 1
        for c in hand:
            out.append({"rank": card_rank(c, self.spec), "suit": None if sym else c % self.spec.suits})
        out.sort(key=lambda d: (-d["rank"], d["suit"] or 0))
        return out

    def payload(self) -> Dict[str, Any]:
        base = {
            "loaded": self.env is not None,
            "error": self.error,
            "presets": discover_policies(),
            "policy_path": self.policy_path,
            "default_path": DEFAULT_POLICY,
        }
        if self.env is None:
            return base
        spec, env = self.spec, self.env
        catalog = claim_catalog(spec)
        hist = list(env._history)
        seat_of = lambda ply: ply % 2  # noqa: E731
        history = []
        for ply, a in enumerate(hist):
            who = "you" if seat_of(ply) == self.human_seat else "bot"
            if a == CALL:
                history.append({"ply": ply, "who": who, "idx": -1, "label": "Call!", "short": "CALL", "kind": "Call"})
            else:
                cl = catalog[a]
                history.append({"ply": ply, "who": who, "idx": a, "label": cl["label"], "short": cl["short"], "kind": cl["kind"]})
        last_claim = max([a for a in hist if a != CALL], default=None)
        legal = [a for a in env.legal_actions() if a != CALL] if not self.bot_to_move else []
        can_call = (not self.over) and (not self.bot_to_move) and CALL in env.legal_actions()

        mine = {}
        for c in self.my_hand:
            r = card_rank(c, spec)
            mine[r] = mine.get(r, 0) + 1

        result = None
        if self.over:
            both = {}
            for c in self.my_hand + self.bot_hand:
                r = card_rank(c, spec)
                both[r] = both.get(r, 0) + 1
            cl = catalog[last_claim]
            caller_ply = len(hist) - 1
            result = {
                "winner": "you" if env._winner == self.human_seat else "bot",
                "caller": "you" if seat_of(caller_ply) == self.human_seat else "bot",
                "claimer": "you" if seat_of(caller_ply - 1) == self.human_seat else "bot",
                "claim": cl["label"],
                "claim_idx": last_claim,
                "was_true": _claim_true(cl["need"], both),
                "need": cl["need"],
                "actual": [[r, both.get(r, 0)] for r, _ in cl["need"]],
                "counts": [[r, both.get(r, 0)] for r in range(1, spec.ranks + 1)],
            }

        if self.over:
            turn = "over"
        elif self.bot_to_move:
            turn = "bot"
        else:
            turn = "you"

        kinds = []
        for k in spec.claim_kinds:
            if any(c["kind"] == k for c in catalog):
                kinds.append({"kind": k, "name": KIND_NAMES[k], "code": KIND_CODES[k]})

        return {
            **base,
            "spec": {
                "ranks": spec.ranks,
                "suits": spec.suits,
                "hand_size": spec.hand_size,
                "claim_kinds": list(spec.claim_kinds),
                "kinds": kinds,
                "short": spec.to_short_str(),
                "n_claims": len(catalog),
                "deck": spec.ranks * spec.suits,
            },
            "policy_kind": type(self.policy).__name__,
            "claims": list(catalog),
            "odds": [round(x, 4) for x in self.odds],
            "hints": self.hints,
            "hand": self._cards(self.my_hand),
            "mine": [[r, mine.get(r, 0)] for r in range(1, spec.ranks + 1)],
            "bot_hand": self._cards(self.bot_hand) if (self.reveal or self.over) else None,
            "bot_cards": spec.hand_size,
            "history": history,
            "last_claim": last_claim,
            "legal": legal,
            "can_call": can_call,
            "turn": turn,
            "you_open": self.human_seat == 0,
            "pending_seat": None if self.pending_seat is None else ("first" if self.pending_seat == 0 else "second"),
            "score": {"you": self.score[0], "bot": self.score[1]},
            "reveal": self.reveal,
            "result": result,
            "bot_moves": self.bot_moves if self.over else [],
            "hand_no": self.hand_no,
        }


# --------------------------------------------------------------------------
# Shared browser helpers (prepended to every vision's JS module)
# --------------------------------------------------------------------------

COMMON_JS = r"""
const LP = {
  send(c, obj) {
    // mark the board as waiting until the next payload redraws it
    Array.from(c.parentElement.children).find(e => e.classList && e.classList.contains('lp-root'))?.classList.add('lp-pending');
    c.setTriggerValue('action', Object.assign({}, obj, {nonce: Date.now() + Math.random()}));
  },
  font(href) {
    if (!document.querySelector(`link[href="${href}"]`)) {
      const l = document.createElement('link'); l.rel = 'stylesheet'; l.href = href; document.head.appendChild(l);
    }
  },
  esc(s) { return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); },
  root(c, cls) {
    let r = Array.from(c.parentElement.children).find(e => e.classList && e.classList.contains('lp-root'));
    if (!r) { r = document.createElement('div'); r.className = 'lp-root ' + (cls || ''); c.parentElement.appendChild(r); }
    r.classList.remove('lp-pending');
    return r;
  },
  ui(c, defaults) { const p = c.parentElement; if (!p.__ui) p.__ui = Object.assign({}, defaults); return p.__ui; },
  keys(c, handler) {
    // Capture on window so our keys win over Streamlit's own shortcuts
    // ("c" clears caches, "r" reruns). A handler returns true when it used the key.
    const p = c.parentElement; p.__keyh = handler;
    if (!p.__keybound) {
      p.__keybound = true;
      window.addEventListener('keydown', e => {
        const t = e.composedPath ? e.composedPath()[0] : e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (p.isConnected && p.__keyh && p.__keyh(e)) { e.preventDefault(); e.stopImmediatePropagation(); }
      }, true);
    }
  },
  pct(x) { return Math.round(100 * x) + '%'; },
  claimOf(d, idx) { return idx == null || idx < 0 ? null : d.claims[idx]; },
  groups(d) {
    const g = {};
    for (const p of d.presets) { const top = p.rel.split('/')[0]; (g[top] = g[top] || []).push(p); }
    return g;
  },
  loaderHTML(d) {
    let h = `<div class="lp-modal" data-close="1"><div class="lp-modal-card">
      <div class="lp-modal-head"><div class="lp-modal-title">Choose an opponent</div><button class="lp-x" data-close="1">×</button></div>
      <div class="lp-pathrow"><input class="lp-path" placeholder="Paste a policy directory…" value="${LP.esc(d.policy_path || '')}"><button class="lp-go">Load</button></div>
      ${d.error ? `<div class="lp-err">${LP.esc(d.error)}</div>` : ''}
      <div class="lp-presets">`;
    const g = LP.groups(d);
    for (const top of Object.keys(g)) {
      h += `<div class="lp-group">${LP.esc(top)}</div>`;
      for (const p of g[top]) {
        const cur = p.path === d.policy_path ? ' current' : '';
        h += `<button class="lp-preset${cur}" data-path="${LP.esc(p.path)}"><span class="lp-preset-game">${LP.esc(p.game)}</span><span class="lp-preset-rel">${LP.esc(p.rel.split('/').slice(1).join(' / ') || p.rel)}</span></button>`;
      }
    }
    return h + `</div></div></div>`;
  },
  bindLoader(root, c, ui, redraw) {
    root.querySelectorAll('[data-close]').forEach(el => el.addEventListener('click', e => {
      if (e.target === el) { ui.loader = false; redraw(); }
    }));
    root.querySelectorAll('.lp-preset').forEach(el => el.addEventListener('click', () => {
      ui.loader = false; ui.loading = el.dataset.path; LP.send(c, {type: 'load', path: el.dataset.path}); redraw();
    }));
    const inp = root.querySelector('.lp-path');
    const go = () => { ui.loader = false; ui.loading = inp.value; LP.send(c, {type: 'load', path: inp.value}); redraw(); };
    root.querySelector('.lp-go')?.addEventListener('click', go);
    inp?.addEventListener('keydown', e => { if (e.key === 'Enter') go(); if (e.key === 'Escape') { ui.loader = false; redraw(); } });
  },
  reviewHTML(d) {
    if (!d.bot_moves || !d.bot_moves.length) return '';
    let h = '<div class="lp-review">';
    for (const m of d.bot_moves) {
      const made = m.idx < 0 ? 'Call' : d.claims[m.idx].label;
      h += `<div class="lp-review-move"><div class="lp-review-head">Bot's move ${Math.floor(m.ply / 2) + 1}: <b>${LP.esc(made)}</b></div>`;
      for (const t of m.top) {
        h += `<div class="lp-review-row${t.idx === m.idx ? ' chosen' : ''}"><span>${LP.esc(t.label)}</span><i style="--p:${t.p}"></i><em>${LP.pct(t.p)}</em></div>`;
      }
      h += '</div>';
    }
    return h + '</div>';
  },

  // ---------- claim calculator: three switchable button layouts ----------
  // grid  : rank claims as kind x rank, two pair / full house as matrices
  // table : one table, a column per rank; every claim sits under its main rank,
  //         rows follow bidding order, so legal claims read like text
  // rows  : one row of chips per claim kind, in bidding order
  CALC_LAYOUTS: [['grid', 'Grid'], ['table', 'Table'], ['rows', 'Rows']],
  calcLayout(key, fallback) {
    try { const v = localStorage.getItem('lp-calc-' + key); return LP.CALC_LAYOUTS.some(x => x[0] === v) ? v : fallback; }
    catch (e) { return fallback; }
  },
  setCalcLayout(key, v) { try { localStorage.setItem('lp-calc-' + key, v); } catch (e) {} },
  nextCalcLayout(v) { const L = LP.CALC_LAYOUTS.map(x => x[0]); return L[(L.indexOf(v) + 1) % L.length]; },
  calcHint(d) {
    if (d.turn === 'bot') return 'The bot is thinking…';
    if (d.turn === 'over') return 'Hand over';
    const last = LP.claimOf(d, d.last_claim);
    return last ? `Beat <b>${LP.esc(last.label)}</b> or call. Hover a button for its name.` : 'Open with any claim. Hover a button for its name.';
  },
  calcHTML(d, layout) {
    const S = d.spec, R = S.ranks, legal = new Set(d.legal);
    const bidBy = {};
    for (const h of d.history) if (h.idx >= 0) bidBy[h.idx] = h.who;
    const mine = Object.fromEntries(d.mine);
    const isCombo = k => k === 'TwoPair' || k === 'FullHouse';
    const rep = (r, n) => String(r).repeat(n);
    const comboText = cl => cl.need.map(([r, n]) => rep(r, n)).join('·');
    const by = {};
    for (const cl of d.claims) (by[cl.kind] = by[cl.kind] || []).push(cl);
    const kinds = S.kinds.filter(k => by[k.kind]);
    const btn = (cl, text, extra = '') => {
      if (!cl) return '<span class="cc-void"></span>';
      const ok = legal.has(cl.idx), who = bidBy[cl.idx], cur = cl.idx === d.last_claim;
      const cls = ['cc', extra, ok ? 'ok' : 'dead', cur ? 'cur' : '', who ? 'by-' + who : ''].join(' ');
      const o = d.odds[cl.idx];
      const odds = d.hints && ok ? `<span class="cc-o">${LP.pct(o)}</span><span class="cc-b" style="--o:${o}"></span>` : '';
      const mark = who ? `<span class="cc-m">${who === 'you' ? 'Y' : 'B'}</span>` : '';
      return `<button class="${cls}" data-idx="${cl.idx}" aria-disabled="${!ok}" ${ok ? '' : 'tabindex="-1"'}>${mark}<span class="cc-t">${text}</span>${odds}</button>`;
    };
    const hdr = r => `<div class="cc-hdr">${r}<i>${'●'.repeat(mine[r] || 0)}</i></div>`;
    let h = '';
    if (layout === 'rows') {
      h += '<div class="calc-rows">';
      for (const k of kinds) {
        h += `<div class="cr-lab">${k.name}</div><div class="cr-chips">`;
        for (const cl of by[k.kind]) h += btn(cl, isCombo(k.kind) ? comboText(cl) : cl.short, isCombo(k.kind) ? 'wide' : '');
        h += '</div>';
      }
      h += '</div>';
    } else if (layout === 'table') {
      h += `<div class="calc-table" style="grid-template-columns:auto repeat(${R}, minmax(0,1fr))"><div></div>`;
      for (let r = 1; r <= R; r++) h += hdr(r);
      for (const k of kinds) {
        const sub = k.kind === 'TwoPair' ? '<small>column pair + button pair</small>' : k.kind === 'FullHouse' ? '<small>column trips + button pair</small>' : '';
        h += `<div class="ct-lab">${k.name}${sub}</div>`;
        for (let r = 1; r <= R; r++) {
          const cells = by[k.kind].filter(cl => cl.ranks[0] === r);
          if (!cells.length) { h += '<div class="ct-cell"></div>'; continue; }
          if (isCombo(k.kind)) {
            const lead = rep(r, k.kind === 'TwoPair' ? 2 : 3);
            h += `<div class="ct-cell multi"><span class="ct-lead">${lead}+</span>${cells.map(cl => btn(cl, rep(cl.ranks[1], 2), 'chip')).join('')}</div>`;
          } else {
            h += `<div class="ct-cell">${btn(cells[0], cells[0].short)}</div>`;
          }
        }
      }
      h += '</div>';
    } else {
      h += '<div class="calc-grid">';
      const singles = kinds.filter(k => !isCombo(k.kind));
      if (singles.length) {
        h += `<div class="cg-panel"><div class="cg-title">Rank claims</div><div class="cg" style="grid-template-columns:auto repeat(${R}, var(--cc-wk))"><div></div>`;
        for (let r = 1; r <= R; r++) h += hdr(r);
        for (const k of singles) {
          h += `<div class="cg-lab">${k.name}</div>`;
          for (let r = 1; r <= R; r++) { const cl = by[k.kind].find(x => x.ranks[0] === r); h += btn(cl, cl ? cl.short : ''); }
        }
        h += '</div></div>';
      }
      if (by.TwoPair) {
        const n = R - 1;
        h += `<div class="cg-panel"><div class="cg-title">Two pair <small>high ↓ · low →</small></div><div class="cg" style="grid-template-columns:auto repeat(${n}, var(--cc-wk))"><div></div>`;
        for (let lo = 1; lo <= n; lo++) h += `<div class="cc-hdr">${lo}</div>`;
        for (let hi = 2; hi <= R; hi++) {
          h += `<div class="cg-lab">${hi}</div>`;
          for (let lo = 1; lo <= n; lo++) {
            const cl = lo < hi ? by.TwoPair.find(x => x.ranks[0] === hi && x.ranks[1] === lo) : null;
            h += btn(cl, cl ? comboText(cl) : '');
          }
        }
        h += '</div></div>';
      }
      if (by.FullHouse) {
        h += `<div class="cg-panel"><div class="cg-title">Full house <small>trips ↓ · pair →</small></div><div class="cg" style="grid-template-columns:auto repeat(${R}, var(--cc-wk))"><div></div>`;
        for (let p = 1; p <= R; p++) h += `<div class="cc-hdr">${p}</div>`;
        for (let t = 1; t <= R; t++) {
          h += `<div class="cg-lab">${t}</div>`;
          for (let p = 1; p <= R; p++) {
            const cl = p !== t ? by.FullHouse.find(x => x.ranks[0] === t && x.ranks[1] === p) : null;
            h += btn(cl, cl ? comboText(cl) : '');
          }
        }
        h += '</div></div>';
      }
      h += '</div>';
    }
    return h;
  },
  calcFrame(d, layout) {
    const sw = LP.CALC_LAYOUTS.map(([v, n]) => `<button class="cs ${v === layout ? 'on' : ''}" data-layout="${v}">${n}</button>`).join('');
    return `<div class="calc-frame calc-${layout}-frame"><div class="calc-head"><div class="calc-read">${LP.calcHint(d)}</div>
      <div class="calc-sw" title="Button layout (G to cycle)">${sw}</div></div><div class="calc-body">${LP.calcHTML(d, layout)}</div></div>`;
  },
  bindCalc(root, d, onBid, onLayout) {
    const read = root.querySelector('.calc-read');
    const base = read ? read.innerHTML : '';
    const bidBy = {};
    for (const h of d.history) if (h.idx >= 0) bidBy[h.idx] = h.who;
    root.querySelectorAll('.cc[data-idx]').forEach(b => {
      const idx = +b.dataset.idx, cl = d.claims[idx];
      b.addEventListener('click', () => { if (b.classList.contains('ok')) onBid(idx); });
      b.addEventListener('mouseenter', () => {
        if (!read) return;
        const who = bidBy[idx];
        const tail = who ? ` · bid by ${who === 'you' ? 'you' : 'the bot'}`
          : b.classList.contains('ok') ? (d.hints ? ` · true ${LP.pct(d.odds[idx])} of the time if the bot's cards were random` : '')
          : ' · too low to bid now';
        read.innerHTML = `<b>${LP.esc(cl.label)}</b>${tail}`;
      });
      b.addEventListener('mouseleave', () => { if (read) read.innerHTML = base; });
    });
    root.querySelectorAll('.calc-sw [data-layout]').forEach(b => b.addEventListener('click', () => onLayout(b.dataset.layout)));
  },
  // Largest button scale (up to maxK) at which the calculator fits its box.
  fitCalc(root, maxK = 1.5) {
    const body = root.querySelector('.calc-body');
    if (!body) return;
    const frame = body.parentElement;
    const fits = k => { frame.style.setProperty('--k', k); return body.scrollHeight <= body.clientHeight + 1 && body.scrollWidth <= body.clientWidth + 1; };
    if (fits(maxK)) return;
    let lo = 0.7, hi = maxK;
    for (let i = 0; i < 7; i++) { const mid = (lo + hi) / 2; if (fits(mid)) lo = mid; else hi = mid; }
    frame.style.setProperty('--k', lo);
  },
  autoFit(c, root, maxK) {
    // tall screens have spare height, so let buttons grow a little more there
    const cap = () => (window.innerHeight > window.innerWidth ? maxK * 1.45 : maxK);
    LP.fitCalc(root, cap());
    const p = c.parentElement;
    p.__fit = () => LP.fitCalc(root, cap());
    if (!p.__fitBound) { p.__fitBound = true; window.addEventListener('resize', () => p.__fit && p.__fit()); }
  },
  // draw your own bid before the server answers (the reply replaces it)
  optimistic(d, idx) {
    const cl = d.claims[idx];
    return Object.assign({}, d, {
      history: d.history.concat([{ply: d.history.length, who: 'you', idx, label: cl.label, short: cl.short, kind: cl.kind}]),
      last_claim: idx, legal: [], can_call: false, turn: 'bot',
    });
  },
};
"""

# Structure of the shared calculator; each vision sets the --cc-* colours/sizes.
COMMON_CSS = r"""
.calc-frame { --k:1; --cc-hk:calc(var(--cc-h) * var(--k)); --cc-wk:calc(var(--cc-w) * var(--k)); --cc-fsk:calc(var(--cc-fs) * var(--k)); display:flex; flex-direction:column; min-height:0; gap:10px; }
.calc-head { display:flex; align-items:center; gap:12px; min-height:30px; }
.calc-read { flex:1; min-width:0; font-size:var(--calc-read-fs, 15px); color:var(--cc-hdr); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.calc-read b { color:var(--cc-read-strong, inherit); }
.calc-sw { display:flex; border:1px solid var(--cc-sw-border); border-radius:8px; overflow:hidden; flex-shrink:0; }
.calc-sw .cs { background:transparent; border:0; color:var(--cc-hdr); font-family:inherit; font-size:12px; font-weight:700; padding:5px 11px; cursor:pointer; letter-spacing:.04em; }
.calc-sw .cs.on { background:var(--cc-sw-on-bg); color:var(--cc-sw-on-fg); }
.calc-body { flex:1; min-height:0; overflow:auto; padding:2px 2px 6px; }
.cc { position:relative; display:inline-flex; align-items:center; justify-content:center; height:var(--cc-hk); min-width:var(--cc-wk); padding:0 calc(8px * var(--k)); box-sizing:border-box;
  border:1px solid var(--cc-border); border-radius:var(--cc-radius); background:var(--cc-bg); color:var(--cc-fg); font-family:var(--cc-font); font-weight:var(--cc-weight, 700);
  font-size:var(--cc-fsk); letter-spacing:.03em; cursor:pointer; overflow:hidden; white-space:nowrap; transition:transform .08s, background .08s, box-shadow .08s; }
.cc.ok { box-shadow:var(--cc-shadow, none); }
.cc.ok:hover { background:var(--cc-hover-bg); transform:translateY(-1px); box-shadow:var(--cc-hover-shadow, 0 4px 10px rgba(0,0,0,.25)); }
.cc.dead { background:var(--cc-dead-bg); color:var(--cc-dead-fg); border-color:transparent; cursor:default; }
.cc.cur { background:var(--cc-cur-bg); color:var(--cc-cur-fg); box-shadow:inset 0 0 0 2px var(--cc-cur-ring); }
.cc .cc-o { position:absolute; top:1px; right:3px; font-family:var(--cc-odds-font, system-ui, sans-serif); font-size:9.5px; font-weight:700; color:var(--cc-odds-fg); letter-spacing:0; }
.cc .cc-b { position:absolute; left:0; bottom:0; height:3px; width:calc(var(--o) * 100%); background:var(--cc-bar); }
.cc .cc-m { position:absolute; top:2px; left:2px; width:13px; height:13px; border-radius:50%; font-family:system-ui, sans-serif; font-size:8.5px; font-weight:800; display:flex; align-items:center; justify-content:center; }
.cc.by-you .cc-m { background:var(--cc-you); color:#111; } .cc.by-bot .cc-m { background:var(--cc-bot); color:#fff; }
.cc-void { display:block; }
.cc-hdr { text-align:center; font-family:var(--cc-font); font-weight:700; font-size:calc(var(--cc-fsk) * .9); color:var(--cc-hdr); line-height:1; }
.cc-hdr i { display:block; font-style:normal; font-size:8px; letter-spacing:1px; color:var(--cc-you); height:10px; margin-top:2px; }
.cg-lab, .ct-lab, .cr-lab { font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--cc-hdr); padding-right:10px; white-space:nowrap; text-align:right; }
.ct-lab small { display:block; font-size:10px; font-weight:500; letter-spacing:.01em; text-transform:none; opacity:.7; margin-top:2px; }
/* grid */
.calc-grid { display:flex; flex-wrap:wrap; gap:calc(16px * var(--k)) calc(28px * var(--k)); align-items:flex-start; }
.cg-title { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--cc-hdr); font-weight:700; margin-bottom:6px; }
.cg-title small { letter-spacing:.02em; text-transform:none; font-weight:500; opacity:.75; }
.cg { display:grid; gap:calc(4px * var(--k)); align-items:center; }
/* table */
.calc-table { display:grid; gap:5px 5px; align-items:center; max-width:var(--ct-max, 1200px); }
.ct-cell { display:flex; gap:3px; flex-wrap:wrap; align-items:center; min-width:0; container-type:inline-size; }
.ct-cell > .cc { flex:1 1 auto; min-width:0; }
.ct-cell.multi > .cc { flex:1 1 24px; min-width:24px; padding:0 2px; height:calc(var(--cc-hk) * .82); font-size:calc(var(--cc-fsk) * .78); }
.ct-cell.multi .cc-o { display:none; }
.ct-lead { display:none; font-family:var(--cc-font); font-weight:700; font-size:calc(var(--cc-fsk) * .7); color:var(--cc-hdr); padding-right:2px; }
@container (min-width: 190px) { .ct-lead { display:inline; } }
/* rows */
.calc-rows { display:grid; grid-template-columns:auto 1fr; gap:calc(6px * var(--k)) 4px; align-items:center; }
.cr-chips { display:flex; flex-wrap:wrap; gap:calc(4px * var(--k)); }
.cr-chips .cc.wide { padding:0 9px; }
"""


def make_component(name: str, *, css: str, js: str, html: Optional[str] = None):
    return st.components.v2.component(name, html=html, css=COMMON_CSS + css, js=COMMON_JS + js)


# --------------------------------------------------------------------------
# App runner shared by every vision
# --------------------------------------------------------------------------

PAGE_CSS = """
<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], footer {display:none !important;}
.block-container, [data-testid="stMainBlockContainer"] {padding:0 !important; max-width:100% !important;}
[data-testid="stAppViewContainer"] > .main, section.main {padding:0 !important;}
[data-testid="stVerticalBlock"] {gap:0 !important;}
html, body, [data-testid="stApp"] {background:__BG__ !important;}
/* no grey-out while Streamlit reruns: the board is redrawn in place */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"] {opacity:1 !important; transition:none !important;}
</style>
"""


def get_arena() -> Arena:
    if "arena" not in st.session_state:
        arena = Arena()
        if os.path.exists(os.path.join(DEFAULT_POLICY, "metadata.json")):
            arena.load(DEFAULT_POLICY)
        st.session_state.arena = arena
    return st.session_state.arena


def run(component: Callable[..., Any], *, key: str, background: str = "#111") -> None:
    """Mount `component` as a full-page board and run the game loop.

    The bot replies inside the same script run as your action, so each click
    costs one round trip; any "thinking" pause is a browser-side animation.
    """

    st.html(PAGE_CSS.replace("__BG__", background))
    arena = get_arena()

    def on_action() -> None:
        state = st.session_state.get(key) or {}
        action = state.get("action") if isinstance(state, dict) else getattr(state, "action", None)
        arena.apply(action)

    # on_action runs before this script body, so the bot sees your move here
    while arena.bot_to_move:
        arena.bot_step()
    component(key=key, data=arena.payload(), on_action_change=on_action, height="content")
