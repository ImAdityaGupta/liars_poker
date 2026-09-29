# From exact CFR+ to sampled regret updates

This bridge keeps **tabular regrets and an exact tabular average policy** on the
18-claim game. It changes only how each iteration's regret increment is
estimated. The point is to identify what sampling changes before introducing
neural-network fitting.

## The quantities being estimated

For information set `I), action `a), and iteration `t`:

- `q(I)`: chance and **opponent** reach before `I`. It excludes the
  updating player's earlier action probabilities.
- `g(I,a)`: the action advantage conditional on reaching `I`. It compares
  forcing `a` with the current action mixture. The continuation value in
  `g` already includes later chance and opponent actions.
- Exact CFR+ adds `q(I) g(I,a)` to the cumulative regret, then clips once:
  `R_t(I,a) = max(0, R_(t-1)(I,a) + q(I) g(I,a))`.

If `q(I)=0`, `g(I,a)` is undefined and the exact increment is zero.
Quantities called *exact* below are recomputed under **each arm's own current
policy**; the arms diverge as they train.

For a sampled update, run `K` independent root deals, sample opponent
actions, and expand every action of the updating player. Let `N(I)` be the
number of roots visiting `I` and `G_k(I,a)` the sampled advantage on a
visit. Each root can visit a given information set at most once. Then

```text
q̂(I) = N(I) / K
ĝ(I,a) = mean of G_k(I,a) over the N(I) visits, defined only if N(I)>0
q̂(I) ĝ(I,a) = (1/K) sum over all K roots of [visit_k(I) × G_k(I,a)].
```

The last expression treats a missed root as a zero contribution. For
example, ten visits among 100 roots with advantage `+1` give `q̂ĝ=+0.1`,
while `ĝ=+1`. This is the difference between an estimated
*counterfactual increment* and an estimated *conditional advantage*.

## The 16-cell design

There are two numbering systems below. **Cell 1–16** is the row number in
the design table. **Arm 0, 1a, 1b, 1c, 2, 3, and 4** are names for the
implemented update rules. The runner uses string names; this map connects
them to the cells.

Choose one option from each column:

| Choice | Options |
| --- | --- |
| Reach multiplier | exact `q`; sampled `q̂=N/K`; `1` if **visited**; `1` if **possible** (`q>0`) |
| Advantage | exact `g`; sampled `ĝ` |
| When `N=0` | **Apply** the chosen update if its value can be computed; or force a **zero increment** |

This makes `4 × 2 × 2 = 16` nominal cells. “Apply” matters only when
the reach multiplier is nonzero on a miss **and** the advantage is available.
When a sampled advantage is required but no root visited `I`, it is
undefined; `q>0` does not supply its value.

| Cell | Reach | Advantage | On no visit | Increment, or reason it collapses |
| ---: | --- | --- | --- | --- |
| 1 | exact `q` | exact `g` | apply | `qg` — **0, exact CFR+** |
| 2 | exact `q` | exact `g` | zero | `1[N>0] qg` — **exact-reach gated** |
| 3 | exact `q` | sampled `ĝ` | apply | **Undefined:** `q` can be positive when `ĝ` is unavailable |
| 4 | exact `q` | sampled `ĝ` | zero | `qĝ` if visited — **2, sample value** |
| 5 | sampled `q̂` | exact `g` | apply | `q̂g` — **1a, sample reach** |
| 6 | sampled `q̂` | exact `g` | zero | Same as **5**: `q̂=0` on a miss |
| 7 | sampled `q̂` | sampled `ĝ` | apply | Same as **8**: `q̂=0` on a miss; no `ĝ` is needed |
| 8 | sampled `q̂` | sampled `ĝ` | zero | `q̂ĝ` if visited — **3, sample both** |
| 9 | `1` if visited | exact `g` | apply | `1[N>0]g` — **1c, unit-reach gated** |
| 10 | `1` if visited | exact `g` | zero | Same as **9** |
| 11 | `1` if visited | sampled `ĝ` | apply | Same as **12**: multiplier is zero on a miss |
| 12 | `1` if visited | sampled `ĝ` | zero | `ĝ` if visited — **4, conditional sampled** |
| 13 | `1` if possible | exact `g` | apply | `g` wherever `q>0` — **1b, ignore reach** |
| 14 | `1` if possible | exact `g` | zero | Same as **9** |
| 15 | `1` if possible | sampled `ĝ` | apply | **Undefined:** `q>0` can hold when `ĝ` is unavailable |
| 16 | `1` if possible | sampled `ĝ` | zero | Same as **12**, hence also arm **4** |

| Implemented arm | Runner name | Cell(s) |
| --- | --- | --- |
| 0, exact CFR+ | `exact` | 1 |
| Exact-reach gated | `exact_reach_gated` | 2 |
| 1a, sample reach | `sample_reach` | 5, 6 |
| 2, sample value | `sample_value` | 4 |
| 3, sample both | `sample_both` | 7, 8 |
| 1c, unit-reach gated | `unit_reach_gated` | 9, 10, 14 |
| 4, conditional sampled | `conditional` | 11, 12, 16 |
| 1b, ignore reach | `ignore_reach` | 13 |

Thus **cell 12 and cell 16** are the same as implemented arm 4. There are
eight distinct executable updates. Cells 3 and 15
would need an additional rule for inventing a value on a miss; that would
be a new experiment, not another setting of this grid.

### Why include the two exact-value gated controls?

Both controls use exact `g`, so their differences cannot come from noisy
sampled advantages:

- **Exact-reach gated** (cell 2) applies `qg` when sampled roots visit `I`
  and zero otherwise. Comparing it with exact CFR+ (cell 1) asks how much
  is lost by missing an update at an infoset, while keeping the correct
  reach scale on every update that is made.
- **Unit-reach gated / arm 1c** (cell 9) applies `g` when visited and zero
  otherwise. Comparing it with exact-reach gated (cell 2) isolates the
  effect of the `q` multiplier, with exact values and the same visit gate.
  Comparing it with ignore-reach / arm 1b (cell 13) isolates the visit gate:
  both use unit reach and exact values, but arm 1b updates every infoset
  with positive reach, including those missed by the sampled roots.

The ordinary neural CFR+ update is closest to **cell 12/16, implemented
arm 4 (`conditional`)**: it trains on visited rows using sampled conditional
advantages and has no explicit `q` multiplier. Its neural target also
contains the previous regret-network prediction and the `1/t` scaling, so
arm 4 matches the sampling/reach choice, not the full neural algorithm.
The optional neural `N/K` experiment instead uses the reach-weighted form
closest to cell 8, arm 3 (`sample_both`).

All executable arms use the same CFR+ regret-matching and exact tabular
average-policy construction. Each computes **one increment per information
set per outer iteration, then clips once**. Sampling noise can still change
the result after clipping, even when a pre-clip increment is unbiased.

## Units and what the bridge tests

The sampled roots include the chance of drawing the updating player's own
hand. The dense solver stores unnormalised opponent-hand counts, so its raw
increment must be converted before comparing it with `N/K`. For this
game there are `C(16,2)=120` own-card combinations and, conditional on
one own hand, `C(14,2)=91` opponent-card combinations. If encoded own
hand `i` represents `c_i` physical hands, set `p_i=c_i/120`.
With dense blocker matrix `A`:

```text
q_exact(I,i) = p_i × (A @ opponent_reach)[I,i] / 91
r_exact(I,i,a) = (p_i / 91) × dense_increment(I,i,a)
g_exact(I,i,a) = dense_increment(I,i,a) / (A @ opponent_reach)[I,i].
```

Here `c_i=6` for a rank pair and `16` for two distinct ranks. The
`p_i/91` factor is fixed for a given tabular hand, so it does not alter
that hand's exact regret-matching policy. It **does** matter when comparing
the numerical sampled and exact increments.

The key comparisons are **0 versus 1a** (reach sampling with exact
advantage), **0 versus 1b** (remove reach without sampling), **0 versus
exact-reach gated** (missed updates alone), **1a versus 1c** (visit
frequency with exact advantage), and **3 versus 4** (visit frequency with
the same sampled advantages). Arm 2 adds sampled advantages while keeping
exact reach, but zeroes unvisited updates; its errors cannot be assigned
to value sampling alone.

The production neural trainer adds further changes: it predicts
`R_t/t` with a network, fits targets only at visited information sets,
and learns the average policy with a second network. Its ordinary sampled
target resembles the visited conditional increment in arm 4. The optional
`N/K` neural target resembles arm 3 algebraically, but its target scale
and regression weighting also change. Tabular agreement therefore does
not guarantee neural-network agreement.

See the [tabular bridge runner](../../scripts/run_cfr_plus_18_tabular_bridge.py)
for the eight implemented arms and the
[bridge experiment](../experiments/2026-09-29_01-42-50_18_claim_tabular_bridge.md)
for measured policies.
