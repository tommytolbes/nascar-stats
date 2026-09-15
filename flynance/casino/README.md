# The Mantis Room

A live casino table where the trained fly brain plays blackjack in your browser.

Open `index.html` in any browser. No build step, no server, no network calls.

## What is actually running

Vesper's and Iris's decisions are **not scripted and not a lookup table**. Both
sets of weights are in `brain-data.js`, and every decision is a live forward pass
through the same 3 → 64 → 56 → 2 (Vesper) or 12 → 64 → 56 → 2 (Iris) ReLU policy
that `trainers.train_reinforce` produced — the same matrix multiplies, the same
softmax, evaluated in JavaScript.

## Nova: learning in the browser

Vesper and Iris arrive pre-trained and frozen — at the table they only run forward
passes, and will make the identical decision in the identical spot forever. Nova is
the other half. She starts from random He-normal weights and runs the full REINFORCE
loop live: return-to-go, a running-mean baseline, an entropy bonus, and Adam decaying
from 5e-4 to 2e-5, exactly as `trainers.py` does it.

She learns down two paths at once:

- **Background training**, in time-boxed chunks (8 ms per animation frame, so the
  table stays responsive) at roughly 5,000 hands per second. This is what carries
  her to ~88% weighted agreement in about 200,000 hands — under a minute of
  watching. Pause it any time.
- **Hands at the table**, one gradient step each. Every hand she plays in front of
  you is real experience: the nursery counter ticks, the neuromodulatory pulse
  fires (+1 dopamine / −1 octopamine / 0 neutral), and her weights change. This
  keeps working after background training finishes, and while it is paused.

At the table she *samples* from her policy rather than taking the greedy action.
That matters: REINFORCE's gradient estimate is only unbiased for on-policy actions,
so a greedily-played hand would not be valid experience to learn from. Exploring is
what makes the hand teachable.

The nursery panel shows her hands lived, her agreement, her recent reward, how many
hands this table has taught her, and a 280-cell grid of every decision, lit where she
matches optimal play. She also brightens as she learns: her colour interpolates from
slate toward brass with her agreement score.

One honest caveat on scale: a table hand is one update against the ~200,000 it takes
to learn this game, so you will not see the grid change from dealing a few. The
counter and the pulse show the mechanism; the background loop supplies the volume.

Nothing about this is a simulation of learning — it is the learning, in
`learner.js`, a direct port of the Python trainer.

### Verifying that she actually learns

A ported *training loop* can be wrong in ways a forward-pass check never catches — a
sign error in the advantage, a missing ReLU mask, a biased baseline — and the symptom
is just "learns slightly worse", invisible without a reference:

```bash
node flynance/casino/verify_learner.mjs 200000
```

It trains one in Node and asserts she reaches the same policy quality the Python
trainer reaches. Current result: **88.5% weighted agreement**, against Python's 88.4%.

## The encoding experiment, seated at the table

Vesper and Iris differ in exactly one thing: how they are shown the dealer's card.
Vesper gets the spec's `upcard / 10`, which puts an ace at `0.1` — numerically the
weakest value on the axis, while strategically it is the dealer's strongest card.
Iris gets a one-hot vector, so each upcard is free to mean whatever it means.

| | Weights | Exact EV | Cells wrong | Weighted agreement |
|---|---|---|---|---|
| Vesper (spec encoding) | 4,010 | −0.056725 | 21 / 280 | 88.4% |
| Iris (one-hot upcard) | 4,586 | −0.052257 | 17 / 280 | 92.4% |
| **Juno (everything one-hot)** | 5,674 | **−0.046957** | **3 / 280** | **98.3%** |
| *perfect play* | — | −0.046556 | 0 | 100% |

Watch the dealer-ace and dealer-7-through-10 hands in particular: that is where
Vesper stands on a hard 15 or 16 while Iris and Juno take the card.

Juno's three remaining mistakes are 12 against a dealer 4, 5 and 6 — the most
marginal calls in the game, costing 0.00026 EV per hand between them. They are
hard for the same reason they are cheap: when two actions are almost equally
good, there is almost no reward signal to separate them.

These are *exact* expected values, computed by dynamic programming rather than by
playing hands. At 10,000 hands the sampling error is about ±0.0096 — wider than
the entire gap between the best fly here and perfect play.

The rail shows her real output: the softmax probabilities for STAND and HIT, the
raw three-number input vector, and whether the decision matches the exact optimum
from `optimal.py`. When she lands on one of her 21 known mistakes, the page says so
as it happens.

## The table

| Fly | Policy |
|---|---|
| **Vesper** | the trained 4,010-weight network, spec encoding (`upcard / 10`) |
| **Iris** | the same mushroom body with a one-hot dealer upcard — 4,586 weights |
| **Juno** | everything one-hot — 5,674 weights, and only 3 of 280 cells wrong |
| **Nova** | the same architecture with the *training loop still attached* — learns in your browser |
| **Echo** | mimics the house — hits below 17 |
| **Stoic** | never takes a card |
| **Dizzy** | the *original specification's* broken training loop, which collapsed to always-hit — subject to one house rule: he stands on 21 |

Dizzy is the experiment's headline finding sitting at the table: same architecture,
trained by the loop the spec actually shipped, busting every single hand.

The one departure from that policy: Dizzy stands on 21. The real always-hit policy
hits there too — a certain bust, since even an ace takes 21 to 22 — and the table
declines to stage it. Every other total he hits, so the collapse is intact and he
still busts essentially every hand; he just no longer does it in the single most
absurd way available. Watch the
chip tally for a few rounds and the cost of the bug is obvious without reading a
single number in the report.

## Regenerating the data

`brain-data.js` is generated, never hand-edited:

```bash
# --ablation also trains and saves Iris's one-hot brain; without it only Vesper
# is produced and export_web simply omits the one-hot weights.
python -m flynance.run_experiment --episodes 200000 --ablation
python -m flynance.export_web
```

The `.npz` model files are deliberately not committed (they are regenerable, and
the repo ignores binaries), so a fresh clone needs that training run before the
export can be reproduced. `brain-data.js` itself *is* committed, so the page
works immediately after cloning.

## Verifying the browser plays the real policy

A second implementation of the same maths deserves the same scrutiny as a port,
so it is checked rather than assumed:

```bash
python -m flynance.export_web --reference flynance/casino/reference_policy.json
node flynance/casino/verify_browser_brain.mjs
```

This replays all 280 decision states through the JavaScript forward pass and
compares against Python. Current result: **0 action mismatches**, maximum
probability error **2.8e-06** (entirely the 6-decimal rounding in the export).
The Python half of the pipeline is covered by `tests/test_flynance_export_web.py`.

## A note on the file format

`index.html` is authored as a Claude Artifact fragment: it opens with `<title>`
and has no `<!doctype>`/`<html>`/`<body>` wrapper, because those are supplied at
publish time. Browsers construct them automatically, so the file still opens
correctly from disk — it is written this way so the repository copy and the
published page stay a single source of truth.
