# The Mantis Room

A live casino table where the trained fly brain plays blackjack in your browser.

Open `index.html` in any browser. No build step, no server, no network calls.

## What is actually running

Vesper's decisions are **not scripted and not a lookup table**. All 4,010 weights
are in `brain-data.js`, and every decision is a live forward pass through the same
3 → 64 → 56 → 2 ReLU policy that `trainers.train_reinforce` produced — the same
matrix multiplies, the same softmax, evaluated in JavaScript.

The rail shows her real output: the softmax probabilities for STAND and HIT, the
raw three-number input vector, and whether the decision matches the exact optimum
from `optimal.py`. When she lands on one of her 21 known mistakes, the page says so
as it happens.

## The table

| Fly | Policy |
|---|---|
| **Vesper** | the trained 4,010-weight network |
| **Echo** | mimics the house — hits below 17 |
| **Stoic** | never takes a card |
| **Dizzy** | the *original specification's* broken training loop, which collapsed to always-hit |

Dizzy is the experiment's headline finding sitting at the table: same architecture,
trained by the loop the spec actually shipped, busting every single hand. Watch the
chip tally for a few rounds and the cost of the bug is obvious without reading a
single number in the report.

## Regenerating the data

`brain-data.js` is generated, never hand-edited:

```bash
python -m flynance.run_experiment --episodes 200000    # trains and saves the brain
python -m flynance.export_web                          # writes brain-data.js
```

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
