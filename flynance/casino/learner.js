/**
 * A fly brain that learns in the browser.
 *
 * Vesper and Iris arrive pre-trained: their weights are frozen and the casino
 * page only runs forward passes. This is the other half -- the same 4,010-weight
 * architecture with the training loop attached, so a fly can start from random
 * weights and learn blackjack while you watch.
 *
 * It is a direct port of flynance/trainers.py::train_reinforce and
 * flynance/brain.py, with the same hyperparameters: He-normal init, Adam
 * (0.9, 0.999, 1e-8), REINFORCE with return-to-go, a running-mean baseline, an
 * entropy bonus, and a learning rate decaying from 5e-4 to 2e-5.
 *
 * Because the port is a second implementation of the same maths, it is checked
 * rather than trusted: casino/verify_learner.mjs trains one in Node and asserts
 * it reaches the agreement with optimal play that the Python trainer reaches.
 *
 * Loaded as a plain script (sets window.FlyLearner); Node can import it after
 * defining globalThis.window.
 */
(function (root) {
  "use strict";

  /* ---------- deterministic randomness -------------------------------- */

  /** mulberry32 -- small, fast, seedable. A seed makes a run repeatable. */
  function makeRng(seed) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    // Box-Muller, for He-normal initialisation.
    next.normal = () => {
      let u = 0, v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return next;
  }

  /* ---------- blackjack, same rules as everywhere else ---------------- */

  const DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

  /** Total and soft-flag for a running hand sum that counts aces as 1. */
  function totals(hardSum, hasAce) {
    return hasAce && hardSum + 10 <= 21
      ? { sum: hardSum + 10, soft: 1 }
      : { sum: hardSum, soft: 0 };
  }

  /* ---------- the network --------------------------------------------- */

  const IN = 3, H1 = 64, H2 = 56, OUT = 2;
  const LR0 = 5e-4, LR1 = 2e-5, BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
  const ENTROPY_COEF = 0.01;

  function zeros(n) { return new Float64Array(n); }

  function heNormal(rng, fanIn, fanOut) {
    const w = new Float64Array(fanIn * fanOut);
    const std = Math.sqrt(2 / fanIn);
    for (let i = 0; i < w.length; i++) w[i] = rng.normal() * std;
    return w;
  }

  function Learner(seed) {
    const rng = makeRng(seed === undefined ? 12345 : seed);
    this.rng = rng;

    this.p = {
      W1: heNormal(rng, IN, H1), b1: zeros(H1),
      W2: heNormal(rng, H1, H2), b2: zeros(H2),
      W3: heNormal(rng, H2, OUT), b3: zeros(OUT),
    };
    this.g = {};
    this.m = {};
    this.v = {};
    for (const k of Object.keys(this.p)) {
      this.g[k] = zeros(this.p[k].length);
      this.m[k] = zeros(this.p[k].length);
      this.v[k] = zeros(this.p[k].length);
    }

    this.t = 0;                 // Adam timestep
    this.hands = 0;             // hands of experience
    this.horizon = 200000;      // hands over which the learning rate decays
    this.baselineSum = 0;       // running-mean baseline over returns
    this.baselineCount = 0;
    this.recent = [];           // recent rewards, for a live win-rate readout

    // Scratch buffers, reused so a long training run does not churn garbage.
    this._z1 = zeros(H1); this._a1 = zeros(H1);
    this._z2 = zeros(H2); this._a2 = zeros(H2);
    this._z3 = zeros(OUT);
  }

  Learner.prototype.forward = function (x) {
    const p = this.p, z1 = this._z1, a1 = this._a1, z2 = this._z2, a2 = this._a2, z3 = this._z3;
    for (let j = 0; j < H1; j++) {
      let s = p.b1[j];
      for (let i = 0; i < IN; i++) s += x[i] * p.W1[i * H1 + j];
      z1[j] = s; a1[j] = s > 0 ? s : 0;
    }
    for (let j = 0; j < H2; j++) {
      let s = p.b2[j];
      for (let i = 0; i < H1; i++) s += a1[i] * p.W2[i * H2 + j];
      z2[j] = s; a2[j] = s > 0 ? s : 0;
    }
    for (let j = 0; j < OUT; j++) {
      let s = p.b3[j];
      for (let i = 0; i < H2; i++) s += a2[i] * p.W3[i * OUT + j];
      z3[j] = s;
    }
    const m = Math.max(z3[0], z3[1]);
    const e0 = Math.exp(z3[0] - m), e1 = Math.exp(z3[1] - m);
    const denom = e0 + e1;
    return [e0 / denom, e1 / denom];
  };

  /** Accumulate gradients for one decision. dlogits is dLoss/dlogits. */
  Learner.prototype.backward = function (x, dlogits) {
    const p = this.p, g = this.g, a1 = this._a1, a2 = this._a2, z1 = this._z1, z2 = this._z2;

    const da2 = zeros(H2);
    for (let j = 0; j < OUT; j++) {
      const d = dlogits[j];
      if (d === 0) continue;
      g.b3[j] += d;
      for (let i = 0; i < H2; i++) {
        g.W3[i * OUT + j] += a2[i] * d;
        da2[i] += p.W3[i * OUT + j] * d;
      }
    }

    const da1 = zeros(H1);
    for (let j = 0; j < H2; j++) {
      const d = z2[j] > 0 ? da2[j] : 0;
      if (d === 0) continue;
      g.b2[j] += d;
      for (let i = 0; i < H1; i++) {
        g.W2[i * H2 + j] += a1[i] * d;
        da1[i] += p.W2[i * H2 + j] * d;
      }
    }

    for (let j = 0; j < H1; j++) {
      const d = z1[j] > 0 ? da1[j] : 0;
      if (d === 0) continue;
      g.b1[j] += d;
      for (let i = 0; i < IN; i++) g.W1[i * H1 + j] += x[i] * d;
    }
  };

  /** One bias-corrected Adam update, then zero the gradients. */
  Learner.prototype.step = function (lr) {
    this.t += 1;
    const bc1 = 1 - Math.pow(BETA1, this.t);
    const bc2 = 1 - Math.pow(BETA2, this.t);
    for (const k of Object.keys(this.p)) {
      const p = this.p[k], g = this.g[k], m = this.m[k], v = this.v[k];
      for (let i = 0; i < p.length; i++) {
        const grad = g[i];
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad;
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad * grad;
        p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + EPS);
        g[i] = 0;
      }
    }
  };

  /** The action this fly would take right now, greedily. */
  Learner.prototype.decide = function (playerSum, upcard, ace) {
    const probs = this.forward([playerSum / 21, upcard / 10, ace]);
    return { action: probs[1] > probs[0] ? 1 : 0, probs: probs };
  };

  /**
   * Sample an action, and return everything needed to learn from it later.
   *
   * Used for hands played at the table: she explores there exactly as she does
   * in training, which is what makes those hands usable as experience.
   */
  Learner.prototype.act = function (playerSum, upcard, ace) {
    const x = [playerSum / 21, upcard / 10, ace];
    const probs = this.forward(x);
    return { action: this.rng() < probs[1] ? 1 : 0, probs: probs, x: x };
  };

  /**
   * Play one hand against the house, learning from it.
   *
   * Returns the terminal reward: +1 dopaminergic, -1 octopaminergic, 0 neutral.
   */
  Learner.prototype.playAndLearn = function () {
    const rng = this.rng;
    const draw = () => DECK[Math.floor(rng() * 13)];

    // Deal.
    let hard = 0, ace = false;
    for (let i = 0; i < 2; i++) {
      const c = draw();
      hard += c; if (c === 1) ace = true;
    }
    const upcard = draw();
    const holeCard = draw();

    // Player decisions, remembering each one for the update.
    const xs = [], acts = [], ps = [];
    let reward = null;
    for (let guard = 0; guard < 25; guard++) {
      const t = totals(hard, ace);
      if (t.sum > 21) { reward = -1; break; }

      const x = [t.sum / 21, upcard / 10, t.soft];
      const probs = this.forward(x);
      const action = rng() < probs[1] ? 1 : 0;   // sample, to keep exploring
      xs.push(x); acts.push(action); ps.push(probs);

      if (action === 0) break;
      const c = draw();
      hard += c; if (c === 1) ace = true;
    }

    if (reward === null) {
      // Player stood: the dealer plays out, standing on soft 17.
      let dHard = upcard + holeCard;
      let dAce = upcard === 1 || holeCard === 1;
      for (let guard = 0; guard < 25; guard++) {
        const dt = totals(dHard, dAce);
        if (dt.sum >= 17) break;
        const c = draw();
        dHard += c; if (c === 1) dAce = true;
      }
      const player = totals(hard, ace).sum;
      const dt = totals(dHard, dAce);
      const dealer = dt.sum > 21 ? 0 : dt.sum;
      reward = player > dealer ? 1 : player < dealer ? -1 : 0;
    }

    return this.learnFromHand(xs, acts, ps, reward);
  };

  /**
   * Apply one REINFORCE update from a hand's experience.
   *
   * Split out from playAndLearn so a hand played *elsewhere* -- at the casino
   * table, dealt by the page rather than by this object's own simulation -- can
   * teach her exactly as a background hand does. The caller records each
   * decision's input, sampled action and probabilities, then hands the whole
   * hand over with its terminal reward.
   *
   * The actions must have been **sampled** from this policy, not taken greedily:
   * REINFORCE's gradient estimate is only unbiased for on-policy actions.
   */
  Learner.prototype.learnFromHand = function (xs, acts, ps, reward) {
    if (!xs.length) return reward;

    // gamma = 1 and the only non-zero reward is terminal, so every decision in
    // the hand shares the same return.
    const meanReturn = this.baselineCount ? this.baselineSum / this.baselineCount : 0;
    const advantage = reward - meanReturn;
    this.baselineSum += reward * xs.length;
    this.baselineCount += xs.length;

    for (let k = 0; k < xs.length; k++) {
      const probs = ps[k], a = acts[k];
      const dlogits = [probs[0], probs[1]];
      dlogits[a] -= 1;
      dlogits[0] *= advantage;
      dlogits[1] *= advantage;

      // Entropy bonus: dH/dz_j = -p_j (log p_j + H), subtracted from the loss.
      const lp0 = Math.log(Math.max(probs[0], 1e-12));
      const lp1 = Math.log(Math.max(probs[1], 1e-12));
      const H = -(probs[0] * lp0 + probs[1] * lp1);
      dlogits[0] += ENTROPY_COEF * probs[0] * (lp0 + H);
      dlogits[1] += ENTROPY_COEF * probs[1] * (lp1 + H);

      // Each decision is replayed so backward() sees the matching activations.
      this.forward(xs[k]);
      this.backward(xs[k], dlogits);
    }

    const progress = Math.min(this.hands / this.horizon, 1);
    this.step(LR0 + (LR1 - LR0) * progress);

    this.hands += 1;
    this.recent.push(reward);
    if (this.recent.length > 2000) this.recent.shift();
    return reward;
  };

  /** Train for n hands. Returns the mean reward over them. */
  Learner.prototype.train = function (n) {
    let total = 0;
    for (let i = 0; i < n; i++) total += this.playAndLearn();
    return total / n;
  };

  /**
   * Agreement with the exact optimum, raw and weighted by how often optimal
   * play visits each state -- the same two numbers the report quotes.
   */
  Learner.prototype.score = function (optimalPolicy, optimalFreq) {
    let n = 0, match = 0, weight = 0, weightMatch = 0;
    for (const key in optimalPolicy) {
      const parts = key.split("_");
      const sum = +parts[0], up = +parts[1], ace = +parts[2];
      const action = this.decide(sum, up, ace).action;
      const f = optimalFreq ? (optimalFreq[key] || 0) : 0;
      n++; weight += f;
      if (action === optimalPolicy[key]) { match++; weightMatch += f; }
    }
    return {
      raw: n ? match / n : 0,
      weighted: weight > 0 ? weightMatch / weight : 0,
      states: n,
    };
  };

  /** Mean reward over the most recent hands -- a live, noisy progress signal. */
  Learner.prototype.recentMean = function () {
    if (!this.recent.length) return 0;
    let s = 0;
    for (let i = 0; i < this.recent.length; i++) s += this.recent[i];
    return s / this.recent.length;
  };

  root.FlyLearner = Learner;
})(typeof window !== "undefined" ? window : globalThis);
