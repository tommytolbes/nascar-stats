/**
 * Prove the browser runs the same networks Python trained.
 *
 *   python -m flynance.export_web --reference flynance/casino/reference_policy.json
 *   node flynance/casino/verify_browser_brain.mjs
 *
 * The casino page reimplements the forward pass in JavaScript, for two networks
 * with different input encodings. That is a second implementation of the same
 * maths, so it gets the same treatment as any other port: replay every decision
 * state through it and compare against the Python reference, rather than
 * assuming a hand-translated matrix multiply and a hand-translated one-hot
 * layout are both correct.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));

// brain-data.js assigns onto `window`; give it one.
globalThis.window = globalThis;
await import(path.join('file://', here, 'brain-data.js'));

function dense(x, w, b, inN, outN) {
  const out = new Float64Array(outN);
  for (let j = 0; j < outN; j++) {
    let s = b[j];
    for (let i = 0; i < inN; i++) s += x[i] * w[i * outN + j];
    out[j] = s;
  }
  return out;
}
const relu = v => v.map(z => (z > 0 ? z : 0));

function forward(W, x, inN) {
  const h1 = relu(dense(x, W.W1, W.b1, inN, 64));
  const h2 = relu(dense(h1, W.W2, W.b2, 64, 56));
  const z = dense(h2, W.W3, W.b3, 56, 2);
  const m = Math.max(z[0], z[1]);
  const e0 = Math.exp(z[0] - m), e1 = Math.exp(z[1] - m);
  return [e0 / (e0 + e1), e1 / (e0 + e1)];
}

/* Encodings — must match flynance/encoding.py exactly. */
const specInput = (sum, up, ace) => [sum / 21, up / 10, ace];
const oneHotInput = (sum, up, ace) => {
  const x = new Array(12).fill(0);
  x[0] = sum / 21;
  if (up >= 1 && up <= 10) x[up] = 1;   // x[1] == ace ... x[10] == ten
  x[11] = ace;
  return x;
};
const fullInput = (sum, up, ace) => {
  const x = new Array(29).fill(0);
  if (sum >= 4 && sum <= 21) x[sum - 4] = 1;        // x[0] == 4 ... x[17] == 21
  if (up >= 1 && up <= 10) x[17 + up] = 1;          // x[18] == ace ... x[27] == ten
  x[28] = ace;
  return x;
};

function check(label, refFile, W, encode, inN) {
  const refPath = path.join(here, refFile);
  if (!fs.existsSync(refPath)) {
    console.log(`${label}: skipped, no ${refFile}`);
    return null;
  }
  const REF = JSON.parse(fs.readFileSync(refPath, 'utf8'));
  let worst = 0, mismatches = 0, n = 0;
  for (const [key, expected] of Object.entries(REF)) {
    const [s, up, ace] = key.split('_').map(Number);
    const got = forward(W, encode(s, up, ace), inN);
    worst = Math.max(worst, Math.abs(got[0] - expected[0]), Math.abs(got[1] - expected[1]));
    if ((got[1] > got[0]) !== (expected[1] > expected[0])) {
      mismatches++;
      console.error(`  MISMATCH ${label} ${key}: js=${got.map(p => p.toFixed(4))} py=${expected}`);
    }
    n++;
  }
  console.log(`${label.padEnd(9)} states=${n}  max prob error=${worst.toExponential(3)}  ` +
              `action mismatches=${mismatches}`);
  return { worst, mismatches };
}

const results = [
  check('spec', 'reference_policy.json', globalThis.FLY_WEIGHTS, specInput, 3),
  globalThis.ONEHOT_WEIGHTS
    ? check('one-hot', 'reference_policy_onehot.json', globalThis.ONEHOT_WEIGHTS, oneHotInput, 12)
    : null,
  globalThis.FULL_WEIGHTS
    ? check('full', 'reference_policy_full.json', globalThis.FULL_WEIGHTS, fullInput, 29)
    : null,
].filter(Boolean);

if (!results.length) {
  console.error('\nNothing verified — generate the reference files first.');
  process.exit(2);
}
const bad = results.filter(r => r.mismatches > 0 || r.worst > 1e-5);
if (bad.length) {
  console.error('\nFAIL: the browser would play a different policy than Python.');
  process.exit(1);
}
console.log('\nPASS: the casino page plays exactly the policies Python trained.');
