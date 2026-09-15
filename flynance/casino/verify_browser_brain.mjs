/**
 * Prove the browser runs the same network Python trained.
 *
 *   python -m flynance.export_web            # regenerate brain-data.js
 *   node flynance/casino/verify_browser_brain.mjs
 *
 * The casino page reimplements the forward pass in JavaScript. That is a second
 * implementation of the same maths, so it needs the same treatment as any other
 * port: replay every decision state through it and compare against the Python
 * reference, rather than assuming a hand-translated matrix multiply is correct.
 *
 * The reference file is written by:
 *   python -m flynance.export_web --reference flynance/casino/reference_policy.json
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));

// brain-data.js assigns onto `window`; give it one.
globalThis.window = globalThis;
await import(path.join('file://', here, 'brain-data.js'));
const W = globalThis.FLY_WEIGHTS;

const refPath = path.join(here, 'reference_policy.json');
if (!fs.existsSync(refPath)) {
  console.error('Missing reference_policy.json. Generate it with:\n' +
    '  python -m flynance.export_web --reference flynance/casino/reference_policy.json');
  process.exit(2);
}
const REF = JSON.parse(fs.readFileSync(refPath, 'utf8'));

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

/** Identical to flyPolicy() in index.html. */
function policy(playerSum, upcard, ace) {
  const h1 = relu(dense([playerSum / 21, upcard / 10, ace], W.W1, W.b1, 3, 64));
  const h2 = relu(dense(h1, W.W2, W.b2, 64, 56));
  const z = dense(h2, W.W3, W.b3, 56, 2);
  const m = Math.max(z[0], z[1]);
  const e0 = Math.exp(z[0] - m), e1 = Math.exp(z[1] - m);
  return [e0 / (e0 + e1), e1 / (e0 + e1)];
}

let worst = 0, mismatches = 0, n = 0;
for (const [key, expected] of Object.entries(REF)) {
  const [s, up, ace] = key.split('_').map(Number);
  const got = policy(s, up, ace);
  worst = Math.max(worst, Math.abs(got[0] - expected[0]), Math.abs(got[1] - expected[1]));
  if ((got[1] > got[0]) !== (expected[1] > expected[0])) {
    mismatches++;
    console.error(`  MISMATCH ${key}: js=${got.map(p => p.toFixed(4))} py=${expected}`);
  }
  n++;
}

console.log(`states compared:       ${n}`);
console.log(`max probability error: ${worst.toExponential(3)}`);
console.log(`action mismatches:     ${mismatches}`);

if (mismatches > 0) {
  console.error('\nFAIL: the browser would play a different policy than Python.');
  process.exit(1);
}
if (worst > 1e-5) {
  console.error(`\nFAIL: probabilities drifted by ${worst.toExponential(3)} (limit 1e-5).`);
  process.exit(1);
}
console.log('\nPASS: the casino page plays exactly the policy Python trained.');
