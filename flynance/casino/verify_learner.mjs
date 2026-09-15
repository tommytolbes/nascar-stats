/**
 * Prove the browser's learning fly actually learns.
 *
 *   node flynance/casino/verify_learner.mjs [hands]
 *
 * learner.js is a JavaScript port of trainers.py::train_reinforce. A port of a
 * training loop can be subtly wrong in ways a forward-pass check will never
 * catch -- a sign error in the advantage, a missing ReLU mask, a baseline that
 * biases the gradient -- and the symptom is simply "it learns a bit worse",
 * which is invisible without a reference. So it is trained here against the same
 * target the Python trainer hits.
 *
 * Python at 200,000 hands reaches ~88% weighted agreement with the exact
 * optimum. This asserts the port lands in the same place.
 */
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
globalThis.window = globalThis;
await import(path.join('file://', here, 'brain-data.js'));
await import(path.join('file://', here, 'learner.js'));

const HANDS = Number(process.argv[2] || 200000);
const TARGET_WEIGHTED = 0.80;   // Python reaches ~0.884; fail well below that

const fly = new globalThis.FlyLearner(7);
console.log(`training ${HANDS.toLocaleString()} hands...\n`);
console.log('     hands   mean reward     raw     weighted');

const started = Date.now();
const checkpoints = 10;
for (let i = 1; i <= checkpoints; i++) {
  fly.train(Math.floor(HANDS / checkpoints));
  const s = fly.score(globalThis.OPTIMAL_POLICY, globalThis.OPTIMAL_FREQ);
  console.log(
    `${fly.hands.toString().padStart(10)}   ${fly.recentMean().toFixed(4).padStart(11)}` +
    `   ${(s.raw * 100).toFixed(1).padStart(5)}%   ${(s.weighted * 100).toFixed(1).padStart(8)}%`
  );
}

const elapsed = (Date.now() - started) / 1000;
const final = fly.score(globalThis.OPTIMAL_POLICY, globalThis.OPTIMAL_FREQ);
console.log(`\nrate: ${Math.round(fly.hands / elapsed).toLocaleString()} hands/sec`);
console.log(`final weighted agreement: ${(final.weighted * 100).toFixed(1)}%`);

if (final.weighted < TARGET_WEIGHTED) {
  console.error(`\nFAIL: expected at least ${(TARGET_WEIGHTED * 100).toFixed(0)}% weighted ` +
                `agreement, got ${(final.weighted * 100).toFixed(1)}%. The port is not learning ` +
                `as well as the Python trainer.`);
  process.exit(1);
}
console.log('\nPASS: the browser learner reaches the same policy quality as Python.');
