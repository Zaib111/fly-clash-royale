// Replays a match recorded by test_js_parity.py in the JS engine and prints its own record.
import { readFileSync } from 'node:fs';
import { Match, ScriptedBot, applyAction, observe, actionMask, DECISION_TICKS } from '../web/js/engine/game.js';

const sc = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const m = new Match(1, sc.cycles);
const bot = new ScriptedBot();
const out = { checkpoints: [], scripted: [] };
let k = 0;
while (!m.done && k < sc.actions.length) {
  const [a0, a1] = sc.actions[k];
  out.scripted.push(bot.act(m, 0));        // JS scripted bot's choice for seat 0 on the same state
  if (k % 20 === 0) {
    const [sp0, v0] = observe(m, 0);
    const [sp1, v1] = observe(m, 1);
    out.checkpoints.push({
      tick: m.ticks, elixir: m.elixir.slice(), towers: m.towers.flat().map((t) => [t.hp, t.alive, t.active]),
      units: m.units.map((u) => [u.uid, u.card, u.owner, u.x, u.y, u.hp, u.deploy]),
      obs: [Array.from(sp0), Array.from(v0), Array.from(sp1), Array.from(v1)],
      mask: [Array.from(actionMask(m, 0)), Array.from(actionMask(m, 1))],
    });
  }
  applyAction(m, 0, a0);
  applyAction(m, 1, a1);
  m.step(DECISION_TICKS);
  k++;
}
out.final = { tick: m.ticks, done: m.done, winner: m.winner, crowns: [m.crowns(0), m.crowns(1)] };
console.log(JSON.stringify(out));
