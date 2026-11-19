// Deterministic Mulberry32 PRNG and Box-Muller Gaussian sampler

export type RNG = () => number;

export function createRng(seed: number): RNG {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: RNG): number {
  let u1 = rng();
  while (u1 <= 1e-12) u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function randomUnitVector(rng: RNG): [number, number, number] {
  const cosTheta = 2.0 * rng() - 1.0;
  const sinTheta = Math.sqrt(Math.max(0, 1.0 - cosTheta * cosTheta));
  const phi = 2.0 * Math.PI * rng();
  return [
    sinTheta * Math.cos(phi),
    sinTheta * Math.sin(phi),
    cosTheta,
  ];
}
