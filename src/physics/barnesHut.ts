import { G, MAX_TREE_DEPTH, MAX_TREE_NODES, N_MACRO, THETA_SQ } from './constants.ts';

const MAX_RAW_NODES = 98304;

export class BarnesHutTree {
  // Raw octree build arrays (pre-allocated, zero GC)
  private readonly children = new Int32Array(MAX_RAW_NODES * 8);
  private readonly rawType = new Uint8Array(MAX_RAW_NODES); // 0=empty, 1=single leaf, 2=internal, 3=multi leaf
  private readonly rawParticle = new Int32Array(MAX_RAW_NODES);
  private readonly rawCx = new Float64Array(MAX_RAW_NODES);
  private readonly rawCy = new Float64Array(MAX_RAW_NODES);
  private readonly rawCz = new Float64Array(MAX_RAW_NODES);
  private readonly rawHalfSize = new Float64Array(MAX_RAW_NODES);

  private readonly rawMass = new Float64Array(MAX_RAW_NODES);
  private readonly rawWx = new Float64Array(MAX_RAW_NODES);
  private readonly rawWy = new Float64Array(MAX_RAW_NODES);
  private readonly rawWz = new Float64Array(MAX_RAW_NODES);
  private readonly rawWeps = new Float64Array(MAX_RAW_NODES);

  // Tight particle AABB bounds per node for accurate, tight Barnes-Hut MAC
  private readonly rawMinX = new Float64Array(MAX_RAW_NODES);
  private readonly rawMaxX = new Float64Array(MAX_RAW_NODES);
  private readonly rawMinY = new Float64Array(MAX_RAW_NODES);
  private readonly rawMaxY = new Float64Array(MAX_RAW_NODES);
  private readonly rawMinZ = new Float64Array(MAX_RAW_NODES);
  private readonly rawMaxZ = new Float64Array(MAX_RAW_NODES);

  private rawCount = 0;

  // Flattened preorder GPU-ready buffers
  public readonly treePosMass = new Float32Array(MAX_TREE_NODES * 4);
  public readonly treeMeta = new Float32Array(MAX_TREE_NODES * 4);

  // Fast CPU traversal side arrays
  private readonly treeSizeSq = new Float32Array(MAX_TREE_NODES);
  private readonly treeSkip = new Int32Array(MAX_TREE_NODES);

  public nodeCount = 0;

  private allocRawNode(cx: number, cy: number, cz: number, halfSize: number): number {
    const idx = this.rawCount;
    if (idx >= MAX_RAW_NODES) {
      throw new Error(`Octree raw node overflow (${MAX_RAW_NODES})`);
    }
    this.rawCount = idx + 1;
    this.rawType[idx] = 0;
    this.rawParticle[idx] = -1;
    this.rawCx[idx] = cx;
    this.rawCy[idx] = cy;
    this.rawCz[idx] = cz;
    this.rawHalfSize[idx] = halfSize;
    this.rawMass[idx] = 0;
    this.rawWx[idx] = 0;
    this.rawWy[idx] = 0;
    this.rawWz[idx] = 0;
    this.rawWeps[idx] = 0;
    this.rawMinX[idx] = Infinity;
    this.rawMaxX[idx] = -Infinity;
    this.rawMinY[idx] = Infinity;
    this.rawMaxY[idx] = -Infinity;
    this.rawMinZ[idx] = Infinity;
    this.rawMaxZ[idx] = -Infinity;
    const cBase = idx * 8;
    for (let k = 0; k < 8; k++) {
      this.children[cBase + k] = -1;
    }
    return idx;
  }

  private getOctant(nodeIdx: number, px: number, py: number, pz: number): number {
    let oct = 0;
    if (px >= this.rawCx[nodeIdx]) oct |= 1;
    if (py >= this.rawCy[nodeIdx]) oct |= 2;
    if (pz >= this.rawCz[nodeIdx]) oct |= 4;
    return oct;
  }

  private getOrCreateChild(nodeIdx: number, oct: number): number {
    const slot = nodeIdx * 8 + oct;
    let child = this.children[slot];
    if (child >= 0) return child;

    const hs = this.rawHalfSize[nodeIdx] * 0.5;
    const cx = this.rawCx[nodeIdx] + ((oct & 1) !== 0 ? hs : -hs);
    const cy = this.rawCy[nodeIdx] + ((oct & 2) !== 0 ? hs : -hs);
    const cz = this.rawCz[nodeIdx] + ((oct & 4) !== 0 ? hs : -hs);
    child = this.allocRawNode(cx, cy, cz, hs);
    this.children[slot] = child;
    return child;
  }

  private setLeafParticle(
    nodeIdx: number,
    pIdx: number,
    px: number,
    py: number,
    pz: number,
    pm: number,
    peps: number,
  ): void {
    this.rawType[nodeIdx] = 1;
    this.rawParticle[nodeIdx] = pIdx;
    this.rawMass[nodeIdx] = pm;
    this.rawWx[nodeIdx] = pm * px;
    this.rawWy[nodeIdx] = pm * py;
    this.rawWz[nodeIdx] = pm * pz;
    this.rawWeps[nodeIdx] = pm * peps;
    this.rawMinX[nodeIdx] = px;
    this.rawMaxX[nodeIdx] = px;
    this.rawMinY[nodeIdx] = py;
    this.rawMaxY[nodeIdx] = py;
    this.rawMinZ[nodeIdx] = pz;
    this.rawMaxZ[nodeIdx] = pz;
  }

  private insertParticle(
    rootIdx: number,
    pIdx: number,
    pos: Float32Array,
    vel: Float32Array,
  ): void {
    const px = pos[pIdx * 4];
    const py = pos[pIdx * 4 + 1];
    const pz = pos[pIdx * 4 + 2];
    const pm = pos[pIdx * 4 + 3];
    const peps = vel[pIdx * 4 + 3];

    let nodeIdx = rootIdx;
    let depth = 0;

    while (true) {
      const nType = this.rawType[nodeIdx];

      if (nType === 0) {
        this.setLeafParticle(nodeIdx, pIdx, px, py, pz, pm, peps);
        return;
      }

      if (nType === 3 || depth >= MAX_TREE_DEPTH) {
        this.rawType[nodeIdx] = 3;
        this.rawParticle[nodeIdx] = -2;
        this.rawMass[nodeIdx] += pm;
        this.rawWx[nodeIdx] += pm * px;
        this.rawWy[nodeIdx] += pm * py;
        this.rawWz[nodeIdx] += pm * pz;
        this.rawWeps[nodeIdx] += pm * peps;
        if (px < this.rawMinX[nodeIdx]) this.rawMinX[nodeIdx] = px;
        if (px > this.rawMaxX[nodeIdx]) this.rawMaxX[nodeIdx] = px;
        if (py < this.rawMinY[nodeIdx]) this.rawMinY[nodeIdx] = py;
        if (py > this.rawMaxY[nodeIdx]) this.rawMaxY[nodeIdx] = py;
        if (pz < this.rawMinZ[nodeIdx]) this.rawMinZ[nodeIdx] = pz;
        if (pz > this.rawMaxZ[nodeIdx]) this.rawMaxZ[nodeIdx] = pz;
        return;
      }

      if (nType === 1) {
        const oldP = this.rawParticle[nodeIdx];
        const oldPx = pos[oldP * 4];
        const oldPy = pos[oldP * 4 + 1];
        const oldPz = pos[oldP * 4 + 2];
        const oldPm = pos[oldP * 4 + 3];
        const oldEps = vel[oldP * 4 + 3];

        this.rawType[nodeIdx] = 2;
        this.rawParticle[nodeIdx] = -1;
        this.rawMass[nodeIdx] = 0;
        this.rawWx[nodeIdx] = 0;
        this.rawWy[nodeIdx] = 0;
        this.rawWz[nodeIdx] = 0;
        this.rawWeps[nodeIdx] = 0;

        const oldOct = this.getOctant(nodeIdx, oldPx, oldPy, oldPz);
        const oldChild = this.getOrCreateChild(nodeIdx, oldOct);
        this.setLeafParticle(oldChild, oldP, oldPx, oldPy, oldPz, oldPm, oldEps);
      }

      const oct = this.getOctant(nodeIdx, px, py, pz);
      nodeIdx = this.getOrCreateChild(nodeIdx, oct);
      depth++;
    }
  }

  private flattenPreorder(rawIdx: number): void {
    // Compress single-child chains
    let curr = rawIdx;
    while (this.rawType[curr] === 2) {
      let childCount = 0;
      let soleChild = -1;
      const cBase = curr * 8;
      for (let k = 0; k < 8; k++) {
        const ch = this.children[cBase + k];
        if (ch >= 0 && this.rawMass[ch] > 0) {
          childCount++;
          soleChild = ch;
        }
      }
      if (childCount === 1) {
        curr = soleChild;
      } else {
        break;
      }
    }

    const outIdx = this.nodeCount;
    if (outIdx >= MAX_TREE_NODES) {
      throw new Error(`Flattened octree overflow (${MAX_TREE_NODES})`);
    }
    this.nodeCount = outIdx + 1;

    const mass = this.rawMass[curr];
    const invMass = mass > 0 ? 1.0 / mass : 0;
    const pBase = outIdx * 4;
    this.treePosMass[pBase] = this.rawWx[curr] * invMass;
    this.treePosMass[pBase + 1] = this.rawWy[curr] * invMass;
    this.treePosMass[pBase + 2] = this.rawWz[curr] * invMass;
    this.treePosMass[pBase + 3] = mass;

    const isLeaf = this.rawType[curr] !== 2;
    let size = -1.0;
    if (!isLeaf) {
      const ex = this.rawMaxX[curr] - this.rawMinX[curr];
      const ey = this.rawMaxY[curr] - this.rawMinY[curr];
      const ez = this.rawMaxZ[curr] - this.rawMinZ[curr];
      // Bounding AABB max side (tighter and physically exact Barnes-Hut cell extent)
      size = Math.max(ex, ey, ez);
    }
    const avgEpsSq = this.rawWeps[curr] * invMass;

    this.treeMeta[pBase] = size;
    this.treeMeta[pBase + 2] = avgEpsSq;
    this.treeMeta[pBase + 3] = 0.0;
    this.treeSizeSq[outIdx] = size < 0.0 ? -1.0 : size * size;

    if (!isLeaf) {
      const cBase = curr * 8;
      for (let k = 0; k < 8; k++) {
        const ch = this.children[cBase + k];
        if (ch >= 0 && this.rawMass[ch] > 0) {
          this.flattenPreorder(ch);
        }
      }
    }

    this.treeMeta[pBase + 1] = this.nodeCount;
    this.treeSkip[outIdx] = this.nodeCount;
  }

  public build(pos: Float32Array, vel: Float32Array, count: number = N_MACRO): void {
    this.rawCount = 0;
    this.nodeCount = 0;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < count; i++) {
      const base = i * 4;
      const x = pos[base];
      const y = pos[base + 1];
      const z = pos[base + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }

    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    const cz = 0.5 * (minZ + maxZ);
    const halfSize = Math.max(1.0, 0.51 * Math.max(maxX - minX, maxY - minY, maxZ - minZ));

    const rootIdx = this.allocRawNode(cx, cy, cz, halfSize);

    for (let i = 0; i < count; i++) {
      this.insertParticle(rootIdx, i, pos, vel);
    }

    // Bottom-up mass, COM, eps^2, and tight AABB bounds
    for (let u = this.rawCount - 1; u >= 0; u--) {
      if (this.rawType[u] !== 2) continue;
      let m = 0;
      let wx = 0;
      let wy = 0;
      let wz = 0;
      let we = 0;
      let x0 = Infinity, x1 = -Infinity;
      let y0 = Infinity, y1 = -Infinity;
      let z0 = Infinity, z1 = -Infinity;
      const cBase = u * 8;
      for (let k = 0; k < 8; k++) {
        const ch = this.children[cBase + k];
        if (ch >= 0 && this.rawMass[ch] > 0) {
          m += this.rawMass[ch];
          wx += this.rawWx[ch];
          wy += this.rawWy[ch];
          wz += this.rawWz[ch];
          we += this.rawWeps[ch];
          if (this.rawMinX[ch] < x0) x0 = this.rawMinX[ch];
          if (this.rawMaxX[ch] > x1) x1 = this.rawMaxX[ch];
          if (this.rawMinY[ch] < y0) y0 = this.rawMinY[ch];
          if (this.rawMaxY[ch] > y1) y1 = this.rawMaxY[ch];
          if (this.rawMinZ[ch] < z0) z0 = this.rawMinZ[ch];
          if (this.rawMaxZ[ch] > z1) z1 = this.rawMaxZ[ch];
        }
      }
      this.rawMass[u] = m;
      this.rawWx[u] = wx;
      this.rawWy[u] = wy;
      this.rawWz[u] = wz;
      this.rawWeps[u] = we;
      this.rawMinX[u] = x0;
      this.rawMaxX[u] = x1;
      this.rawMinY[u] = y0;
      this.rawMaxY[u] = y1;
      this.rawMinZ[u] = z0;
      this.rawMaxZ[u] = z1;
    }

    this.flattenPreorder(rootIdx);
  }

  public computeAccelStackless(
    x: number,
    y: number,
    z: number,
    selfEpsSq: number,
    outAccel: Float64Array,
    outOffset: number,
    thetaSq: number = THETA_SQ,
  ): void {
    let ax = 0.0;
    let ay = 0.0;
    let az = 0.0;

    const posMass = this.treePosMass;
    const meta = this.treeMeta;
    const sizeSqArr = this.treeSizeSq;
    const skipArr = this.treeSkip;
    const nNodes = this.nodeCount;
    let node = 0;

    while (node < nNodes) {
      const base = node * 4;
      const sSq = sizeSqArr[node];
      const dx = posMass[base] - x;
      const dy = posMass[base + 1] - y;
      const dz = posMass[base + 2] - z;
      const distSq = dx * dx + dy * dy + dz * dz;

      if (sSq < 0.0 || sSq < thetaSq * distSq) {
        if (distSq > 1e-10) {
          const mass = posMass[base + 3];
          const epsPairSq = 0.5 * (selfEpsSq + meta[base + 2]);
          const denom = distSq + epsPairSq;
          const invR3 = mass / (denom * Math.sqrt(denom));
          ax += dx * invR3;
          ay += dy * invR3;
          az += dz * invR3;
        }
        node = skipArr[node];
      } else {
        node++;
      }
    }

    outAccel[outOffset] = G * ax;
    outAccel[outOffset + 1] = G * ay;
    outAccel[outOffset + 2] = G * az;
  }

  public computeAllAccels(
    pos: Float32Array,
    vel: Float32Array,
    accOut: Float64Array,
    count: number = N_MACRO,
    thetaSq: number = THETA_SQ,
  ): void {
    const posMass = this.treePosMass;
    const meta = this.treeMeta;
    const sizeSqArr = this.treeSizeSq;
    const skipArr = this.treeSkip;
    const nNodes = this.nodeCount;

    let netFx = 0.0;
    let netFy = 0.0;
    let netFz = 0.0;
    let totalMass = 0.0;

    for (let i = 0; i < count; i++) {
      const pBase = i * 4;
      const x = pos[pBase];
      const y = pos[pBase + 1];
      const z = pos[pBase + 2];
      const pm = pos[pBase + 3];
      const selfEpsSq = vel[pBase + 3];

      let ax = 0.0;
      let ay = 0.0;
      let az = 0.0;
      let node = 0;

      while (node < nNodes) {
        const base = node * 4;
        const sSq = sizeSqArr[node];
        const dx = posMass[base] - x;
        const dy = posMass[base + 1] - y;
        const dz = posMass[base + 2] - z;
        const distSq = dx * dx + dy * dy + dz * dz;

        if (sSq < 0.0 || sSq < thetaSq * distSq) {
          if (distSq > 1e-10) {
            const denom = distSq + 0.5 * (selfEpsSq + meta[base + 2]);
            const invR3 = posMass[base + 3] / (denom * Math.sqrt(denom));
            ax += dx * invR3;
            ay += dy * invR3;
            az += dz * invR3;
          }
          node = skipArr[node];
        } else {
          node++;
        }
      }

      const gx = G * ax;
      const gy = G * ay;
      const gz = G * az;
      const aBase = i * 3;
      accOut[aBase] = gx;
      accOut[aBase + 1] = gy;
      accOut[aBase + 2] = gz;

      netFx += pm * gx;
      netFy += pm * gy;
      netFz += pm * gz;
      totalMass += pm;
    }

    // Enforce Newton's 3rd Law (eliminate tiny asymmetric octree monopole COM drift)
    if (totalMass > 0) {
      const meanAx = netFx / totalMass;
      const meanAy = netFy / totalMass;
      const meanAz = netFz / totalMass;
      for (let i = 0; i < count; i++) {
        const aBase = i * 3;
        accOut[aBase] -= meanAx;
        accOut[aBase + 1] -= meanAy;
        accOut[aBase + 2] -= meanAz;
      }
    }
  }
}

export function stepLeapfrogKDK(
  tree: BarnesHutTree,
  pos: Float32Array,
  vel: Float32Array,
  acc: Float64Array,
  dt: number,
  count: number = N_MACRO,
  hasInitialAccel: boolean = false,
): void {
  if (!hasInitialAccel) {
    tree.build(pos, vel, count);
    tree.computeAllAccels(pos, vel, acc, count);
  }

  const halfDt = 0.5 * dt;

  for (let i = 0; i < count; i++) {
    const pBase = i * 4;
    const aBase = i * 3;
    const vx = vel[pBase] + halfDt * acc[aBase];
    const vy = vel[pBase + 1] + halfDt * acc[aBase + 1];
    const vz = vel[pBase + 2] + halfDt * acc[aBase + 2];
    vel[pBase] = vx;
    vel[pBase + 1] = vy;
    vel[pBase + 2] = vz;

    pos[pBase] += dt * vx;
    pos[pBase + 1] += dt * vy;
    pos[pBase + 2] += dt * vz;
  }

  tree.build(pos, vel, count);
  tree.computeAllAccels(pos, vel, acc, count);

  for (let i = 0; i < count; i++) {
    const pBase = i * 4;
    const aBase = i * 3;
    vel[pBase] += halfDt * acc[aBase];
    vel[pBase + 1] += halfDt * acc[aBase + 1];
    vel[pBase + 2] += halfDt * acc[aBase + 2];
  }
}

const shrinkDist = new Float64Array(512);
const shrinkIdx = new Int32Array(512);

export function findBulgeCenter(
  pos: Float32Array,
  startIdx: number,
  count: number,
): [number, number, number] {
  let cx = 0;
  let cy = 0;
  let cz = 0;
  let mSum = 0;

  for (let i = 0; i < count; i++) {
    const pBase = (startIdx + i) * 4;
    const m = pos[pBase + 3];
    cx += m * pos[pBase];
    cy += m * pos[pBase + 1];
    cz += m * pos[pBase + 2];
    mSum += m;
    shrinkIdx[i] = startIdx + i;
  }
  cx /= mSum;
  cy /= mSum;
  cz /= mSum;

  let activeCount = count;
  for (let iter = 0; iter < 4; iter++) {
    let maxR2 = 0;
    for (let i = 0; i < activeCount; i++) {
      const pBase = shrinkIdx[i] * 4;
      const dx = pos[pBase] - cx;
      const dy = pos[pBase + 1] - cy;
      const dz = pos[pBase + 2] - cz;
      const r2 = dx * dx + dy * dy + dz * dz;
      shrinkDist[i] = r2;
      if (r2 > maxR2) maxR2 = r2;
    }

    const cutR2 = maxR2 * 0.30;
    let nextActive = 0;
    let ncx = 0;
    let ncy = 0;
    let ncz = 0;
    let nm = 0;

    for (let i = 0; i < activeCount; i++) {
      if (shrinkDist[i] <= cutR2) {
        const pIdx = shrinkIdx[i];
        const pBase = pIdx * 4;
        const m = pos[pBase + 3];
        shrinkIdx[nextActive++] = pIdx;
        ncx += m * pos[pBase];
        ncy += m * pos[pBase + 1];
        ncz += m * pos[pBase + 2];
        nm += m;
      }
    }

    if (nextActive < 24) break;
    cx = ncx / nm;
    cy = ncy / nm;
    cz = ncz / nm;
    activeCount = nextActive;
  }

  return [cx, cy, cz];
}
