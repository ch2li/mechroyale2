"""Offline tactical map: directional cover bits for every coarse (4 m) walkable cell.

bit (band*16 + k) = 1  <=> an enemy standing at distance BAND[band] in direction k (k*22.5 deg from +x)
from the cell, eyes 8 m above its ground, has NO line of sight to a mech standing in the cell
(aim point 6 m above the cell's ground). Rays are marched over a 1 m occluder heightfield built from
terrain + solid prisms (bridges whose underside clears 6 m are treated as open underneath).
Output: tmap.bin = little-endian uint16 x 3 per coarse cell (row-major, cnx*cny), 0 for non-walkable.
"""
import sys, time, array, json, math
import numpy as np
sys.path.insert(0, '/home/claude/bot')
import nav

ARENA = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/kit/mechfight-kit/arena'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/tmap/tmap.bin'
BANDS = (25.0, 60.0, 120.0)
NDIR = 16
t0 = time.time()
W = nav.World(ARENA, {"step_up": 1.2, "height": 9, "radius": 2.5})
nx, ny = W.nx, W.ny
g = np.frombuffer(W.ground, dtype=np.int16).astype(np.float32).reshape(ny, nx) * W.scale
top = np.frombuffer(W.top, dtype=np.float32).reshape(ny, nx)
base = np.frombuffer(W.base, dtype=np.float32).reshape(ny, nx)
solid = (top > g + 0.5) & (base < g + 6.0)          # prism that blocks a ray at mech heights
H = np.where(solid, np.maximum(g, top), g).astype(np.float32)   # occluder height field
cs, cnx, cny = W.cs, W.cnx, W.cny
cells = [c for c in range(cnx * cny) if W.nbr[c]]
cx = np.array([W.minx + (c % cnx) * cs + cs // 2 for c in cells], dtype=np.float32)
cy = np.array([W.miny + (c // cnx) * cs + cs // 2 for c in cells], dtype=np.float32)
def samp(arr, x, y):
    i = np.clip(np.rint(x - W.minx).astype(np.int32), 0, nx - 1)
    j = np.clip(np.rint(y - W.miny).astype(np.int32), 0, ny - 1)
    return arr[j, i]
walk = np.frombuffer(bytes(W.walk), dtype=np.uint8).reshape(ny, nx)
cz = samp(g, cx, cy) + 6.0
masks = np.zeros((len(cells), 3), dtype=np.uint16)
for b, D in enumerate(BANDS):
    n = int(D / 1.0)
    ts = (np.arange(1, n, dtype=np.float32) / n)[None, :]
    for k in range(NDIR):
        a = k * 2 * math.pi / NDIR
        ox = cx + D * math.cos(a); oy = cy + D * math.sin(a)
        oz = samp(g, ox, oy) + 8.0
        px = ox[:, None] + (cx - ox)[:, None] * ts
        py = oy[:, None] + (cy - oy)[:, None] * ts
        pz = oz[:, None] + (cz - oz)[:, None] * ts
        blocked = (samp(H, px, py) > pz).any(axis=1)
        # an observer position inside a wall / off-map cannot shoot from there
        obs_ok = samp(walk, ox, oy) == 1
        offmap = (ox < W.minx) | (ox > W.minx + nx - 1) | (oy < W.miny) | (oy > W.miny + ny - 1)
        covered = blocked | ~obs_ok | offmap
        masks[covered, b] |= np.uint16(1 << k)
    print('band', D, 'done', round(time.time() - t0, 1), 's', flush=True)
out = np.zeros((cnx * cny, 3), dtype='<u2')
out[np.array(cells)] = masks
out.tofile(OUT)
exp = 48 - np.unpackbits(masks.view(np.uint8), axis=1).sum(axis=1)
print('cells', len(cells), 'bytes', out.nbytes, 'exposed-bits mean', round(float(exp.mean()), 1),
      'fully hidden from 60m ring:', int(((masks[:, 1]) == 0xFFFF).sum()), 'time', round(time.time() - t0, 1))
