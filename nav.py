"""Static map model: walk grid, line-of-sight heights, and a time-sliced A* planner.

Grids
- fine grid: 1 m cells aligned with terrain.i16 (cell i,j centered at minX+i, minY+j).
  * ground[k]   terrain height (cm, int16)
  * top[k]/base[k]  extent of the tallest prism over the cell (for shot line-of-sight)
  * walk[k]     1 = a mech centre can stand here (walls dilated by the mech radius)
- coarse grid: 4 m cells for A*; nbr[c] is a bitmask of the 8 directions that are walkable
  (checked by sampling the fine walk grid along the segment).
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import sys
import time

DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
SQ2 = math.sqrt(2.0)


class World:
    def __init__(self, map_dir: str, mech_cfg: dict):
        t0 = time.perf_counter()
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        T = w["terrain"]
        self.nx, self.ny = nx, ny = int(T["nx"]), int(T["ny"])
        self.minx, self.miny = float(T["minX"]), float(T["minY"])
        self.step = float(T.get("step", 1.0))
        self.scale = float(T.get("scale", 0.01))
        g = array.array("h")
        with open(os.path.join(map_dir, "v2", T.get("file", "terrain.i16")), "rb") as f:
            g.frombytes(f.read())
        if len(g) != nx * ny:
            raise ValueError("terrain size mismatch")
        if sys.byteorder != "little":
            g.byteswap()
        self.ground = g
        step_up = float(mech_cfg.get("step_up", 1.2))
        height = float(mech_cfg.get("height", 9.0))
        radius = float(mech_cfg.get("radius", 2.5))

        n = nx * ny
        self.top = array.array("f", [-1e9]) * n
        self.base = array.array("f", [1e9]) * n
        rows = [0] * ny  # blocked bitmask per row (bit i = cell i)

        prisms = sorted(w.get("prisms", []), key=lambda p: p["top"])
        for p in prisms:
            ring = p["ring"]
            polys = [ring] + list(p.get("holes") or [])
            edges = []
            ys = ring[1::2]
            for poly in polys:
                m = len(poly) // 2
                for a in range(m):
                    x0, y0 = poly[2 * a], poly[2 * a + 1]
                    x1, y1 = poly[2 * ((a + 1) % m)], poly[2 * ((a + 1) % m) + 1]
                    if y0 != y1:
                        edges.append((x0, y0, x1, y1))
            j0 = max(0, int(math.ceil(min(ys) - self.miny)))
            j1 = min(ny - 1, int(math.floor(max(ys) - self.miny)))
            ptop, pbase = float(p["top"]), float(p["base"])
            kind = p.get("kind", "building")
            for j in range(j0, j1 + 1):
                yc = self.miny + j
                xs_cross = []
                for (x0, y0, x1, y1) in edges:
                    if (y0 <= yc < y1) or (y1 <= yc < y0):
                        xs_cross.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
                if len(xs_cross) < 2:
                    continue
                xs_cross.sort()
                row = j * nx
                for q in range(0, len(xs_cross) - 1, 2):
                    a = max(0, int(math.ceil(xs_cross[q] - self.minx)))
                    b = min(nx - 1, int(math.floor(xs_cross[q + 1] - self.minx)))
                    if b < a:
                        continue
                    # heights for LOS (sorted by top: taller overwrite)
                    cnt = b - a + 1
                    self.top[row + a: row + b + 1] = array.array("f", [ptop]) * cnt
                    self.base[row + a: row + b + 1] = array.array("f", [pbase]) * cnt
                    # movement blocking: use ground at the span middle (cheap) and ends
                    gmid = max(g[row + a], g[row + b], g[row + (a + b) // 2]) * self.scale
                    gmin = min(g[row + a], g[row + b], g[row + (a + b) // 2]) * self.scale
                    if ptop - gmid > step_up and pbase - gmin < height + 0.5:
                        rows[j] |= ((1 << cnt) - 1) << a
            _ = kind
        # steep terrain (a rise of more than step_up within one metre is a wall)
        lim = int(step_up / self.scale)
        for j in range(ny):
            row = j * nx
            bits = 0
            for i in range(nx - 1):
                d = g[row + i + 1] - g[row + i]
                if d > lim or -d > lim:
                    bits |= 3 << i
            if j + 1 < ny:
                rown = row + nx
                for i in range(nx):
                    d = g[rown + i] - g[row + i]
                    if d > lim or -d > lim:
                        bits |= 1 << i
                        rows[j + 1] |= 1 << i
            rows[j] |= bits
        # map edge
        full = (1 << nx) - 1
        rows[0] = rows[-1] = full
        for j in range(ny):
            rows[j] |= 1 | (1 << (nx - 1))
        self.raw_rows = rows
        # dilate by mech radius (Chebyshev, in 1 m cells)
        r = max(1, int(round(radius)))
        hrow = []
        for v in rows:
            d = v
            for s in range(1, r + 1):
                d |= (v << s) | (v >> s)
            hrow.append(d & full)
        dil = []
        for j in range(ny):
            d = 0
            for s in range(-r, r + 1):
                jj = j + s
                if 0 <= jj < ny:
                    d |= hrow[jj]
            dil.append(d)
        walk = bytearray(n)
        tr = bytes.maketrans(b"01", b"\x01\x00")
        for j in range(ny):
            s = format(dil[j], "0%db" % nx)[::-1]
            walk[j * nx:(j + 1) * nx] = s.encode().translate(tr)
        self.walk = walk
        # raw blocked (undilated) as bytes too, for cover search
        blk = bytearray(n)
        tr2 = bytes.maketrans(b"01", b"\x00\x01")
        for j in range(ny):
            s = format(rows[j], "0%db" % nx)[::-1]
            blk[j * nx:(j + 1) * nx] = s.encode().translate(tr2)
        self.blocked = blk

        # coarse grid
        self.cs = cs = 4
        self.cnx = cnx = nx // cs
        self.cny = cny = ny // cs
        cfree = bytearray(cnx * cny)
        for cj in range(cny):
            fj = cj * cs + cs // 2
            for ci in range(cnx):
                fi = ci * cs + cs // 2
                if walk[fj * nx + fi]:
                    cfree[cj * cnx + ci] = 1
        self.cfree = cfree
        nbr = bytearray(cnx * cny)
        for cj in range(cny):
            for ci in range(cnx):
                c = cj * cnx + ci
                if not cfree[c]:
                    continue
                m = 0
                fi0 = ci * cs + cs // 2
                fj0 = cj * cs + cs // 2
                for d, (dx, dy) in enumerate(DIRS8):
                    ni, nj = ci + dx, cj + dy
                    if not (0 <= ni < cnx and 0 <= nj < cny) or not cfree[nj * cnx + ni]:
                        continue
                    ok = True
                    for s in range(1, cs):
                        if not walk[(fj0 + dy * s) * nx + fi0 + dx * s]:
                            ok = False
                            break
                    if ok:
                        m |= 1 << d
                nbr[c] = m
        self.nbr = nbr
        # connected components of the coarse walk graph
        comp = array.array("i", [-1]) * (cnx * cny)
        label = 0
        for c0 in range(cnx * cny):
            if comp[c0] != -1 or not nbr[c0]:
                continue
            comp[c0] = label
            stack = [c0]
            while stack:
                c = stack.pop()
                m = nbr[c]
                for d in range(8):
                    if (m >> d) & 1:
                        dx, dy = DIRS8[d]
                        nc = c + dy * cnx + dx
                        if comp[nc] == -1:
                            comp[nc] = label
                            stack.append(nc)
            label += 1
        self.comp = comp
        # optional precomputed tactical map (directional cover bits), see build_tmap.py
        self.tmap = None
        try:
            tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmap.bin")
            if os.path.exists(tp):
                tm = array.array("H")
                with open(tp, "rb") as f:
                    tm.frombytes(f.read())
                if sys.byteorder != "little":
                    tm.byteswap()
                if len(tm) == cnx * cny * 3:
                    self.tmap = tm
        except Exception:
            self.tmap = None
        self.load_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ queries
    def ground_at(self, x: float, y: float) -> float:
        fx = x - self.minx
        fy = y - self.miny
        i = int(fx)
        j = int(fy)
        if i < 0 or j < 0 or i >= self.nx - 1 or j >= self.ny - 1:
            i = min(max(i, 0), self.nx - 1)
            j = min(max(j, 0), self.ny - 1)
            return self.ground[j * self.nx + i] * self.scale
        tx, ty = fx - i, fy - j
        g = self.ground
        k = j * self.nx + i
        a = g[k] * (1 - tx) + g[k + 1] * tx
        b = g[k + self.nx] * (1 - tx) + g[k + self.nx + 1] * tx
        return (a * (1 - ty) + b * ty) * self.scale

    def fidx(self, x: float, y: float) -> int:
        i = int(round(x - self.minx))
        j = int(round(y - self.miny))
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return -1
        return j * self.nx + i

    def walkable(self, x: float, y: float) -> bool:
        k = self.fidx(x, y)
        return k >= 0 and self.walk[k] == 1

    def seg_walkable(self, x0, y0, x1, y1, step=1.0) -> bool:
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        n = int(L / step) + 1
        walk = self.walk
        nx = self.nx
        minx, miny = self.minx, self.miny
        for s in range(1, n + 1):
            t = s / n
            i = int(round(x0 + dx * t - minx))
            j = int(round(y0 + dy * t - miny))
            if i < 0 or j < 0 or i >= nx or j >= self.ny or not walk[j * nx + i]:
                return False
        return True

    def los(self, a, b, step=1.5) -> bool:
        """Straight line a->b (3D) clear of terrain and prisms (shots; trees ignored)."""
        x0, y0, z0 = a
        dx, dy, dz = b[0] - x0, b[1] - y0, b[2] - z0
        L = math.hypot(dx, dy)
        n = int(L / step) + 1
        nx, ny = self.nx, self.ny
        minx, miny = self.minx, self.miny
        top, base, g, sc = self.top, self.base, self.ground, self.scale
        for s in range(1, n):
            t = s / n
            i = int(round(x0 + dx * t - minx))
            j = int(round(y0 + dy * t - miny))
            if i < 0 or j < 0 or i >= nx or j >= ny:
                return False
            k = j * nx + i
            z = z0 + dz * t
            if g[k] * sc > z + 0.3:
                return False
            if base[k] <= z <= top[k]:
                return False
        return True

    # tactical map
    def covered_from(self, c: int, ex: float, ey: float) -> bool:
        """True if a mech in coarse cell c cannot be shot from (ex, ey) according to the tactical map."""
        tm = self.tmap
        if tm is None:
            return False
        x, y = self.ccenter(c)
        dx, dy = ex - x, ey - y
        d = math.hypot(dx, dy)
        band = 0 if d < 40 else (1 if d < 90 else 2)
        a = math.atan2(dy, dx) % (2 * math.pi)
        f = a / (math.pi / 8)
        k0 = int(f) % 16
        k1 = (k0 + 1) % 16
        m = tm[c * 3 + band]
        return bool((m >> k0) & 1) and bool((m >> k1) & 1)

    def exposure(self, c: int) -> int:
        """How many of the 32 mid/long-range directions can shoot into cell c."""
        tm = self.tmap
        if tm is None:
            return 16
        e = 0
        for band in (1, 2):
            e += 16 - bin(tm[c * 3 + band]).count("1")
        return e

    def reachable_cells(self, x: float, y: float, max_steps: int = 13):
        """BFS over the coarse walk graph: {cell: path length in metres}."""
        s = self.nearest_free_c(x, y)
        if s < 0:
            return {}
        cnx, cs = self.cnx, self.cs
        dist = {s: 0.0}
        frontier = [s]
        for _ in range(max_steps):
            nxt = []
            for c in frontier:
                m = self.nbr[c]
                for d in range(8):
                    if (m >> d) & 1:
                        dx, dy = DIRS8[d]
                        nc = c + dy * cnx + dx
                        nd = dist[c] + (cs * SQ2 if dx and dy else cs)
                        if nc not in dist or nd < dist[nc]:
                            if nc not in dist:
                                nxt.append(nc)
                            dist[nc] = nd
            frontier = nxt
        return dist

    # coarse helpers
    def cidx(self, x: float, y: float) -> int:
        ci = int((x - self.minx) // self.cs)
        cj = int((y - self.miny) // self.cs)
        if ci < 0 or cj < 0 or ci >= self.cnx or cj >= self.cny:
            return -1
        return cj * self.cnx + ci

    def ccenter(self, c: int):
        ci, cj = c % self.cnx, c // self.cnx
        return (self.minx + ci * self.cs + self.cs // 2, self.miny + cj * self.cs + self.cs // 2)

    def comp_at(self, x: float, y: float) -> int:
        c = self.nearest_free_c(x, y, 4)
        return self.comp[c] if c >= 0 else -1

    def nearest_free_c(self, x: float, y: float, maxr: int = 8) -> int:
        c = self.cidx(x, y)
        if c >= 0 and self.nbr[c]:
            return c
        ci = int((x - self.minx) // self.cs)
        cj = int((y - self.miny) // self.cs)
        best, bd = -1, 1e18
        for r in range(1, maxr + 1):
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < self.cnx and 0 <= nj < self.cny:
                        k = nj * self.cnx + ni
                        if self.nbr[k]:
                            cx, cy = self.ccenter(k)
                            d = (cx - x) ** 2 + (cy - y) ** 2
                            if d < bd:
                                bd, best = d, k
            if best >= 0:
                return best
        return -1


class Planner:
    """One A* job at a time on the coarse grid, resumable across ticks.

    goal: (gx, gy, gr) -> reach any free cell within gr metres of (gx, gy).
    """

    def __init__(self, world: World):
        self.w = world
        self.job = None
        self.path = None  # list of (x, y) world points
        self.goal = None
        self.done_t = -1e9
        self.failed = False
        self.risk = None  # optional callable risk(cell, g) -> extra cost per metre (exposure, zone, chokes)

    def request(self, sx, sy, goal, now):
        w = self.w
        s = w.nearest_free_c(sx, sy)
        if s < 0:
            self.job = None
            self.failed = True
            return
        gx, gy, gr = goal
        gr = max(gr, w.cs * 0.75)  # coarse centres are cs apart: never ask for less
        goal = (gx, gy, gr)
        n = w.cnx * w.cny
        g = {s: 0.0}
        came = {s: -1}
        h0 = max(0.0, math.hypot(*(a - b for a, b in zip(w.ccenter(s), (gx, gy)))) - gr)
        self.job = {"open": [(h0, 0.0, s)], "g": g, "came": came, "goal": goal, "best": (h0, s), "exp": 0,
                    "risk": self.risk}
        self.goal = goal
        self.failed = False
        _ = n

    def busy(self):
        return self.job is not None

    def run(self, deadline: float, now: float):
        job = self.job
        if job is None:
            return
        w = self.w
        cnx = w.cnx
        cs = w.cs
        nbr = w.nbr
        minx, miny = w.minx + cs // 2, w.miny + cs // 2
        gx, gy, gr = job["goal"]
        gr2 = gr * gr
        openh = job["open"]
        g = job["g"]
        came = job["came"]
        best = job["best"]
        risk = job.get("risk")
        steps = 0
        pc = time.perf_counter
        found = -1
        while openh:
            f, gc, c = heapq.heappop(openh)
            if gc > g.get(c, 1e18) + 1e-9:
                continue
            ci, cj = c % cnx, c // cnx
            x, y = minx + ci * cs, miny + cj * cs
            dxg, dyg = x - gx, y - gy
            if dxg * dxg + dyg * dyg <= gr2:
                found = c
                break
            m = nbr[c]
            for d in range(8):
                if not (m >> d) & 1:
                    continue
                dx, dy = DIRS8[d]
                nc = c + dy * cnx + dx
                st = cs * SQ2 if dx and dy else cs
                ng = gc + st
                if risk is not None:
                    ng += st * risk(nc, gc)
                if ng < g.get(nc, 1e18):
                    g[nc] = ng
                    came[nc] = c
                    nx_, ny_ = x + dx * cs, y + dy * cs
                    h = math.hypot(nx_ - gx, ny_ - gy) - gr
                    if h < 0:
                        h = 0.0
                    if h < best[0]:
                        best = (h, nc)
                    heapq.heappush(openh, (ng + h, ng, nc))
            steps += 1
            if steps & 63 == 0 and pc() > deadline:
                job["best"] = best
                job["exp"] += steps
                if job["exp"] > 12000:
                    found = best[1]
                    break
                return
        if found < 0:
            found = best[1]  # unreachable: go as close as possible
        pts = []
        c = found
        while c != -1:
            pts.append(w.ccenter(c))
            c = came[c]
        pts.reverse()
        self.path = pts
        self.job = None
        self.done_t = now
