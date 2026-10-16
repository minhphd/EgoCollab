"""Walking room in a BEHAVIOR home at full resolution: clearance from the scene's 1 cm floor map and a planner that
keeps bodies off furniture.

OmniGibson plans on a 10 cm map made by resizing the 1 cm one, which samples a few pixels per cell, so thin obstacles
(wall lines, furniture edges) can vanish from it and clearance measured on it runs high. Here a 10 cm cell's clearance
is the least over its pixels; A* keeps out of cells under `hard` m and pays for coming nearer than `soft` m, and the
path is straightened wherever a straight line keeps `line` m of clearance.

    fl = Floor(scene_dir / "layout/floor_trav_0.png")
    fl.clearance(xy); fl.plan(a, b)
"""

from __future__ import annotations

import heapq

import numpy as np

FINE, CELL = 0.01, 0.10  # m per pixel of the dataset map, of the planning grid


class Floor:
    def __init__(self, png, hard: float = 0.22, soft: float = 0.45, line: float = 0.30):
        import cv2
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        self.free = (img == 255).astype(np.uint8)
        self.n = self.free.shape[0]
        self.k = int(round(CELL / FINE))
        self.m = self.n // self.k
        self.hard, self.soft, self.line = hard, soft, line
        self._measure()

    def _measure(self) -> None:
        import cv2
        m, k = self.m, self.k
        self.dist = cv2.distanceTransform(self.free, cv2.DIST_L2, 5) * FINE
        self.cell = self.dist[:m * k, :m * k].reshape(m, k, m, k).min(axis=(1, 3))  # the least clearance in each cell
        self.__dict__.pop("_labels", None)

    def block(self, lo, hi) -> None:
        """Take the floor under an (x, y) box (a thing standing on it that the map does not hold) and measure again."""
        r0, c0 = int(lo[1] / FINE + self.n / 2), int(lo[0] / FINE + self.n / 2)
        r1, c1 = int(hi[1] / FINE + self.n / 2) + 1, int(hi[0] / FINE + self.n / 2) + 1
        self.free[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = 0
        self._measure()

    def clearance(self, xy) -> float:
        """Distance (m) from xy to the nearest obstacle (OmniGibson's convention: row from y, column from x)."""
        r, c = int(xy[1] / FINE + self.n / 2), int(xy[0] / FINE + self.n / 2)
        return float(self.dist[r, c]) if 0 <= r < self.n and 0 <= c < self.n else 0.0

    def _rc(self, xy) -> tuple[int, int]:
        return int((xy[1] / FINE + self.n / 2) // self.k), int((xy[0] / FINE + self.n / 2) // self.k)

    def _xy(self, rc) -> np.ndarray:
        return np.array([((rc[1] + 0.5) * self.k - self.n / 2) * FINE, ((rc[0] + 0.5) * self.k - self.n / 2) * FINE])

    def plan(self, a, b, avoid=(), hard: float | None = None) -> np.ndarray | None:
        """Waypoints (N, 2) from a to b, or None when b cannot be reached. Cells near a and b may be tight (an agent
        can start or end close to furniture); elsewhere the hard clearance (or `hard`) holds."""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        s, g = self._rc(a), self._rc(b)
        m, C, hard = self.m, self.cell, self.hard if hard is None else hard
        ok = lambda rc: 0 <= rc[0] < m and 0 <= rc[1] < m and (C[rc] >= hard or max(abs(rc[0] - s[0]), abs(rc[1] - s[1])) <= 3
                                                               or max(abs(rc[0] - g[0]), abs(rc[1] - g[1])) <= 3) and C[rc] > 0.05
        if not (0 <= g[0] < m and 0 <= g[1] < m):
            return None
        if avoid:  # cells within r of a given point (another agent standing in the way) are closed
            C = C.copy()
            rr, cc = np.mgrid[0:m, 0:m]
            for xy, r in avoid:
                q = self._rc(xy)
                C[(rr - q[0]) ** 2 + (cc - q[1]) ** 2 <= (r / CELL) ** 2] = 0.0
        h = lambda rc: CELL * float(np.hypot(rc[0] - g[0], rc[1] - g[1]))
        best, came, todo = {s: 0.0}, {}, [(h(s), 0.0, s)]
        steps = [(dr, dc, CELL * float(np.hypot(dr, dc))) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        while todo:
            _, d, u = heapq.heappop(todo)
            if u == g:
                break
            if d > best.get(u, np.inf):
                continue
            for dr, dc, w in steps:
                v = (u[0] + dr, u[1] + dc)
                if not ok(v):
                    continue
                nd = d + w * (1.0 + 3.0 * max(0.0, self.soft - C[v]) / self.soft)
                if nd < best.get(v, np.inf):
                    best[v], came[v] = nd, u
                    heapq.heappush(todo, (nd + h(v), nd, v))
        if g not in came and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(came[cells[-1]])
        pts = [a] + [self._xy(rc) for rc in reversed(cells[1:-1])] + [b]
        return self._straighten(np.array(pts))

    def _clear_line(self, p, q) -> bool:
        n = max(2, int(np.linalg.norm(q - p) / 0.05))
        return all(self.clearance(p + (q - p) * t) >= self.line for t in np.linspace(0, 1, n)[1:-1])

    def _straighten(self, pts: np.ndarray) -> np.ndarray:
        """Drop waypoints wherever a straight line from the last kept one keeps line m of clearance."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear_line(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def region(self, xy) -> int:
        """Which connected stretch of walkable floor (cells with the hard clearance) xy belongs to: that of the nearest
        walkable cell within 1 m, or -1."""
        import cv2
        if not hasattr(self, "_labels"):
            _, self._labels = cv2.connectedComponents((self.cell >= self.hard).astype(np.uint8), connectivity=8)
        r, c = self._rc(xy)
        for rad in range(0, 11):
            r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, self.m), max(c - rad, 0), min(c + rad + 1, self.m)
            win = self._labels[r0:r1, c0:c1]
            if (win > 0).any():
                rr, cc = np.nonzero(win > 0)
                k = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
                return int(win[rr[k], cc[k]])
        return -1
