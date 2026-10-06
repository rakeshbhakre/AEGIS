"""F4 correlation engine — lagged cross-correlation during event vs before,
merged with FMECA prior edges. Deterministic, auditable (plan §4.3)."""
from __future__ import annotations
import numpy as np

from ..contracts.models import CausalLink

MAX_LAG = 24


def _n(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def correlate(scaled_series: np.ndarray, channels: list, t0: int, t1: int,
              all_names: list, prior_edges: dict, max_links: int = 8) -> list:
    """scaled_series: (T, nch) full history. t0/t1 frame indices of the event."""
    if t1 - t0 < 6 or t0 < 130:
        return []
    ev = scaled_series[max(t0, t1 - 120):t1]
    base = scaled_series[t0 - 120:t0 - 4]
    ev_n, base_n = _n(ev), _n(base)
    idx = {c: i for i, c in enumerate(all_names)}
    hot = [c for c in channels if c in idx]
    links: list[CausalLink] = []
    for a in hot:
        ia = idx[a]
        for j, b in enumerate(all_names):
            if j <= ia:
                continue
            # correlation over lags 0..MAX_LAG (a leads b at lag>0)
            best = (0.0, 0)
            for lag in range(MAX_LAG + 1):
                n_ = len(ev_n) - lag
                if n_ < 12:
                    break
                c_ev = float((ev_n[:n_, ia] * ev_n[lag:lag + n_, j]).sum() / n_)
                c_bs = float((base_n[:, ia] * base_n[:, j]).mean()) if len(base_n) > 12 else 0.0
                d = abs(c_ev) - abs(c_bs)
                if d > best[0]:
                    best = (d, lag)
            delta, lag = best
            prior = prior_edges.get(tuple(sorted((a, b))), 0.0)
            strength = max(0.0, delta) + 0.45 * prior
            if delta > 0.14 or prior > 0:
                links.append(CausalLink(src=a if lag == 0 else a,
                                        dst=b, lag_frames=int(lag),
                                        strength=round(min(1.0, abs(strength)), 3),
                                        prior=float(prior)))
    links.sort(key=lambda l: -l.strength)
    return links[:max_links]
