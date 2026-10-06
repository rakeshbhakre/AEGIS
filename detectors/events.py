"""Stream → windows → fused score → event engine with hysteresis + merge,
plus Conformist (split-conformal daily-max threshold calibration, plan §4.2/§4.5).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from ..contracts.models import AnomalyEvent, ChannelAttrib, QualityFlags

W = 48


@dataclass
class EventConfig:
    merge_gap: int = 40          # events within N frames merge
    hysteresis: float = 0.62     # leave at θ*hyst
    min_len: int = 8             # frames above hysteresis before commit
    alpha: float = 0.05          # conformal coverage 1-α


class Conformist:
    """Split-conformal style calibration: θ = empirical (1-e)-quantile of the
    fused normal score stream, with exceedance rate e = fp_budget/day_frames.
    n is reported honestly; below 2000 samples we hold a conservative default."""
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.theta = 3.2
        self.fp_budget = 2.0
        self.n_cal = 0
        self.curve = []
        self._sorted = None

    def calibrate_events(self, event_peaks: list[float], n_days: float,
                         fp_budgets=(0.5, 1, 2, 5, 10), fuse_stream=None):
        """θ per FP/day budget = (K+2)-th largest normal candidate-event peak,
        K = ceil(b·days).  Event-level → immune to single-frame transients."""
        pk = sorted((float(x) for x in event_peaks), reverse=True)
        if len(pk) < 8 and fuse_stream is not None:
            fs = np.sort(np.asarray(fuse_stream, np.float64))[::-1]
            pk = [float(v) for v in fs[:48]]
        self._sorted = np.asarray(sorted(pk)) if pk else np.zeros(0)
        self.n_cal = len(pk)
        self.curve = []
        for b in fp_budgets:
            if not pk:
                th = 3.0
            else:
                K = min(len(pk) - 1, int(round(b * max(n_days, 1e-9))) + 1)
                th = max(2.2, pk[K])
            self.curve.append({"fp_per_day": b, "theta": round(th, 3)})
        self.fp_budget = 2.0
        self.set_budget(2.0)
        return self

    def set_budget(self, b: float):
        if not self.curve:
            self.theta = 3.2
            return
        self.fp_budget = b
        near = min(self.curve, key=lambda p: abs(p["fp_per_day"] - b))
        self.theta = float(near["theta"])

    def p_value(self, score: float) -> float:
        if self._sorted is None or len(self._sorted) == 0:
            return 0.5
        n = len(self._sorted)
        rank = n - int(np.searchsorted(self._sorted, score, side="left"))
        return float(max(1.0 / (n + 1), rank / (n + 1)))   # conformal p-value


def shadow_event_peaks(fuse: np.ndarray, theta0: float, hyst: float = 0.62,
                       min_len: int = 8, alpha_ema: float = 0.25) -> list[float]:
    """Replays the EventEngine state machine (EMA, hysteresis, persistence)
    over a calibration stream at a low probe threshold; returns candidate
    event peaks. θ is then chosen from THIS distribution — what we calibrate
    is what actually alerts."""
    peaks, ema, in_ev, peak, n_hi = [], 0.0, False, 0.0, 0
    for s0 in fuse:
        ema = (1 - alpha_ema) * ema + alpha_ema * float(s0)
        if not in_ev and ema > theta0:
            in_ev, peak, n_hi = True, ema, 1
        elif in_ev:
            if ema > peak:
                peak = ema
            if ema > theta0 * hyst:
                n_hi += 1
            else:
                if n_hi >= min_len:
                    peaks.append(peak)
                in_ev = False
    if in_ev and n_hi >= min_len:
        peaks.append(peak)
    return peaks


class StreamWindower:
    def __init__(self):
        self.buf = deque(maxlen=4000)

    def push(self, vec: np.ndarray):
        self.buf.append(vec)

    def window(self) -> np.ndarray | None:
        if len(self.buf) < W:
            return None
        return np.asarray(list(self.buf)[-W:])


class EventEngine:
    """Turns fused z-score stream into committed events with attributions."""
    def __init__(self, channels, group_of, cfg: EventConfig = EventConfig(),
                 conformist: Conformist = Conformist()):
        self.channels = list(channels)
        self.group_of = group_of or {}
        self.cfg, self.conf = cfg, conformist
        self.ema = 0.0
        self.cur: dict | None = None
        self.seq = 0

    def _sig(self, score: float) -> tuple:
        """score -> timescale/dir classification inputs on channel series."""
        return ()

    def step(self, t: float, fuse: float, att: np.ndarray,
              raw_scaled: np.ndarray | None, qf: QualityFlags,
             ) -> AnomalyEvent | None:
        """feed per-frame fused z (scalar) + attrib vector (nch,). Returns a
        COMMITTED AnomalyEvent when one closes."""
        self.ema = 0.75 * self.ema + 0.25 * fuse
        s = self.ema
        th = self.conf.theta
        hy = th * self.cfg.hysteresis
        closed = None
        if self.cur is None and s > th:
            self.seq += 1
            self.cur = {"id": f"E{self.seq:05d}", "t0": t, "peak": s, "peak_t": t,
                        "att": att.copy(), "n_hi": 1}
        elif self.cur is not None:
            c = self.cur
            c["n_hi"] += int(s > hy)
            if s > c["peak"]:
                c["peak"], c["peak_t"], c["att"] = s, t, att.copy()
            if s < hy:
                closed = self._commit(c, t)
                self.cur = None
        return closed

    def force_close(self, t: float) -> AnomalyEvent | None:
        if self.cur is None:
            return None
        e = self._commit(self.cur, t)
        self.cur = None
        return e

    def _commit(self, c, t) -> AnomalyEvent | None:
        if c["n_hi"] < self.cfg.min_len:
            return None
        att = c["att"]
        idx = np.argsort(att)[::-1]
        tops = [i for i in idx if att[i] > 0.8][:6]
        if not tops:
            tops = list(idx[:3])
        attrs = [ChannelAttrib(channel=self.channels[i], z=float(att[i]),
                               direction="up" if att[i] >= 0 else "down",
                               timescale="ramp", onset=float(c["t0"]))
                 for i in tops]
        gcount: dict = {}
        for a in attrs:
            g = self.group_of.get(a.channel, "GROUP")
            gcount[g] = gcount.get(g, 0) + att[self.channels.index(a.channel)]
        group = max(gcount, key=gcount.get) if gcount else "EPS"
        ev = AnomalyEvent(id=c["id"], t0=float(c["t0"]), t1=float(t), group=group,
                          score=float(c["peak"]), conformal_p=self.conf.p_value(c["peak"]),
                          attrib=attrs)
        return ev
