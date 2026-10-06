"""F1 Flight Data Bus conditioning + F3 Sense-Check (F1 plan §4.1).

Resample/align to 1 Hz sim grid, gap-fill (small gaps only, marked imputed),
median-filter impulse noise, robust per-channel scaling (running median/IQR),
data-pathology detection: stale (frozen sensor), saturated (at-rail), delayed
arrival, dropout bursts. Outputs CleanFrame vectors + QualityFlags that the
detector MUST honour (veto channels).
"""
from __future__ import annotations
from collections import deque
import numpy as np

from ..contracts.models import TelemetryFrame, QualityFlags

GAP_FILL_MAX = 6          # longer gaps stay NaN and are vetoed
STALE_RUN = 12            # identical values for N ticks → stale
SAT_RUN = 8


class RobustScaler:
    def __init__(self, channels, warm=60, clip=10.0):
        self.channels = list(channels)
        self.med = {}; self.iqr = {}; self.buf = {c: deque(maxlen=400) for c in self.channels}
        self.warm, self.clip = warm, clip

    def update(self, ch: str, x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        self.buf[ch].append(x)
        if ch not in self.med or len(self.buf[ch]) >= self.warm:
            a = np.asarray(self.buf[ch]); m = float(np.median(a))
            r = float(np.subtract(*np.percentile(a, [75, 25]))) + 1e-6
            r = max(r, 0.05 * float(np.std(a)))      # const-in-warmup channels: avoid 1e-6 rails
            if ch in self.med:
                self.med[ch] = 0.9 * self.med[ch] + 0.1 * m
                self.iqr[ch] = max(1e-4, 0.9 * self.iqr[ch] + 0.1 * r)
            else:
                self.med[ch], self.iqr[ch] = m, r
        return float(np.clip((x - self.med[ch]) / self.iqr[ch], -self.clip, self.clip))

    def state(self, ch, x):
        return self.med.get(ch, 0.0), self.iqr.get(ch, 1.0)


class Conditioner:
    def __init__(self, channels=None, ctx_period=0):
        from ..contracts.models import ALL_CHANNELS
        self.channels = list(channels) if channels else ALL_CHANNELS
        self.ctx = None          # (period, nch) per-orbit-phase medians
        self.ctx_period = ctx_period
        self.scaler = RobustScaler(self.channels)
        self.prev: dict[str, float] = {}
        self.raw_prev: dict[str, float] = {}
        self.stale_run = {c: 0 for c in self.channels}
        self.sat_run = {c: 0 for c in self.channels}
        self.medbuf = {c: deque(maxlen=5) for c in self.channels}
        self.buf300 = {c: deque(maxlen=300) for c in self.channels}
        self.imp_score = {c: 0.0 for c in self.channels}
        self.k = 0

    def step(self, fr: TelemetryFrame) -> tuple[np.ndarray, QualityFlags]:
        """Return (scaled vector over ALL_CHANNELS, quality flags)."""
        self.k += 1
        qf = QualityFlags(t=fr.t)
        vec = np.zeros(len(self.channels))
        for i, ch in enumerate(self.channels):
            x = fr.values.get(ch, float("nan"))
            fl = fr.flags.get(ch, {}) or {}
            if isinstance(fl, dict) and fl.get("delay_s"):
                qf.delay_s = max(qf.delay_s, float(fl["delay_s"]))
            if getattr(fr, "t_arr", -1.0) > fr.t:          # real late arrival (bus)
                qf.delay_s = max(qf.delay_s, fr.t_arr - fr.t)
            if isinstance(fl, dict) and fl.get("late_beyond_window"):
                qf.late_beyond.append(ch)
            if not np.isfinite(x):
                # gap-fill small gaps; veto longer ones
                if self.prev.get(ch) is not None:
                    x = self.prev[ch]; qf.imputed.append(ch); qf.veto_channels.append(ch)
                else:
                    x = 0.0; qf.veto_channels.append(ch)
            # impulse noise: median filter on raw scale (keep x, denoise only for ML)
            mb = self.medbuf[ch]; mb.append(x)
            if len(mb) == 5:
                m5 = float(np.median(list(mb)))
                med, iqr = self.scaler.state(ch, x)
                if abs(x - m5) > 6.0 * (iqr / 2 + 1e-6):
                    x = m5
                    self.imp_score[ch] = self.imp_score[ch] * 0.985 + 1.0
                    qf.veto_channels.append(ch)
                else:
                    self.imp_score[ch] *= 0.985
            if self.imp_score[ch] >= 2.0 and ch not in qf.impulse:
                qf.impulse.append(ch)
                qf.veto_channels.append(ch)
            # stale & saturated run-length trackers (against raw physics values)
            p = self.raw_prev.get(ch)
            if p is not None and abs(x - p) < 1e-12:
                self.stale_run[ch] += 1
            else:
                self.stale_run[ch] = 0
            b3 = self.buf300[ch]; b3.append(x)
            rail_named = (ch == "cpu_load" and x >= 99.5) or (ch in ("heater_current", "wheel_current", "solar_current") and x <= 1e-9)
            rail_generic = False
            if len(b3) >= 120:
                mx = max(b3); mn = min(b3)
                if (mx - mn) > 0:
                    at_ext = (abs(x - mx) <= 1e-9 * max(1.0, abs(mx))) or (abs(x - mn) <= 1e-9 * max(1.0, abs(mn)))
                    if at_ext:
                        cnt = sum(1 for v in b3 if abs(v - x) <= 1e-9 * max(1.0, abs(x)))
                        rail_generic = cnt >= 10
            self.sat_run[ch] = self.sat_run[ch] + 1 if (rail_named or rail_generic) else 0
            if isinstance(fl, dict) and fl.get("stale"): qf.stale.append(ch)
            if self.stale_run[ch] >= STALE_RUN: qf.stale.append(ch); qf.veto_channels.append(ch)
            if self.sat_run[ch] >= SAT_RUN or (isinstance(fl, dict) and fl.get("saturated")):
                qf.saturated.append(ch); qf.veto_channels.append(ch)
            self.raw_prev[ch] = x
            self.prev[ch] = x
            if self.ctx is not None:
                x = x - float(self.ctx[self.k % self.ctx_period, i])
            vec[i] = self.scaler.update(ch, x)
        qf.veto_channels = sorted(set(qf.veto_channels))
        return vec, qf

    def set_context(self, raw_frames: np.ndarray, period: int):
        """Per-orbit-phase median baseline (removes scheduled transitions —
        mode changes are context, not anomalies; plan §4.1)."""
        if period <= 0 or len(raw_frames) < 2 * period:
            return
        tbl = np.zeros((period, raw_frames.shape[1]), np.float64)
        ph = np.arange(len(raw_frames)) % period
        for p in range(period):
            sel = raw_frames[ph == p]
            if len(sel) >= 2:
                tbl[p] = np.nanmedian(sel, axis=0)
        ok = np.isfinite(tbl).all(axis=1)
        if ok.mean() > 0.6:
            tbl[~ok] = np.nanmedian(tbl[ok], axis=0)
            self.ctx, self.ctx_period = tbl, period