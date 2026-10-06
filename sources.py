"""Data sources: live twin stream, file replay (SMAP/ESA-style CSV/NPZ),
each yielding TelemetryFrame at sim rate. Fault injection is a twin feature;
file sources support synthetic fault overlay for demos if requested."""
from __future__ import annotations
import math
import numpy as np

from .contracts.models import TelemetryFrame, GROUP_OF, ALL_CHANNELS
from .twin.spacecraft import FaultInjector, SpacecraftTwin


class BusModel:
    """Ground-station link layer (organiser §29/§30): frames may ARRIVE late
    (up to delay_max, event time preserved on frame.t) and out of arrival
    order. The console reorders by EVENT time within a bounded window —
    frames later than the window are dropped-and-counted, never mis-stamped.
    This makes 'delayed telemetry' a tested data path, not a slogan."""

    def __init__(self, seed=11, reorder_window_s: float = 30.0):
        import random as _r
        self.rng = _r.Random(seed)
        self.delay_prob = 0.0
        self.delay_max_s = 0.0
        self.swap_prob = 0.0
        self.window = float(reorder_window_s)
        self.pending: dict[int, tuple[float, TelemetryFrame]] = {}   # t → (arrival_t, frame)
        self.delivered_t = -1.0
        self.late_drops = 0
        self.delayed_in = 0
        self.swapped = 0

    def set(self, delay_prob=None, delay_max_s=None, swap_prob=None):
        if delay_prob is not None:  self.delay_prob = min(1.0, max(0.0, float(delay_prob)))
        if delay_max_s is not None: self.delay_max_s = max(0.0, float(delay_max_s))
        if swap_prob is not None:   self.swap_prob = min(1.0, max(0.0, float(swap_prob)))
        return self.state()

    @property
    def active(self):
        return self.delay_prob > 0 or self.swap_prob > 0

    def state(self):
        return {"delay_prob": self.delay_prob, "delay_max_s": self.delay_max_s,
                "swap_prob": self.swap_prob, "reorder_window_s": self.window,
                "late_drops": self.late_drops, "delayed_in": self.delayed_in,
                "swapped": self.swapped, "active": (self.delay_prob > 0 or self.swap_prob > 0)}

    def offer(self, fr: TelemetryFrame) -> list[TelemetryFrame]:
        """Put one event-time frame on the bus; returns frames now deliverable
        (sorted by event time; monotonic — physics never sees time go back)."""
        t = fr.t
        if not (self.delay_prob > 0 or self.swap_prob > 0):
            out = [fr]
            self.pending = {}
            self.delivered_t = t
            return out
        delay = (self.rng.uniform(0.5, self.delay_max_s)
                 if self.rng.random() < self.delay_prob and self.delay_max_s > 0 else 0.0)
        if delay > 0:
            fr.t_arr = t + delay
            self.delayed_in += 1
        else:
            fr.t_arr = t
        if self.swap_prob > 0 and self.rng.random() < self.swap_prob:
            fr.t_arr += self.rng.uniform(1.0, 6.0)   # arrive after a YOUNGER frame
            self.swapped += 1
        arr = fr.t_arr
        self.pending[int(t)] = (arr, fr)
        # drop-and-count anything older than the reorder window vs current arrival horizon
        horizon = max(a for a, _ in self.pending.values())
        cutoff = horizon - self.window
        stale = [k for k, (a, f_) in self.pending.items() if f_.t <= cutoff and f_.t > self.delivered_t]
        for k in stale:
            # deliver late frames with a quality marker — never pretend they weren't there
            f_ = self.pending.pop(k)[1]
            f_.flags.setdefault("late_beyond_window", True)
        now_ready = [f_ for (a, f_) in self.pending.values() if a <= horizon]
        ready = sorted(now_ready, key=lambda f_: f_.t)
        out = []
        for f_ in ready:
            if f_.t <= self.delivered_t:
                continue
            if f_.flags.get("late_beyond_window") and f_.t < self.delivered_t - 0.5:
                self.late_drops += 1
                self.pending.pop(int(f_.t), None)
                continue
            gap_to_truth = horizon - f_.t
            if f_.t < cutoff:
                self.late_drops += 1
                self.pending.pop(int(f_.t), None)
                continue
            self.pending.pop(int(f_.t), None)
            self.delivered_t = f_.t
            out.append(f_)
        return out


class TwinSource:
    channels = ALL_CHANNELS
    group_of = GROUP_OF
    kind = "twin"

    def __init__(self, seed=7):
        self.twin = SpacecraftTwin(seed)
        self.inj = FaultInjector(self.twin)
        self.bus = BusModel(seed=seed + 1)

    def frame(self, k: int) -> TelemetryFrame:
        return self.inj.frame(k)

    def frames(self, k: int) -> list[TelemetryFrame]:
        """bus-aware: returns 0..n frames deliverable at tick k (event-time sorted)."""
        return self.bus.offer(self.inj.frame(k))

    def set_quality(self, **kw):
        out = {}
        if "noise" in kw or "missing" in kw:
            out["link"] = self.inj.set_quality(noise=kw.get("noise"), missing=kw.get("missing"))
        if any(x in kw for x in ("delay_prob", "delay_max_s", "swap_prob")):
            out["bus"] = self.bus.set(kw.get("delay_prob"), kw.get("delay_max_s"), kw.get("swap_prob"))
        return out

    def inject(self, fid, dur=420, at=0, **p):
        return self.inj.inject(fid, at, dur, **p)

    def inject_data(self, ch, kind, dur=120, at=0):
        return self.inj.inject_data(ch, kind, at, dur)

    def storm(self, fids, at=0):
        return self.inj.storm(at, fids)


class FileSource:
    """Replays a (T, C) matrix; labels (per-channel anomaly intervals) optional.
    Channel naming: ch0..chC-1 grouped as 'SET1'. Works for SMAP npz or CSVs."""
    kind = "file"

    def __init__(self, data: np.ndarray, channels=None, labels=None,
                 inject=None, dt=4.0):
        self.data = np.asarray(data, np.float32)
        self.T, self.C = self.data.shape
        self.channels = channels or [f"ch{i}" for i in range(self.C)]
        self.group_of = {c: "TELEMETRY" for c in self.channels}
        self.labels = labels or {}          # {chan: [(s,e),...]}
        self.inject = inject or []          # list of (fault_spec, start, dur)
        self.dt = dt
        self.k = -1

    def frame(self, k=None) -> TelemetryFrame | None:
        self.k += 1
        k = self.k
        if k >= self.T:
            return None
        row = self.data[k]
        vals = {c: float(row[i]) if not np.isnan(row[i]) else float("nan")
                for i, c in enumerate(self.channels)}
        flags = {}
        for fid, start, dur in self.inject:
            if start <= k < start + dur:
                vals.update(apply_overlay(fid, row, k - start, self.channels))
                flags["_truth"] = {"fid": fid, "start": start}
        return TelemetryFrame(t=float(k) * self.dt, values=vals, flags=flags)


OVERLAYS = {
    "heater_stall":   {"heater_current": -0.9, "battery_temp": -0.06, "battery_voltage": -0.02},
    "cell_sag":       {"battery_voltage": -0.02, "soc": -0.12, "battery_current": 0.03},
    "wheel_friction": {"wheel_current": 0.09, "wheel_temp": 0.6, "wheel_speed": -6.0},
    "sensor_bias":    {"solar_current": 0.5},
    "thermal_runaway": {"battery_temp": 0.22},
    "gap": {"__veto__": True},
}


def apply_overlay(fid, row, rel, channels):
    spec = OVERLAYS.get(fid, {})
    out = {}
    ramp = min(1.0, rel / 40.0)
    for ch, g in spec.items():
        if ch == "__veto__":
            continue
        i = channels.index(ch) if ch in channels else None
        if i is not None and not np.isnan(row[i]):
            out[ch] = float(row[i]) + g * ramp
    if spec.get("__veto__"):
        i = 0
        out[channels[i]] = float("nan")
    return out


def load_smap(path="data/SMAP", sats=None, split="test"):
    """Merlion-style SMAP dir: {split}/{SAT}.npy + labeled_anomalies.csv."""
    import os, csv, json
    d = os.path.join(path)
    assert os.path.isdir(d), f"SMAP dir not found: {d}"
    files = sorted(f for f in os.listdir(os.path.join(d, split)) if f.endswith(".npy"))
    if sats:
        files = [f for f in files if f[:-4] in sats]
    lab_path = os.path.join(d, "labeled_anomalies.csv")
    labels_by_sat: dict = {}
    if os.path.exists(lab_path):
        with open(lab_path) as f:
            for row in csv.DictReader(f):
                labels_by_sat.setdefault(row["chan_id"], []).append(
                    (json.loads(row["anomaly_sequences"]), row.get("class")))
    out = []
    for fn in files:
        sid = fn[:-4]
        X = np.load(os.path.join(d, split, fn))
        anns = []
        for seqs, _cls in labels_by_sat.get(sid, []):
            for s, e, *_ in seqs:
                anns.append((int(s), int(e)))
        out.append(FileSource(X, channels=[f"{sid}-v{i}" for i in range(X.shape[1])],
                             labels={"TELEMETRY": anns}, dt=4.0))
    return out
