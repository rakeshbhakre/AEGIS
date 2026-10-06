"""AEGIS runtime engine — one code path for live console, headless demo and
bench (streaming frame-by-frame). Plan §3: modules composed over contracts.
"""
from __future__ import annotations
import json
import os
import time
import numpy as np

from ..contracts.models import TelemetryFrame, QualityFlags, EvidenceBundle
from ..conditioner.stream import Conditioner
from ..detectors.models import MADDetector, AEDetector, IForestDetector, Ensemble
from ..detectors.events import StreamWindower, EventEngine, Conformist, W
from ..reasoning import graph as graph_mod
from ..reasoning import rca as rca_mod
from ..reasoning.guard import decide as guard_decide, state_from_values
from ..reasoning import ontology

MAX_RETAINED_EVENTS = 200
MAX_LATENCY_SAMPLES = 4000


class Engine:
    def __init__(self, channels, group_of=None, day_frames=3600, train_ae=True,
                 ont=None, labels_path=None, seed=0):
        self.channels = list(channels)
        self.group_of = group_of or {c: "GROUP" for c in channels}
        self.cond = Conditioner(self.channels)
        self.win = StreamWindower()
        self.mad = MADDetector(); self.ae = AEDetector(seed=seed)
        self.ifo = IForestDetector(seed=seed)
        self.ens = Ensemble(self.mad, self.ae, self.ifo)
        self.conf = Conformist()
        self.events_eng = EventEngine(self.channels, self.group_of,
                                      cfg_type_fix(), conformist=self.conf)
        self.ont = ont or ontology.get()
        self.history = np.zeros((0, len(self.channels)), np.float32)  # scaled
        self.raw_hist = np.zeros((0, len(self.channels)), np.float32)
        self.bus_window = 30.0
        self.qf_log: list[set] = []
        self.qf_q: list[float] = []   # per-frame data-quality score 0..1
        self.k = 0
        self.fuse_stream: list[float] = []
        self.events: list[EvidenceBundle] = []
        self.total_events = 0
        self.subscribers = []
        self.feedback_priors: dict = {}
        try:
            fp_path = os.path.join(os.path.dirname(os.path.dirname(labels_path or ".")),
                                   "feedback_priors.json") if labels_path else None
            if fp_path:
                self._fp_path = fp_path
                if os.path.exists(fp_path):
                    self.feedback_priors = json.load(open(fp_path))
        except Exception:
            pass
        self.last_actions: dict = {}
        self.labels_path = labels_path
        self.latency_ms: list[float] = []
        self.day_frames = day_frames
        self.train_ae = train_ae
        self.n_frames_fit = 0
        self._warm_done = False

    # ------------------------------------------------------------------ lifecycle
    def warm(self, frames: list[TelemetryFrame], log=lambda m: None):
        """Warmup on NORMAL telemetry: fits scaler, MAD, IF, AE (stride 4),
        then calibrates conformal thresholds on a stride-1 fused stream."""
        vecs, raws = [], []
        for fr in frames:
            v, qf = self.cond.step(fr)
            vecs.append(v); raws.append([fr.values.get(c, np.nan) for c in self.channels])
            self.qf_log.append(set(qf.veto_channels))
        X = np.asarray(vecs, np.float32)
        self.history = X; self.raw_hist = np.asarray(raws, np.float32)
        self.n_frames_fit = len(X)
        # second pass WITH orbital-context removal
        period = getattr(self, "ctx_period", 3600)
        self.cond.set_context(np.nan_to_num(self.raw_hist.astype(np.float64)), period)
        if self.cond.ctx is None and len(frames) < 2 * period:
            log(f"WARN: warmup {len(frames)} frames < 2×period({period}) — "
                f"orbital context skipped; expect inflated FP budget")
        if self.cond.ctx is not None:
            vecs2 = []
            for fr in frames:
                v, _ = self.cond.step(TelemetryFrame(t=fr.t, values=dict(fr.values)))
                vecs2.append(v)
            X = np.asarray(vecs2, np.float32)
            self.history = X
        Xw4 = self._windows(X, stride=4)
        self.mad.fit(X)
        self.ifo.fit(Xw4)
        per_ch_err = None
        if self.train_ae and len(Xw4) > 60:
            log("training autoencoder (numpy MLP)…")
            self.ae.fit(Xw4, log=log)
            per_ch_err = self.ae.recon_err(Xw4)
        # ---- stride-1 fused stream for calibration.
        # Statistic = MAX over channels of |mean robust-z in window|: a
        # localized sustained fault must dominate (percentiles dilute it).
        Xw1 = self._windows(X, stride=1)
        Z1 = self.mad.z(Xw1)
        mad_stat = np.abs(Z1.mean(axis=1)).max(axis=1)
        S = 600
        slow_stat = _slow_for_windows(self.mad.z(X), W, 1, S)
        if per_ch_err is not None:
            err1 = self.ae.recon_err(Xw1)
            ea, eb = _rob2(err1)
            errn1 = (err1 - ea) / np.where(eb > 1e-9, eb, 1.0)
            self.ens.cal = {"err": (ea, eb)}
            ae_stat = errn1.max(axis=1)
        else:
            self.ens.cal = {"err": (np.zeros(len(self.channels)),
                                    np.ones(len(self.channels)))}
            ae_stat = np.zeros(len(Xw1))
        if_stat = self.ifo.score(Xw1) if self.ifo.m is not None else np.zeros(len(Xw1))
        self.stats_cal = {"mad": _rob(mad_stat), "ae": (0.0, 1.0),
                          "if": _rob(if_stat), "slow": _rob(slow_stat)}
        self.slow_len = S
        fuse = self._fuse(mad_stat, ae_stat, if_stat, slow_stat)
        from ..detectors.events import shadow_event_peaks
        peaks = shadow_event_peaks(fuse, theta0=2.0)
        n_days = max(1e-6, len(fuse) / float(self.day_frames))
        self.conf.calibrate_events(peaks, n_days, fuse_stream=fuse)
        log(f"calibrated θ={self.conf.theta:.2f}  curve={self.conf.curve}")
        self._warm_done = True
        return self

    def _windows(self, X, stride=1):
        if len(X) < W + 1:
            return np.zeros((0, W, X.shape[1]), np.float32)
        xw = np.lib.stride_tricks.sliding_window_view(X, W, axis=0).transpose(0, 2, 1)
        return np.ascontiguousarray(xw[::stride]) if stride > 1 else xw

    def _fuse(self, mad_stat, ae_stat, if_stat, slow_stat=None):
        cm, ci = self.stats_cal["mad"], self.stats_cal["if"]
        # ±40σ saturation: near-degenerate calibration IQRs (constant-during-
        # warmup channels on real telemetry) must not explode the stream.
        zm = np.clip((np.asarray(mad_stat) - cm[0]) / cm[1], -40, 40)
        zi = np.clip((np.asarray(if_stat) - ci[0]) / ci[1], -40, 40)
        if slow_stat is not None and "slow" in self.stats_cal:
            cs = self.stats_cal["slow"]
            zs = np.clip((np.asarray(slow_stat) - cs[0]) / cs[1], -40, 40)
        else:
            zs = np.zeros_like(zm)
        za = np.clip(np.asarray(ae_stat), -40, 40)
        if self.ae.W_:
            return 0.26 * zm + 0.30 * za + 0.16 * zi + 0.28 * zs
        return 0.36 * zm + 0.28 * zi + 0.36 * zs

    # ------------------------------------------------------------------ stream
    def step(self, fr: TelemetryFrame) -> EvidenceBundle | None:
        t0 = time.perf_counter()
        self.k += 1
        v, qf = self.cond.step(fr)
        nch_ = max(1, len(self.channels))
        _qlb = getattr(qf, "late_beyond", [])
        _dpen = (0.15 + 0.25 * min(1.0, qf.delay_s / max(1e-9, self.bus_window))) if qf.delay_s > 0 else 0.0
        qscore = 1.0 - min(1.0, 0.35 * len(qf.veto_channels) / nch_
                           + 0.20 * len(qf.imputed) / nch_
                           + 0.10 * len(qf.stale) / nch_
                           + 0.10 * len(qf.saturated) / nch_
                           + _dpen + 0.30 * (1.0 if _qlb else 0.0))
        self.qf_q.append(qscore)
        self.history = np.vstack([self.history, v[None, :]]) if len(self.history) else v[None, :].astype(np.float32)
        raw = np.asarray([fr.values.get(c, np.nan) for c in self.channels], np.float32)
        self.raw_hist = np.vstack([self.raw_hist, raw[None, :]]) if len(self.raw_hist) else raw[None, :]
        self.qf_log.append(set(qf.veto_channels))
        if self.k % 3600 == 0:  # trim memory (keep last 2 days)
            keep = 2 * self.day_frames
            self.history = self.history[-keep:]; self.raw_hist = self.raw_hist[-keep:]
            self.qf_log = self.qf_log[-keep:]
            self.qf_q = self.qf_q[-keep:]
            self.fuse_stream = self.fuse_stream[-keep:]
            self.latency_ms = self.latency_ms[-MAX_LATENCY_SAMPLES:]
        closed = None
        if self._warm_done:
            w = self.history[-W:]
            if w.shape[0] == W:
                fuse, att, _ = self._score_window(w)
                self.fuse_stream.append(float(fuse))
                ev = self.events_eng.step(fr.t, float(fuse), att, None, qf)
                if ev is not None:
                    closed = self._assemble(ev, qf)
        self.latency_ms.append((time.perf_counter() - t0) * 1000)
        self._broadcast({"type": "frame", "t": fr.t, "values": dict(fr.values),
                         "quality": {"imputed": qf.imputed, "stale": qf.stale,
                                     "saturated": qf.saturated, "delay_s": qf.delay_s},
                         "score": self.fuse_stream[-1] if self.fuse_stream else 0.0,
                         "theta": self.conf.theta})
        if closed:
            self._broadcast({"type": "event", "bundle": closed.to_dict()})
        return closed

    def _score_window(self, w: np.ndarray):
        Wm = w[None, :, :]
        per_ch = self.mad.z(Wm)[0].mean(axis=0)
        mad_stat = float(np.abs(per_ch).max())
        if self.ae.W_ is not None:
            err = self.ae.recon_err(Wm)[0]
            a, b = self.ens.cal["err"]
            errn = (err - a) / np.where(b > 1e-9, b, 1.0)
            att = np.clip(0.5 * per_ch + 0.5 * errn, -40, 40)
            ae_stat = float(np.abs(errn).max())
        else:
            att, ae_stat = per_ch, 0.0
        if_stat = float(self.ifo.score(Wm)[0]) if self.ifo.m is not None else 0.0
        S = getattr(self, "slow_len", 600)
        if len(self.history) >= S:
            tailz = self.mad.z(self.history[-S:])
            slow_stat = float(np.abs(np.asarray(tailz).mean(axis=0)).max())
        else:
            slow_stat = 0.0
        fuse = float(self._fuse(np.array([mad_stat]), np.array([ae_stat]),
                                np.array([if_stat]), np.array([slow_stat]))[0])
        return fuse, att, None

    def _assemble(self, ev, qf) -> EvidenceBundle:
        off = len(self.history) - W
        i0, i1 = int(off), int(off) + 1  # window-based indices into history tail
        base0 = max(0, len(self.history) - 300)
        seg_hist = self.history[base0:]
        # event frames approximate to the last ~150 samples of history
        t1i = len(seg_hist) - 1
        span = max(12, min(int(ev.t1 - ev.t0), 90))
        t0i = max(0, t1i - span)
        att = np.zeros(len(self.channels), np.float32)
        for a in ev.attrib:
            if a.channel in self.channels:
                att[self.channels.index(a.channel)] = a.z
        veto_share, veto_cnt = {}, {}
        if self.qf_log:
            n = len(self.qf_log)
            lo = max(0, n - int(max(0.0, self.k - ev.t0)) - 1)
            hi = max(lo + 8, n - int(max(0.0, self.k - ev.t1)))
            window = self.qf_log[lo:hi] or self.qf_log[-span - 2:]
            qw = self.qf_q[lo:hi] or self.qf_q[-span - 2:]
            ev.quality = round(float(np.mean(qw)) if qw else 1.0, 3)
            _vs = 100.0 * (sum(len(s_) for s_ in window) / max(1, len(window) * len(self.channels))) if window else 0.0
            ev.quality_detail = {"max_delay_s": round(float(getattr(qf, "delay_s", 0.0) or 0.0), 1),
                                 "veto_share_pct": round(_vs, 1)}
            for ch in self.channels:
                c = sum(1 for s in window if ch in s)
                veto_cnt[ch] = c
                veto_share[ch] = c / max(1, len(window))
        sig = rca_mod.extract_signature(self.channels, seg_hist, t0i, t1i, att, veto_share)
        def _is_data(a):
            return veto_share.get(a.channel, 0) > 0.5 or veto_cnt.get(a.channel, 0) >= 4
        # z-weighted veto mass: a pathology channel DOMINATING the evidence
        # (not merely present) reclassifies the event as data-origin.
        tot_w = sum(a.z * a.z for a in ev.attrib) or 1.0
        data_w = sum(a.z * a.z for a in ev.attrib if _is_data(a))
        data_like = sum(1 for a in ev.attrib if _is_data(a))
        # corroboration clause: a HEALTH fault can legitimately pin a channel to
        # its rail (stalled heater → 0 A). If an UNVETOED channel is independently
        # alarming (z ≥ 2.5), the vehicle — not the instrument — is talking.
        corroborated = any(a.z >= 2.5 and not _is_data(a) for a in ev.attrib)
        if not corroborated and ev.attrib and (data_w / tot_w >= 0.5 or
                          data_like >= max(1, int(0.6 * len(ev.attrib)))):
            ev.origin = "DATA"
            ev.veto_reasons = [f"{a.channel}: quality-flagged {veto_share[a.channel]*100:.0f}% of event"
                               for a in ev.attrib if _is_data(a)]
        links = graph_mod.correlate(seg_hist, [a.channel for a in ev.attrib],
                                    t0i, t1i, self.channels, self.ont.prior_edges)
        if getattr(ev, "quality", 1.0) < 0.55:
            ev.veto_reasons = list(ev.veto_reasons) + [
                f"INSUFFICIENT EVIDENCE — data quality {ev.quality*100:.0f}%: conclusions are "
                "provisional, request fresh/complete telemetry"]
        causes = rca_mod.rank(sig, self.ont, self.feedback_priors,
                              data_fault=ev.origin == "DATA")
        if getattr(ev, "quality", 1.0) < 0.85:      # confidence shrinks with worse data
            for cc in causes:
                cc.p = round(cc.p * max(0.35, ev.quality), 3)
        vals = {c: float(self.raw_hist[-1, i]) if not np.isnan(self.raw_hist[-1, i]) else 0.0
                for i, c in enumerate(self.channels)}
        snap = state_from_values(vals, 600.0, self.last_actions)
        acts = guard_decide(ev, causes, snap, self.ont)
        for a in acts:
            if a.allowed and a.tier == "auto_safe":
                self.last_actions[a.action_id] = self.k
        bundle = EvidenceBundle(
            event=ev, links=links, causes=causes, actions=acts,
            provenance={"model": "aegis-v0.1-numpyAE" if self.ae.W_ else "aegis-v0.1-robust",
                        "fit_frames": self.n_frames_fit, "theta": round(self.conf.theta, 2),
                        "fp_budget": self.conf.fp_budget, "t": ev.t1})
        self.events.append(bundle)
        self.total_events += 1
        if len(self.events) > MAX_RETAINED_EVENTS:
            del self.events[:-MAX_RETAINED_EVENTS]
        return bundle

    # ------------------------------------------------------------------ actions
    def set_fp_budget(self, b: float):
        self.conf.set_budget(b)
        return self.conf.theta

    def label(self, event_id: str, verdict: str, fault_id: str | None = None):
        """Operator confirm/override → prior update (flywheel-lite)."""
        if self.labels_path:
            with open(self.labels_path, "a") as f:
                f.write(json.dumps({"event": event_id, "verdict": verdict,
                                    "fault": fault_id, "t": self.k}) + "\n")
        for b in self.events:
            if b.event.id == event_id:
                fid = fault_id or (b.causes[0].fault_id if b.causes else None)
                if fid:
                    cur = float(self.feedback_priors.get(fid, self.ont.priors.get(fid, 0.1)))
                    self.feedback_priors[fid] = cur * (1.30 if verdict == "confirm" else 0.70)
                    self._save_priors()
                return {"ok": True, "updated": fid}
        return {"ok": False}

    def refresh(self):
        """Re-calibrate conformal thresholds on the current normal-only span
        (exclude ±120 frames around events). Fast; AE retrain is the slow
        version run via `make refresh-full`."""
        if len(self.history) < 200:
            return
        ex = np.ones(len(self.history), bool)
        base = len(self.history) - 600
        for b in self.events[-12:]:
            t = int(base + max(0, min(len(self.history) - 1,
                                      (b.event.t1) )))
            ex[max(0, t - 120):t + 60] = False
        X = self.history[ex]
        Xw = np.lib.stride_tricks.sliding_window_view(X, W, axis=0).transpose(0, 2, 1)
        Xw = np.ascontiguousarray(Xw[::4])
        if len(Xw) < 8:
            return
        Z = self.mad.z(Xw)
        mad_stat = np.abs(Z.mean(axis=1)).max(axis=1)
        if self.ae.W_:
            a, b = self.ens.cal["err"]
            errn = (self.ae.recon_err(Xw) - a) / np.where(b > 1e-9, b, 1.0)
            ae_stat = errn.max(axis=1)
        else:
            ae_stat = np.zeros(len(Xw))
        if_stat = self.ifo.score(Xw) if self.ifo.m is not None else np.zeros(len(Xw))
        S = getattr(self, "slow_len", 600)
        slow_w = _slow_for_windows(self.mad.z(X), W, 4, S)
        slow_w = slow_w[:len(Xw)] if len(slow_w) >= len(Xw) else np.resize(slow_w, len(Xw))
        fuse = self._fuse(mad_stat, ae_stat, if_stat, slow_w)
        from ..detectors.events import shadow_event_peaks
        peaks = shadow_event_peaks(fuse, theta0=2.0)
        self.conf.calibrate_events(peaks, n_days=max(0.4, len(fuse) / float(self.day_frames)),
                                   fuse_stream=fuse)

    def _save_priors(self):
        p = getattr(self, "_fp_path", None)
        if p:
            json.dump(self.feedback_priors, open(p, "w"), indent=1)

    def export_edge(self, path):
        """int8-style edge bundle: AE weights + calibrations + θ. The edge
        runtime executes the SAME numpy forward (structural parity)."""
        flat = {"mad_mu": np.asarray(self.mad.mu, np.float32),
                "mad_sd": np.asarray(self.mad.sd, np.float32),
                "mad_valid": np.asarray(getattr(self.mad, "valid", np.ones(len(self.channels))), np.float32),
                "theta": np.float32(self.conf.theta),
                "channels": " ".join(self.channels)}
        for k, (a, b) in self.stats_cal.items():
            flat[f"sc_{k}"] = np.array([float(np.atleast_1d(a).mean()),
                                        float(np.atleast_1d(b).mean())], np.float32)
        if self.ae.W_:
            ws = self.ae.W_["ws"]
            flat["ae_nlayers"] = np.int32(len(ws))
            for i, (W, b) in enumerate(ws):
                flat[f"W{i}"] = np.asarray(W, np.float32)
                flat[f"b{i}"] = np.asarray(b, np.float32)
            a, b = self.ens.cal["err"]
            flat["err_cal"] = np.stack([np.asarray(a, np.float32), np.asarray(b, np.float32)])
        np.savez_compressed(path, **flat)
        return os.path.getsize(path)

    def load_pretrained(self, path):
        """Restore the AE weights + error calibration from an edge bundle
        (data/aegis_edge.npz, written by export_edge). Called before warm() so
        conformal θ calibrates ON the pretrained-AE stream — parity with the
        trained model by construction. Returns what was loaded, or None."""
        import os as _os
        import numpy as _np
        if not _os.path.exists(path):
            return None
        try:
            d = _np.load(path, allow_pickle=False)
            if "channels" not in d or str(d["channels"]) != " ".join(self.channels):
                return None
            if "ae_nlayers" not in d:
                return None
            nl = int(d["ae_nlayers"])
            ws = [(d[f"W{i}"], d[f"b{i}"]) for i in range(nl)]
            self.ae.W_ = {"ws": ws, "val": 0.0}
            if "err_cal" in d:
                a, b = _np.asarray(d["err_cal"], _np.float32)
                self.ens.cal["err"] = (float(_np.mean(a)), float(_np.mean(b)))
            return {"ae_layers": nl, "err_cal": "err_cal" in d}
        except Exception:
            return None

    def stats(self):
        lat = np.asarray(self.latency_ms[-MAX_LATENCY_SAMPLES:])
        return {
            "k": self.k, "theta": round(self.conf.theta, 3),
            "fp_budget": self.conf.fp_budget, "curve": self.conf.curve,
            "events": self.total_events, "score": round(self.fuse_stream[-1], 3) if self.fuse_stream else 0,
            "lat_p50": round(float(np.percentile(lat, 50)), 2) if len(lat) else 0,
            "lat_p99": round(float(np.percentile(lat, 99)), 2) if len(lat) else 0,
            "ae": bool(self.ae.W_), "ae_val": (round(float(self.ae.W_.get("val") or 0.0), 4) if self.ae.W_ else None),
            "history_span": [0, self.k],
        }

    def _broadcast(self, msg):
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass


def _slow_for_windows(Z, W, stride, S):
    """Per-window |max_c mean z over the S frames ending at that window's end|
    — orbit-synchronous (S multiple of period ⇒ periodic swings cancel)."""
    n = (Z.shape[0] - W) // stride + 1
    ends = np.arange(n) * stride + (W - 1)
    starts = np.clip(ends - S + 1, 0, None)
    C = np.cumsum(Z, axis=0, dtype=np.float64)
    C = np.vstack([np.zeros((1, Z.shape[1])), C])
    mean = (C[ends + 1] - C[starts]) / (ends - starts + 1)[:, None]
    return np.abs(mean).max(axis=1).astype(np.float32)


def _rob(v):
    v = np.asarray(v, np.float64)
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25]))) or 1e-6
    return med, max(abs(iqr), 1e-6)


def _rob2(v):
    v = np.asarray(v, np.float64)
    return np.median(v, axis=0), np.subtract(*np.percentile(v, [75, 25], axis=0)) + 1e-9


def cfg_type_fix():
    from ..detectors.events import EventConfig
    return EventConfig()
