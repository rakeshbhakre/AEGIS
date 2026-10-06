"""F5 Root-Cause Engine — signature overlap + Bayesian re-rank (plan §4.4).

Everything is inspectable: the per-fault matched/missed/spurious tables ARE
the explanation. Priors update from operator feedback (the flywheel hook).
"""
from __future__ import annotations
import math
import numpy as np

from ..contracts.models import ALL_CHANNELS, CauseCandidate

BETA = 1.7


def extract_signature(channels, scaled_series: np.ndarray, t0: int, t1: int,
                      attrib_z: np.ndarray, veto_share: dict | None = None,
                      z_min: float = 0.8) -> dict:
    """Per-channel {direction, kind, z} from the event window vs its baseline."""
    out = {}
    for i, ch in enumerate(channels):
        z = float(attrib_z[i])
        if z < z_min:
            continue
        seg = scaled_series[max(t0, t1 - 96):t1, i]
        base = scaled_series[max(0, t0 - 120):max(1, t0 - 2), i]
        if len(seg) < 8:
            continue
        d = float(seg.mean() - base.mean())
        diff = np.diff(seg)
        slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) if len(seg) > 6 else 0.0
        step = abs(float(seg[:max(3, len(seg) // 4)].mean() - seg[max(3, len(seg) // 4):].mean()))
        var = float(diff.std()) if len(diff) > 1 else 1.0
        if (veto_share or {}).get(ch, 0.0) > 0.5 or var < 1e-6:
            kind = "freeze"
        elif step > 1.6 and step > 3.5 * abs(slope) * len(seg):
            kind = "jump"
        elif abs(slope) * len(seg) > 1.2 and abs(np.sign(diff).mean()) > 0.72:
            kind = "ramp"
        elif (np.abs(diff) > 6).sum() <= 3 and (np.abs(diff) > 6).any():
            kind = "spike"
        else:
            kind = "ramp"
        direction = "flat" if abs(d) < 0.25 else ("up" if d > 0 else "down")
        out[ch] = {"direction": direction, "kind": kind, "z": round(z, 2)}
    return out


def rank(sig: dict, ont, feedback_priors: dict | None = None, top_k: int = 3,
         data_fault: bool = False) -> list[CauseCandidate]:
    obs = sig or {}
    scored: list[tuple] = []
    for fid in ont.fault_ids():
        spec = ont.signature(fid)
        score, matched, missed, spurious = 0.0, [], [], []
        exp_chs = set(spec)
        for ch, s in obs.items():
            w = min(1.0, max(0.4, s["z"] / 3.0))
            if ch in exp_chs:
                e = spec[ch]
                d_ok = e["dir"] in (s["direction"], "any")
                k_ok = e["kind"] in (s["kind"], "any")
                if d_ok and k_ok:
                    score += 1.0 * w
                    matched.append(ch)
                elif d_ok or k_ok:
                    score += 0.45 * w
                    matched.append(ch)
                else:
                    score -= 0.35 * w
                    spurious.append(ch)
            else:
                score -= 0.12 * w
                spurious.append(ch)
        for ch in exp_chs:
            if ch not in obs:
                score -= 0.30
                missed.append(ch)
        if data_fault and not ont.is_data_fault(fid):
            score -= 0.8
        if not data_fault and ont.is_data_fault(fid):
            score -= 0.45
        prior = float((feedback_priors or {}).get(fid, ont.priors.get(fid, 0.1)))
        lp = math.log(max(prior, 1e-3)) + BETA * score
        scored.append((lp, fid, matched, missed, spurious))
    mx = max(lp for lp, *_ in scored)
    exps = sorted([(fid, math.exp(lp - mx), m, ms, sp) for lp, fid, m, ms, sp in scored],
                  key=lambda x: -x[1])
    tot = sum(p for _, p, _, _, _ in exps) or 1.0
    return [CauseCandidate(fault_id=fid, name=ont.faults[fid]["name"],
                           p=round(p / tot, 3), matched=m, missed=ms, spurious=sp)
            for fid, p, m, ms, sp in exps[:top_k]]
