"""Event-wise metrics — precision-recall on intervals, F-beta with beta<1
(precision-first, as ESA-ADB recommends), false alarms per day, time-to-detect."""
from __future__ import annotations
import numpy as np


def merge_intervals(ivs, gap=0):
    ivs = sorted(tuple(sorted((int(a), int(b)))) for a, b in ivs if b > a)
    out = []
    for a, b in ivs:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def match_events(preds, gts, grace_before=300, grace_after=900):
    """preds: [(t0,t1)] event windows; gts: [(a,b)] fault intervals.
    A pred matches a gt if its onset falls in [a-grace_before, b+grace_after]
    (or overlaps). Each gt gets at most one match (earliest pred)."""
    gts = merge_intervals(gts)
    used = set()
    tp = []            # (gt_i, pred_i, ttd)
    fp = []
    for pi, (p0, p1) in enumerate(sorted(preds, key=lambda x: x[0])):
        best = None
        for gi, (g0, g1) in enumerate(gts):
            if gi in used:
                continue
            if (g0 - grace_before) <= p0 <= (g1 + grace_after):
                if best is None or g0 < gts[best][0]:
                    best = gi
        if best is None:
            fp.append((p0, p1))
        else:
            used.add(best)
            ttd = max(0.0, p0 - gts[best][0])
            tp.append((best, pi, ttd))
    fn = [i for i in range(len(gts)) if i not in used]
    return tp, fp, fn


def f_beta(tp: int, fp: int, fn: int, beta: float = 0.5):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0, prec, rec
    f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
    return f, prec, rec


def summarize(preds, gts, span_frames, day_frames=3600, beta=0.5,
              grace_before=300, grace_after=900):
    tp, fp, fn = match_events(preds, gts, grace_before=grace_before,
                              grace_after=grace_after)
    f, prec, rec = f_beta(len(tp), len(fp), len(fn), beta)
    days = max(1e-9, span_frames / day_frames)
    ttd = [x[2] for x in tp if x[2] > 0]
    return {
        "f_event": round(f, 3), "precision": round(prec, 3), "recall": round(rec, 3),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "fp_per_day": round(len(fp) / days, 2),
        "ttd_median_s": round(float(np.median(ttd)), 1) if ttd else 0.0,
        "ttd_p90_s": round(float(np.percentile(ttd, 90)), 1) if ttd else 0.0,
    }
