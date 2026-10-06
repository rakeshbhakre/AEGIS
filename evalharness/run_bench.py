"""AEGIS benchmark suite — one command: `make bench`.

  twin      : 8 health faults × N episodes → detection F0.5, FP/day, TTD,
              RCA top-1/top-3, guard stats
  datafault : 4 data pathologies → HEALTH-alert rejection rate (Sense-Check)
  smap      : real NASA SMAP test telemetry → event F0.5 on expert labels
Writes data/bench.json + README benchmark block.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

from ..sources import TwinSource, load_smap
from ..runtime.engine import Engine
from .metrics import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
HEALTH_FAULTS = ["heater_stall", "cell_sag", "wheel_friction", "thermal_runaway",
                 "payload_overload", "sensor_drift", "sensor_stuck", "sensor_bias"]


def make_engine(warm_n=7400, ae=False, log=lambda m: None):
    src = TwinSource(seed=11)
    eng = Engine(src.channels, src.group_of, train_ae=ae, day_frames=3600)
    eng.warm([src.frame(k) for k in range(warm_n)], log=log)
    return src, eng


def bench_twin(episodes=10, gap=1600, dur_lo=380, dur_hi=700, seed=3):
    rng = np.random.default_rng(seed)
    src, eng = make_engine()
    rows = {f: {"det": 0, "ep": 0, "ttd": [], "top1": 0, "top3": 0, "guard_auto": 0,
                "guard_prop": 0, "false_auto": 0} for f in HEALTH_FAULTS}
    all_pred = []          # (t0,t1)
    plan = []
    k = 7700               # ≥ warm_n + margin: faults never pollute calibration
    for f in HEALTH_FAULTS:
        for e in range(episodes):
            dur = int(rng.integers(dur_lo, dur_hi))
            plan.append((f, k + int(rng.integers(0, 400)), dur))
            k += dur + gap
    for f, start, dur in plan:
        src.inject(f, at=start, dur=dur)
    total = k + 600
    ev_by_id = {}
    for i in range(7400, total):
        b = eng.step(src.frame(i))
        if b:
            all_pred.append((b.event.t0, b.event.t1, b))
    for f, start, dur in plan:
        m = [p for p in all_pred if (start - 300) <= p[0] <= (start + dur + 300)]
        r = rows[f]
        r["ep"] += 1
        if m:
            r["det"] += 1
            b = m[0][2]
            r["ttd"].append(max(0.0, b.event.t0 - start))
            cands = [c.fault_id for c in b.causes]
            r["top1"] += int(bool(cands) and cands[0] == f)
            r["top3"] += int(f in cands)
            autos = [a for a in b.actions if a.tier == "auto_safe" and a.allowed]
            r["guard_auto"] += len(autos) > 0
            r["guard_prop"] += any(a.allowed for a in b.actions if a.tier != "auto_safe")
            # "false auto" if auto fired for the WRONG top cause
            if autos and cands and cands[0] != f:
                r["false_auto"] += 1
    # FP on unclaimed predictions
    claimed = set()
    for f, start, dur in plan:
        for pi, p in enumerate(all_pred):
            if (start - 300) <= p[0] <= (start + dur + 300):
                claimed.add(pi)
    fps = [p for i, p in enumerate(all_pred) if i not in claimed]
    out = []
    for f, r in rows.items():
        out.append({
            "fault": f, "episodes": r["ep"],
            "detect_rate": round(r["det"] / max(1, r["ep"]), 3),
            "ttd_med_s": round(float(np.median(r["ttd"])) if r["ttd"] else 0, 1),
            "rca_top1": round(r["top1"] / max(1, r["det"]), 3),
            "rca_top3": round(r["top3"] / max(1, r["det"]), 3),
            "guarded_auto": r["guard_auto"], "proposed": r["guard_prop"],
            "false_auto_on_wrong_cause": r["false_auto"],
        })
    days = (total - 7400) / 3600.0
    return {"per_fault": out,
            "unmatched_fp_per_day": round(len(fps) / max(1e-9, days), 3),
            "episodes": sum(r["ep"] for r in rows.values()),
            "detections": sum(r["det"] for r in rows.values())}


def bench_limit(episodes=10, gap=1600, dur_lo=380, dur_hi=700, seed=3, warm_n=7400):
    """Traditional housekeeping limit-checking baseline on the SAME injected
    schedule bench_twin uses, identical matching tolerance. Limits learned from
    warmup only (chronological, no future data): med ± max(3.5·robustσ, floor)
    with 12-frame persistence and ≥120 s merge — a conventional 3σ-style
    warning band. Measures recall/precision on faults AND the false-alarm load
    under ×3 sensor noise on a clean stream."""
    import numpy as np
    from ..sources import TwinSource
    from .metrics import merge_intervals
    HEALTH = list(HEALTH_FAULTS)
    HOLD = 12
    rng = np.random.default_rng(seed)
    src = TwinSource(seed=7)
    plan = []
    k = warm_n + 300
    for f in HEALTH:
        for e in range(episodes):
            dur = int(rng.integers(dur_lo, dur_hi))
            plan.append((f, k + int(rng.integers(0, 400)), dur))
            k += dur + gap
    for f, start, dur in plan:
        src.inject(f, at=start, dur=dur)
    total = k + 600
    X = np.empty((total, len(src.channels)), np.float64)
    for i2 in range(total):
        fr = src.frame(i2)
        X[i2] = [fr.values.get(c, float("nan")) for c in src.channels]
    base = X[:warm_n]
    med = np.nanmedian(base, axis=0)
    sd = np.nanmedian(np.abs(base - med), axis=0) * 1.4826
    sd = np.maximum(sd, 0.02 * np.nanstd(base, axis=0)) + 1e-9
    up, lo = med + 3.5 * sd, med - 3.5 * sd
    def limit_events(Xseg):
        flags = ((Xseg > up) | (Xseg < lo)) & np.isfinite(Xseg)
        n, m = flags.shape
        in_run = np.zeros(m, bool); start = np.zeros(m, int)
        runs = []
        for i3 in range(n):
            hit = flags[i3]
            beg = hit & ~in_run
            in_run |= beg
            start = np.where(beg, i3, start)
            endc = np.where(~hit & in_run)[0]
            for c in endc:
                if i3 - start[c] >= HOLD:
                    runs.append((int(start[c]), i3))
                in_run[c] = False
            # a frame that ENDS one run may START the next — handled next iter
        for c in np.where(in_run)[0]:
            if n - start[c] >= HOLD:
                runs.append((int(start[c]), n))
        return merge_intervals(runs, gap=120)
    ev = [(a + warm_n, b + warm_n) for a, b in limit_events(X[warm_n:])]
    per, ttds_all = {}, []
    claimed = set()
    for f, start, dur in plan:
        m = [pi for pi, (a, b2) in enumerate(ev)
             if (start - 300) <= a <= (start + dur + 300) and pi not in claimed]
        r = per.setdefault(f, {"ep": 0, "det": 0, "ttd": []})
        r["ep"] += 1
        if m:
            r["det"] += 1
            claimed.add(m[0])
            lag = ev[m[0]][0] - start
            r["ttd"].append(max(0.0, lag))
            ttds_all.append(lag)
    fps = len([pi for pi in range(len(ev)) if pi not in claimed])
    days = (total - warm_n) / 3600.0
    prec = len(claimed) / max(1, len(ev))
    rec = sum(r["det"] for r in per.values()) / max(1, len(plan))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    src2 = TwinSource(seed=9)
    src2.inj.set_quality(noise=3.0)
    Xc = np.empty((7200, len(src.channels)), np.float64)
    for i4 in range(7200):
        fr = src2.frame(i4)
        Xc[i4] = [fr.values.get(c, float("nan")) for c in src.channels]
    b2a = Xc[:3600]
    m2 = np.nanmedian(b2a, axis=0)
    s2 = np.nanmedian(np.abs(b2a - m2), axis=0) * 1.4826
    s2 = np.maximum(s2, 0.02 * np.nanstd(b2a, axis=0)) + 1e-9
    seg = Xc[3600:]
    up2, lo2 = m2 + 3.5 * s2, m2 - 3.5 * s2
    saved = (up, lo)
    up, lo = up2, lo2
    nfa = len(limit_events(seg))
    up, lo = saved
    return {
        "per_fault": [{"fault": f, "episodes": r["ep"],
                       "detect_rate": round(r["det"] / max(1, r["ep"]), 3),
                       "ttd_med_s": round(float(np.median(r["ttd"])) if r["ttd"] else 0, 1)}
                      for f, r in per.items()],
        "summary": {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                    "alerts": len(ev), "matched": len(claimed), "fp": fps,
                    "fp_per_day": round(fps / max(1e-9, days), 2),
                    "ttd_median_s": round(float(np.median(ttds_all)) if ttds_all else 0, 1)},
        "noise_x3_false_alarms_clean_1h": nfa,
        "params": "med±3.5·robustσ, 12-frame persistence, 120 s merge",
        "note": "conventional housekeeping limit check; limits from warmup only; "
                "no multivariate, no temporal, no corroboration by construction",
    }


def bench_datafault(n=28, gap=1100):
    src, eng = make_engine()
    chs = src.channels
    rng = np.random.default_rng(5)
    plan = []
    k = 600
    for i in range(n):
        kind = ["gap", "stale", "spike", "saturate"][i % 4]
        ch = chs[int(rng.integers(0, len(chs)))]
        plan.append((ch, kind, k + int(rng.integers(0, 300))))
        k += gap
    for ch, kind, at in plan:
        src.inject_data(ch, kind, at=at, dur=160)
    health = 0
    data_events = 0
    for i in range(7400, k + 400):     # live only after warm + fault pre-injection
        b = eng.step(src.frame(i))
        if b:
            if b.event.origin == "DATA":
                data_events += 1
            else:
                # LEAK = detector keyed on the pathology channel itself.
                t = b.event.t0
                achs = {a.channel for a in b.event.attrib}
                for ch_, kind_, at_ in plan:
                    if at_ - 200 <= t <= at_ + 500 and ch_ in achs:
                        health += 1
                        break
    tot = len(plan)
    return {"injected": tot, "health_false_alarms": health,
            "data_origin_alerts": data_events,
            "rejection_rate": round(1 - health / max(1, tot), 3)}


def bench_smap(n_sats=6, warm=3000):
    if not os.path.isdir(os.path.join(DATA, "SMAP")):
        return {"note": "SMAP not downloaded — run make data"}
    rows = []
    ids = [x for x in _smap_sat_ids()]
    for fi, sid in enumerate(ids[:n_sats * 4]):
        if len(rows) >= n_sats:
            break
        srcs = load_smap(os.path.join(DATA, "SMAP"), sats=[sid])
        if not srcs:
            continue
        s = srcs[0]
        if s.T < 4000 or not s.labels.get("TELEMETRY"):
            continue                     # mirror holds truncated/flag-only series
        warm_i = max(1500, min(warm, s.T // 3))
        eng = Engine(s.channels, s.group_of, train_ae=False, day_frames=1440)
        w = [s.frame() for _ in range(warm_i)]
        w = [f for f in w if f]
        eng.warm(w, log=lambda m: None)
        preds = []
        for i in range(len(w), s.T):
            fr = s.frame()
            if fr is None:
                break
            b = eng.step(fr)
            if b:
                preds.append((b.event.t0, b.event.t1))
        gt = [((a) * 4.0, (b) * 4.0) for a, b in s.labels.get("TELEMETRY", [])]
        span = max(1, s.T - len(w))
        summ = summarize([(p[0] / 4, p[1] / 4) for p in preds],
                         [(int(a / 4), int(b / 4)) for a, b in gt],
                         span_frames=span, day_frames=360,
                         grace_before=90, grace_after=400)
        rows.append({"sat": sid, **summ})
    if not rows:
        return {"note": "no SMAP satellites parsed"}
    agg = {k: round(float(np.mean([r[k] for r in rows])), 3)
           for k in ("f_event", "precision", "recall", "fp_per_day")}
    return {"satellites": rows, "mean": agg}


def _smap_sat_ids():
    """Prefer data-rich satellites (float channels, not flag-only)."""
    d = os.path.join(DATA, "SMAP", "test")
    score = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".npy"):
            continue
        try:
            a = np.load(os.path.join(d, f), mmap_mode="r")
            sl = np.asarray(a[::7])
            rich = sum(int(np.unique(sl[:, i][np.isfinite(sl[:, i])]).size) > 100
                       for i in range(sl.shape[1]))
            score.append((rich, os.path.splitext(f)[0]))
        except Exception:
            continue
    return [sid for _r, sid in sorted(score, reverse=True)]


def edge_stats():
    """Latency / footprint of the streaming step (the 'edge-lite' path)."""
    src, eng = make_engine()
    import time
    lats = []
    for i in range(7400, 8600):
        t0 = time.perf_counter()
        eng.step(src.frame(i))
        lats.append((time.perf_counter() - t0) * 1000)
    lat = np.asarray(lats)
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"step_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "step_ms_p99": round(float(np.percentile(lat, 99)), 2),
            "rss_peak_mb": round(rss_mb, 1),
            "note": "single-process numpy path; flight OBC would run the same forward in int8"}


def emit_tables(bench: dict):
    lines = ["", "## Benchmark summary (generated by `make bench`)", "",
             "| bench | metric | value |", "|---|---|---|"]
    t = bench.get("twin", {})
    for r in t.get("per_fault", []):
        lines.append(f"| twin | {r['fault']}: detect | {r['detect_rate']} |")
    if t:
        lines.append(f"| twin | RCA top-3 (mean) | {round(float(np.mean([r['rca_top3'] for r in t['per_fault']])),3)} |")
        lines.append(f"| twin | unmatched FP/day | {t.get('unmatched_fp_per_day')} |")
    if "datafault" in bench:
        d = bench["datafault"]
        lines.append(f"| sense-check | data-fault rejection | {d['rejection_rate']} ({d['health_false_alarms']}/{d['injected']} leaked) |")
    if "smap" in bench and "mean" in bench["smap"]:
        m = bench["smap"]["mean"]
        lines.append(f"| NASA SMAP (real) | event F0.5 | {m['f_event']} |")
        lines.append(f"| NASA SMAP (real) | precision / recall | {m['precision']} / {m['recall']} |")
        lines.append(f"| NASA SMAP (real) | FP/day | {m['fp_per_day']} |")
    e = bench.get("edge", {})
    if e:
        lines.append(f"| edge-lite | step latency p50/p99 (ms) | {e['step_ms_p50']} / {e['step_ms_p99']} |")
    txt = "\n".join(lines) + "\n"
    rp = os.path.join(ROOT, "README.md")
    if os.path.exists(rp):
        old = open(rp).read()
        if "<!-- BENCH:BEGIN -->" in old:
            head, rest = old.split("<!-- BENCH:BEGIN -->", 1)
            _, tail = rest.split("<!-- BENCH:END -->", 1)
            open(rp, "w").write(head + "<!-- BENCH:BEGIN -->\n" + txt + "<!-- BENCH:END -->" + tail)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma list: twin,datafault,smap,edge")
    ap.add_argument("--episodes", type=int, default=8)
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    bench = {}
    os.makedirs(DATA, exist_ok=True)
    prev_path = os.path.join(DATA, "bench.json")
    if os.path.exists(prev_path):
        try:
            bench.update(json.load(open(prev_path)))
        except Exception:
            pass
    def save():
        json.dump(bench, open(os.path.join(DATA, "bench.json"), "w"), indent=1)
    for key, fn in [("twin", lambda: bench_twin(episodes=args.episodes)),
                    ("limit", bench_limit), ("datafault", bench_datafault), ("smap", bench_smap),
                    ("edge", edge_stats)]:
        if key in skip:
            continue
        print(f":: {key} bench", flush=True)
        try:
            bench[key] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            bench.setdefault("errors", {})[key] = repr(e)
        save()
    print(emit_tables(bench))
    print("wrote data/bench.json")


if __name__ == "__main__":
    sys.exit(main())
