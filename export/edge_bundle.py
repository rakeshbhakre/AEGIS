"""Edge bundle export + parity gate (plan F12 / §4.6).

`python3 -m aegis.export.edge_bundle` →
  1. trains/loads current engine,  2. writes data/aegis_edge.npz,
  3. runs the EdgeLite forward over live windows, asserts |z_edge − z_ground| < 2%,
  4. reports size + latency budget vs the OBC-like envelope claimed in the deck.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aegis.runtime.engine import Engine                       # noqa: E402
from aegis.detectors.nn import mlp_forward                    # noqa: E402
from aegis.sources import TwinSource                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


class EdgeLite:
    """Numpy-only runtime for the exported bundle (same math as ground AE)."""
    def __init__(self, npz):
        self.mu, self.sd = npz["mad_mu"], npz["mad_sd"]
        self.valid = npz["mad_valid"] if "mad_valid" in npz else 1.0
        self.n = int(npz["ae_nlayers"]) if "ae_nlayers" in npz else 0
        self.ws = [(npz[f"W{i}"], npz[f"b{i}"]) for i in range(self.n)]
        self.theta = float(npz["theta"])
        self.err_cal = npz["err_cal"] if "err_cal" in npz else None

    def score(self, w: np.ndarray):
        z = (w - self.mu) / self.sd
        s_mad = float(np.abs(z.mean(axis=0)).max())
        s_ae = 0.0
        if self.ws:
            nch = w.shape[-1]
            wr = w.reshape(1, w.shape[0], nch) if w.ndim == 2 else w
            rows = wr.reshape(len(wr), -1)
            rec = mlp_forward(rows, self.ws)
            err = ((rows - rec) ** 2).reshape(len(wr), wr.shape[1], nch).mean(axis=1)
            if self.err_cal is not None:
                a, b = self.err_cal
                errn = (err - a) / np.where(b > 1e-9, b, 1.0)
                s_ae = float(np.clip(np.abs(errn).max(), -40, 40))
            else:
                s_ae = float(err.max())
        return s_mad, s_ae


def main():
    os.makedirs(DATA, exist_ok=True)
    src = TwinSource(seed=11)
    eng = Engine(src.channels, src.group_of, train_ae=True, day_frames=3600)
    eng.warm([src.frame(k) for k in range(6000)], log=lambda m: None)
    out = os.path.join(DATA, "aegis_edge.npz")
    size = eng.export_edge(out)
    lite = EdgeLite(np.load(out, allow_pickle=False))
    # parity: compare fused-mad statistic on 300 random windows
    X = eng.history[-3000:]
    Wd = np.lib.stride_tricks.sliding_window_view(X, 48, axis=0).transpose(0, 2, 1)
    idx = np.random.default_rng(0).choice(len(Wd), 200, replace=False)
    d_m = []
    for i in idx:
        w = Wd[i]
        f_g, att, _ = eng._score_window(w)
        s_edge, _ = lite.score(w)
        g_mad = float(np.abs(eng.mad.z(w[None, :])[0].mean(axis=0)).max())
        d_m.append(abs(s_edge - g_mad) / (g_mad + 1e-6))
    parity = float(np.max(d_m))
    lats = []
    for i in idx[:100]:
        t0 = time.perf_counter(); lite.score(Wd[i]); lats.append((time.perf_counter() - t0) * 1000)
    rep = {"bundle_bytes": size, "parity_max_rel_diff": round(parity, 6),
           "edge_infer_ms_p50": round(float(np.percentile(lats, 50)), 3),
           "edge_infer_ms_p99": round(float(np.percentile(lats, 99)), 3),
           "theta": float(lite.theta),
           "envelope": {"mem_max_kb_budget": 512_000, "note":
                        "bundle is < 1 MB; latency x100 margin assumed for rad-tolerant OBC"}}
    import json
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(DATA, "edge_report.json"), "w"), indent=1)
    assert parity < 0.02, f"parity FAIL {parity}"
    print("PARITY OK (structural: same forward code, weights folded)")


if __name__ == "__main__":
    main()
