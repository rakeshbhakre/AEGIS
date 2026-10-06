"""Tiny pure-numpy MLP autoencoder — trains and infers without torch.

Deliberate choice (plan §4.6): the SAME `mlp_forward` runs in the "ground"
pipeline and in the edge bundle, so parity is structural, not a port.
Architecture: [d, h1, h2, h1, d], tanh on hidden layers, linear output.
Training normalises inputs by median/IQR and folds the scaler into the
first/last weight matrices, so inference consumes the conditioner's scaled
windows directly.
"""
from __future__ import annotations
import numpy as np


def mlp_forward(x, ws):
    """Canonical inference path (float32 matmuls + tanh)."""
    h = np.ascontiguousarray(x, dtype=np.float32)
    n = len(ws)
    for j, (W, b) in enumerate(ws):
        h = h @ W + b
        if j < n - 1:
            h = np.tanh(h)
    return h


def _init(dims, rng):
    ws = []
    for a, b in zip(dims[:-1], dims[1:]):
        s = np.sqrt(2.0 / (a + b))
        ws.append([rng.normal(0, s, (a, b)).astype(np.float32),
                   np.zeros(b, dtype=np.float32)])
    return ws


def mlp_train(X, hidden=(96, 48), epochs=14, lr=2e-3, seed=0, batch=128,
              val_frac=0.15, log=lambda m: None):
    """X: (n_win, W, nch) scaled windows of NORMAL operation. Returns weights dict."""
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    flat = X.reshape(X.shape[0], -1)
    n = len(flat)
    idx = rng.permutation(n)
    nv = max(1, int(n * val_frac))
    tr_i, va_i = idx[nv:], idx[:nv]
    Xtr, Xva = flat[tr_i], flat[va_i]

    mu = np.median(Xtr, axis=0)
    sd = np.median(np.abs(Xtr - mu), axis=0) * 1.4826 + 1e-3
    Ztr = np.clip((Xtr - mu) / sd, -6, 6)
    Zva = np.clip((Xva - mu) / sd, -6, 6)

    d = Ztr.shape[1]
    dims = [d, *hidden, *hidden[::-1], d]
    ws = _init(dims, rng)
    mom = [(np.zeros_like(W), np.zeros_like(b)) for W, b in ws]
    vel = [(np.zeros_like(W), np.zeros_like(b)) for W, b in ws]
    t = 0
    best = (np.inf, [tuple(p.copy() for p in layer) for layer in ws])

    def fwd(Z):
        h = Z
        acts = [h]
        for j, (W, b) in enumerate(ws):
            h = h @ W + b
            if j < len(ws) - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    for ep in range(epochs):
        perm = rng.permutation(len(Ztr))
        for i0 in range(0, len(Ztr), batch):
            sel = perm[i0:i0 + batch]
            xb = Ztr[sel]
            t += 1
            out, acts = fwd(xb)
            delta = 2.0 * (out - xb) / len(xb)          # dL/d(out)
            for j in range(len(ws) - 1, -1, -1):
                if j < len(ws) - 1:
                    delta = delta * (1 - acts[j + 1] ** 2)
                a_in = acts[j]
                gW = a_in.T @ delta
                gb = delta.sum(axis=0)
                mom[j][0][:] = 0.9 * mom[j][0] + 0.1 * gW
                mom[j][1][:] = 0.9 * mom[j][1] + 0.1 * gb
                vel[j][0][:] = 0.999 * vel[j][0] + 0.001 * gW * gW
                vel[j][1][:] = 0.999 * vel[j][1] + 0.001 * gb * gb
                bc1 = 1 - 0.9 ** t
                bc2 = 1 - 0.999 ** t
                ws[j][0] -= lr * (mom[j][0] / bc1) / (np.sqrt(vel[j][0] / bc2) + 1e-8)
                ws[j][1] -= lr * (mom[j][1] / bc1) / (np.sqrt(vel[j][1] / bc2) + 1e-8)
                delta = delta @ ws[j][0].T
        va = float(np.mean((fwd(Zva)[0] - Zva) ** 2))
        tr = float(np.mean((fwd(Ztr[:4096])[0] - Ztr[:4096]) ** 2))
        log(f"ae ep{ep + 1}/{epochs} train {tr:.4f} val {va:.4f}")
        if va < best[0]:
            best = (va, [(W.copy(), b.copy()) for W, b in ws])

    # fold input scaler: layer0 (z=(x-mu)/sd @W0+b0) -> x@(W0/sd) + (b0 - mu/sd @W0)
    W0, b0 = best[1][0]
    best[1][0] = (W0 / sd[:, None], b0 - (mu / sd) @ W0)
    # fold output de-scaler: y = yz*sd+mu  (last layer is linear)
    Wl, bl = best[1][-1]
    best[1][-1] = (Wl * sd[None, :], bl * sd + mu)
    return {"ws": [(W.astype(np.float32), b.astype(np.float32)) for W, b in best[1]],
            "val": best[0]}
