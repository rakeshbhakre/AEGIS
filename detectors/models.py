"""Detectors: MAD baseline, AE (sklearn MLP / numpy), Isolation-Forest,
ensemble fusion, event engine with hysteresis, conformist calibration.

All inference is numpy-only after fit() → the same code path runs in the edge
build (see aegis/export/edge_bundle.py).
"""
from __future__ import annotations
import time
import numpy as np
from collections import deque

W = 48  # window length (samples)


class MADDetector:
    """Pure-numpy robust baseline; also the source of per-channel attributions."""
    def __init__(self):
        self.ready = False
        self.valid = None

    def fit(self, X: np.ndarray):  # X: (n, nch) scaled
        self.mu = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.mu), axis=0) * 1.4826
        # Floor by 0.15·std: quantized/bimodal channels (MAD=0 on flag-like
        # telemetry) would otherwise divide micro-jitter by 1e-6 and saturate
        # max-of-z statistics; Gaussian channels are unaffected (MAD≈std).
        self.sd = np.maximum(mad, 0.15 * np.std(X, axis=0, dtype=np.float64)) + 1e-9
        # quasi-constant channels (MAD ≪ data range: setpoint-like, quantized
        # jitter) would saturate max-of-z stats → exclude them from z entirely.
        F = np.isfinite(X)
        rng = np.where(F.any(0), X[F].max(0) - X[F].min(0), 0.0) if X.ndim == 2 else \
            np.where(F.any((0, 1)), X[F].max((0, 1)) - X[F].min((0, 1)), 0.0)
        self.valid = ((self.sd > 1e-3 * np.maximum(rng, 1e-30)) & (rng > 0)).astype(np.float32)
        self.ready = True
        return self

    def z(self, X: np.ndarray) -> np.ndarray:
        if not self.ready:
            return np.zeros_like(X)
        return ((X - self.mu) / self.sd) * getattr(self, "valid", 1.0)


class AEDetector:
    """MLP autoencoder with pure-numpy Adam training + numpy inference.

    Kept deliberately small (816→96→48→96→816 on flattened 48×nch windows) so
    training is minutes on CPU and inference is ~1 ms/frame — the edge budget.
    Falls back gracefully when untrained (returns zeros → ensemble re-weights).
    """
    def __init__(self, hidden=(96, 48), epochs=14, lr=2e-3, seed=0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.W_ = None

    def fit(self, X: np.ndarray, log=lambda m: None):
        from .nn import mlp_train
        t0 = time.time()
        self.W_ = mlp_train(X, hidden=self.hidden, epochs=self.epochs,
                            lr=self.lr, seed=self.seed, log=log)
        self.train_s = time.time() - t0
        return self

    def recon_err(self, X: np.ndarray) -> np.ndarray:
        """per-window channel-wise error (n_win, nch)"""
        from .nn import mlp_forward
        if self.W_ is None:
            return np.zeros((X.shape[0], X.shape[2]))
        flat = X.reshape(X.shape[0], -1)
        rec = mlp_forward(flat, self.W_["ws"])
        nch = X.shape[2]
        return ((flat - rec) ** 2).reshape(-1, X.shape[1], nch).mean(axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        e = self.recon_err(X)
        s = np.exp(np.clip(e.mean(axis=1), -10, 10))
        return s


class IForestDetector:
    """IsolationForest on engineered window features (catches slow drifts)."""
    def __init__(self, n_estimators=180, seed=0):
        self.n, self.seed = n_estimators, seed
        self.m = None

    @staticmethod
    def feats(X: np.ndarray) -> np.ndarray:
        n, w, c = X.shape
        mean = X.mean(axis=1)
        std = X.std(axis=1) + 1e-9
        t = np.arange(w)
        tt = t - t.mean()
        denom = (tt ** 2).sum() + 1e-9
        slope = (X - mean[:, None, :]) * tt[None, :, None]
        slope = slope.sum(axis=1) / denom
        maxdiff = np.abs(np.diff(X, axis=1)).max(axis=1)
        return np.hstack([mean, std * 2, slope, maxdiff, np.abs(mean) ** 1.5])

    def fit(self, X: np.ndarray):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=self.n, random_state=self.seed,
                                 contamination=0.02).fit(self.feats(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.m is None:
            return np.zeros(X.shape[0])
        f = self.feats(X)
        return -self.m.score_samples(f)


class Ensemble:
    """Fuse per-window scores with robust calibration into z-space; EMA +
    hysteresis handled by the EventEngine downstream."""
    def __init__(self, mad: MADDetector, ae: AEDetector, ifo: IForestDetector):
        self.mad, self.ae, self.ifo = mad, ae, ifo
        self.cal = {}

    def calibrate(self, win_scores: np.ndarray, err_ch: np.ndarray):
        """win_scores: (n_win, 3) [ae, if, mad] ; err_ch: (n_win, nch)"""
        self.cal["ae"] = _robust(win_scores[:, 0])
        self.cal["if"] = _robust(win_scores[:, 1])
        self.cal["err"] = (_robust(err_ch.mean(axis=0)))

    def window_scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        per_ch = self.mad.z(X).clip(-10, 10)
        mad_s = np.percentile(per_ch, 96, axis=(1, 2)) if n else np.zeros(0)
        ae_e = self.ae.recon_err(X) if self.ae.W_ is not None else np.zeros((n, X.shape[2]))
        ae_s = np.exp(np.clip(ae_e.mean(axis=1), -10, 10)) if n else np.zeros(0)
        if_s = self.ifo.score(X) if n else np.zeros(0)
        z_ae = _z(ae_s, self.cal.get("ae", (0.0, 1.0)))
        z_if = _z(if_s, self.cal.get("if", (0.0, 1.0)))
        # per-channel attribution z from both mad & ae error
        a, b = self.cal.get("err", (np.zeros(X.shape[2]), np.ones(X.shape[2])))
        att = 0.5 * per_ch + 0.5 * (ae_e - a) / np.where(b > 1e-9, b, 1.0)
        fuse = 0.30 * z_ae + 0.25 * z_if + 0.45 * mad_s if self.ae.W_ is not None \
            else 0.55 * z_if + 0.45 * mad_s
        return fuse, att, ae_e


def _robust(v):
    v = np.asarray(v)
    if v.size == 0:
        return (0.0, 1.0) if v.ndim == 1 else (np.zeros(v.shape[-1]), np.ones(v.shape[-1]))
    med = np.median(v, axis=0)
    iqr = np.subtract(*np.percentile(v, [75, 25], axis=0)) if v.ndim > 1 else (np.median(np.abs(v - med)) * 1.4826)
    iqr = np.asarray(iqr) + 1e-9
    return (med, np.where(np.abs(iqr) < 1e-12, 1.0, iqr))


def _z(v, cal):
    med, iqr = cal
    return (np.asarray(v) - med) / iqr
