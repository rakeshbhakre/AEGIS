"""AEGIS synthetic spacecraft digital twin (v2 — balanced power budget).

A small but physics-shaped model of a LEO microsat bus: EPS (battery/SOC +
heater + BMS clamps), ADCS (reaction wheel + attitude), thermal RC nodes,
comms, payload.  8 fault modes with ground-truth signatures + 4 data
pathologies (gap, stale, spike, saturation) for the Sense-Check bench.

Energy bookkeeping is explicit so the *normal* stream is orbit-periodic with
stable stats (deep negative eclipse discharge + recharge, ~20% DoD):
    battery: 30 Wh Li-ion; BMS clamps SOC to [15%, 98%]
    orbit:   3600 s, eclipse 30%→70%
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

from ..contracts.models import ALL_CHANNELS, TelemetryFrame

CAP_WH = 30.0
SOC_MIN, SOC_MAX = 0.15, 0.98


@dataclass
class FaultState:
    fid: str
    start: int
    dur: int
    params: dict = field(default_factory=dict)

    def active(self, k: int) -> float:
        if k < self.start or k > self.start + self.dur:
            return 0.0
        return min(1.0, (k - self.start) / 30.0)      # 30-tick onset ramp
    # note: decay handled by dur; signature ramps in with `active`


def _sq(k: int, per: int, on_len: int, edge: int) -> float:
    """Soft square duty cycle in [0,1] with `edge`-frame linear ramps."""
    p = k % per
    if p < edge:            return p / edge
    if p < on_len - edge:   return 1.0
    if p < on_len:          return (on_len - p) / edge
    return 0.0


class SpacecraftTwin:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.state = {
            "soc": 0.80, "battery_temp": 12.5, "wheel_speed": 1150.0,
            "att_err": 0.0, "panel_temp": -4.0, "radiator_temp": 4.0,
            "payload_temp": 16.0, "heater_on": 0.0, "heater_t": -999,
        }

    # ------------------------------------------------------------------ physics
    def step(self, k: int, faults: list[FaultState]):
        r = self.rng
        s = self.state
        orbit = (k % 3600) / 3600.0
        eclipse = 0.30 < orbit < 0.70
        solar_now = max(0.0, math.cos(orbit * 2 * math.pi))
        fx = {f.fid: f.active(k) for f in faults if f.active(k) > 0}

        # --- loads (W)
        bus_w = 4.2 + (0.8 if not eclipse else 0.4) + 0.5 * _sq(k, 240, 90, 8)
        pay_duty = _sq(k, 600, 240, 10)
        pay_w = 1.2 + 8.5 * pay_duty
        pay_w += 9.5 * fx.get("payload_overload", 0.0) * (0.4 + pay_duty)
        heat_w = 2.0 * s["heater_on"] * (1.0 if s["soc"] > 0.32 else 0.0)
        wheel_term = 0.1 + 0.6 * min(1.0, abs(s["wheel_speed"] - 1150) / 900)

        # --- generation (W); sag when array hot
        p_sol = 36.0 * solar_now * (1 - 0.12 * min(1.0, max(0.0, s["panel_temp"]) / 40))
        p_sol -= 6.5 * fx.get("cell_sag", 0.0) * _sq(k, 240, 90, 8)

        p_draw = bus_w + pay_w + heat_w + wheel_term
        net = p_sol - p_draw
        if "cell_sag" in fx:
            net -= 3.2 * fx["cell_sag"]
        s["soc"] += net / (CAP_WH * 3600.0)
        s["soc"] = min(SOC_MAX, max(SOC_MIN, s["soc"]))
        # BMS low-soc load shedding (keeps sim sane under deep faults)
        shed = 1.0 if s["soc"] > SOC_MIN + 0.01 else 0.55

        # --- terminal voltage & current
        v = 6.8 + 1.9 * (s["soc"] - 0.4) + 0.02 * (s["battery_temp"] - 12.0) \
            - 0.10 * (p_draw * shed / 12.0) + r.gauss(0, 0.008)
        if "cell_sag" in fx:
            v -= 0.7 * fx["cell_sag"] * (0.5 + 0.5 * (1 - solar_now))
        i_bat = (p_draw * shed) / max(3.5, v) + r.gauss(0, 0.04)
        if "cell_sag" in fx:
            i_bat += 0.8 * fx["cell_sag"]

        # --- thermal nodes (heater cycles; cell_sag & runaway couple in)
        # orbit-synchronous heater schedule (eclipse-side pulses); the
        # context baseline removes this exactly — residual = real deviation.
        if s.get("heater_block"):
            s["heater_on"] = 0.0                        # stalled: fail-safe disabled
        else:
            ph = k % 3600
            want = (1250 <= ph < 1550) or (1700 <= ph < 2000)
            if want and s["battery_temp"] < 15.5:
                s["heater_on"] = 1.0
            elif not want and s["battery_temp"] < 9.5:
                s["heater_on"] = 1.0
            elif not want:
                s["heater_on"] = 0.0
        if s["soc"] <= SOC_MIN + 1e-9:
            s["heater_on"] = 0.0                      # BMS cut on heater
        dT = 0.0038 * (13.0 * s["heater_on"] - (s["battery_temp"] - 4.5))
        if "heater_stall" in fx:            # lost heater + degraded insulation
            dT -= 0.030 * fx["heater_stall"]
        if "thermal_runaway" in fx:
            dT += 0.05 * fx["thermal_runaway"] * (1.0 + max(0.0, (s["battery_temp"] - 20) / 15))
        s["battery_temp"] = max(-30.0, min(85.0, s["battery_temp"] + dT + r.gauss(0, 0.03)))
        s["panel_temp"] += 0.012 * ((26 * solar_now - 9) - s["panel_temp"]) + r.gauss(0, 0.04)
        s["radiator_temp"] += 0.010 * ((8 + 14 * solar_now) - s["radiator_temp"])

        # --- ADCS
        s["att_err"] += r.gauss(0, 0.02) - 0.06 * s["att_err"]
        fric = fx.get("wheel_friction", 0.0)
        if fric > 0:
            s["att_err"] += 0.012 * fric
        wheel_target = 1150 + 700 * s["att_err"] * (1 - 0.35 * fric)
        s["wheel_speed"] += 0.32 * (wheel_target - s["wheel_speed"]) + r.gauss(0, 2.4)
        wheel_i = 0.85 + 1.0 * abs(s["wheel_speed"] - 1150) / 700 + 2.8 * fric \
            + 0.35 * max(0.0, (s["wheel_speed"] - 1150) / 900)
        wheel_t = 20.0 + 0.00002 * (s["wheel_speed"] - 1000) ** 2 + 26.0 * fric \
            + r.gauss(0, 0.12)

        # --- comm & payload housekeeping
        sig = -78 + 8 * math.sin(orbit * 2 * math.pi) + r.gauss(0, 0.35)
        pay_temp_target = 14.0 + 0.9 * pay_w
        s["payload_temp"] += 0.010 * (pay_temp_target - s["payload_temp"])
        cpu = 24 + 34 * (1 - eclipse) + 22 * fx.get("payload_overload", 0.0) * pay_duty \
            + 14 * fric + r.gauss(0, 2.0)

        vals = {
            "battery_voltage": round(v, 4),
            "battery_current": round(i_bat, 4),
            "soc": round(100 * s["soc"], 2),
            "solar_current": round(p_sol / 7.0 + r.gauss(0, 0.05), 4),   # A at bus
            "heater_current": round(2.0 * s["heater_on"] + r.gauss(0, 0.015), 4),
            "battery_temp": round(s["battery_temp"], 3),
            "wheel_speed": round(s["wheel_speed"], 1),
            "wheel_current": round(wheel_i, 4),
            "wheel_temp": round(wheel_t, 3),
            "attitude_err": round(abs(s["att_err"]), 4),
            "panel_temp": round(s["panel_temp"], 3),
            "radiator_temp": round(s["radiator_temp"], 3),
            "signal_strength": round(sig, 3),
            "payload_power": round(pay_w, 3),
            "payload_temp": round(s["payload_temp"], 3),
            "cpu_load": round(max(1.0, min(99.5, cpu)), 2),
        }
        assert set(vals) == set(ALL_CHANNELS), "twin channels must match contracts"

        truth: dict = {"channels": {}, "fid": None, "start": None}
        for f in faults:
            a = f.active(k)
            if a <= 0:
                continue
            kind, chs = FAULT_EFFECTS[f.fid]
            for ch, fn in chs.items():
                vals[ch] = round(fn(vals[ch], k, a, r), 4)
                truth["channels"].setdefault(ch, {"dir": "up", "kind": kind})
            truth["fid"], truth["start"] = f.fid, f.start
        return vals, truth


# extra channel nudges on top of physics (used for sensor-only faults)
def _drift(v, k, a, r): return v + 0.05 * a * (k / 600)
def _bias(v, k, a, r):  return v + 0.9 * a
def _freeze12(v, k, a, r): return 12.0


FAULT_EFFECTS: dict[str, tuple[str, dict]] = {
    # ---- health faults (physics already handles most; these add channel ops)
    "heater_stall": ("jump", {
        "heater_current": lambda v, k, a, r: v * (1 - 0.96 * a),
    }),
    "sensor_drift": ("ramp", {
        "battery_voltage": _drift,
    }),
    "sensor_stuck": ("freeze", {
        "battery_temp": _freeze12,
    }),
    "sensor_bias": ("jump", {
        "solar_current": _bias,
    }),
    "cell_sag": ("ramp", {}),          # physics: v, soc, i effects
    "wheel_friction": ("ramp", {}),    # physics: current, temp, speed, att
    "thermal_runaway": ("jump", {}),   # physics: temperature loop
    "payload_overload": ("jump", {}),  # physics: pay_w, cpu
    "link_loss": ("data_gap", {"signal_strength": lambda v, k, a, r: min(v, -112)}),
}

DATA_PATHOLOGIES = ("gap", "stale", "spike", "saturate")


class FaultInjector:
    def __init__(self, twin: SpacecraftTwin):
        self.twin = twin
        self.faults: list[FaultState] = []
        self.pathology: dict = {}
        self.frozen: dict = {}
        # ---- link/quality model (organiser §12): live-adjustable, SIMULATION ----
        self.noise_gain = 1.0     # multiplies measurement noise on every channel
        self.missing_rate = 0.0   # per-sample independent dropout probability
        self.CH_FLOOR = {"wheel_speed": 8.0, "cpu_load": 1.2, "signal_strength": 0.8,
                         "battery_soc": 0.004}

    def set_quality(self, noise: float | None = None, missing: float | None = None):
        if noise is not None:
            self.noise_gain = max(0.0, float(noise))
        if missing is not None:
            self.missing_rate = min(0.6, max(0.0, float(missing)))
        return {"noise_gain": self.noise_gain, "missing_rate": self.missing_rate}

    def inject(self, fid: str, at: int = 0, dur: int = 420, **params):
        self.faults.append(FaultState(fid, at, dur, params))
        return self

    def inject_data(self, channel: str, kind: str, at: int, dur: int):
        assert kind in DATA_PATHOLOGIES
        self.pathology[channel] = {"kind": kind, "start": at, "until": at + dur,
                                   "hold": None}
        return self

    def storm(self, at: int, fids: list[str], dur: int = 480):
        for i, f in enumerate(fids):
            self.faults.append(FaultState(f, at + 20 * i, dur))
        return self

    def frame(self, k: int) -> TelemetryFrame:
        vals, truth = self.twin.step(k, self.faults)
        r = self.twin.rng
        _miss: list = []
        # global noise / missing-quality layer (applied to the measurement,
        # not the physics — a sensor/link effect, so the detector must weigh it)
        if self.noise_gain != 1.0:
            for ch in list(vals):
                amp = abs(vals[ch]) if abs(vals[ch]) > 1e-6 else 1.0
                vals[ch] += r.gauss(0.0, 0.006 * amp * (self.noise_gain - 1.0)) \
                            + r.gauss(0.0, self.CH_FLOOR.get(ch, 0.0) * (self.noise_gain - 1.0))
        if self.missing_rate > 0.0:
            for ch in list(vals):
                if r.random() < self.missing_rate:
                    vals[ch] = float("nan")
                    _miss.append(ch)
        # physics-internal heater suppression for heater_stall is handled via
        # channel op only; to also stop heating physically we clear the relay:
        for f in self.faults:
            if f.fid == "heater_stall":
                self.twin.state["heater_block"] = 1.0 if f.active(k) > 0 else 0.0
        flags: dict = {}
        for ch in _miss:
            flags[ch] = {"dropout": True}
        for ch, d in self.pathology.items():
            if d["start"] <= k <= d["until"]:
                if d["kind"] == "gap" and (k - d["start"]) % 9 < 4:
                    vals[ch] = float("nan"); flags[ch] = {"dropout": True}
                elif d["kind"] == "stale":
                    if d["hold"] is None:
                        d["hold"] = vals[ch]
                    vals[ch] = d["hold"]; flags[ch] = {"stale": True}
                elif d["kind"] == "spike" and (k - d["start"]) % 47 == 0:
                    vals[ch] = vals[ch] * 3 + 10; flags[ch] = {"noise": True}
                elif d["kind"] == "saturate":
                    vals[ch] = 100.0 if ch == "cpu_load" else 0.0
                    flags[ch] = {"saturated": True}
            elif d["start"] > k:
                d["hold"] = None
        if k % 977 == 13:
            flags.setdefault("panel_temp", {})["delay_s"] = 12.0
        truth["faults"] = [(f.fid, f.start, f.dur)
                           for f in self.faults if f.start <= k < f.start + f.dur]
        return TelemetryFrame(t=float(k), values=vals, flags={**flags, "_truth": truth})
