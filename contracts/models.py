"""AEGIS contracts — frozen interfaces between modules (plan §3).

Every module is a pure function over these dataclasses. Pydantic models are for
the API boundary only; runtime internals use these for speed.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TelemetryFrame:
    """Raw frame entering the pipeline (what a CCSDS collector would emit).

    t      = EVENT time (spacecraft clock, seconds) — the truth for physics.
    t_arr  = ARRIVAL time at the ground collector; frames may arrive late or
             out of arrival order, but temporal reasoning MUST use t.
             (t_arr == t when the link is ideal.)"""
    t: float
    values: dict
    flags: dict = field(default_factory=dict)
    t_arr: float = -1.0            # seconds on the same clock; <0 → == t


@dataclass
class QualityFlags:
    """Per-frame data-quality assessment (F3 Sense-Check)."""
    t: float
    imputed: list = field(default_factory=list)     # channels with filled gaps
    impulse: list = field(default_factory=list)     # repeated median-filter rejects (spike noise)
    stale: list = field(default_factory=list)       # frozen sensor
    saturated: list = field(default_factory=list)   # at-rail
    delayed: list = field(default_factory=list)      # channels whose value arrived late
    late_beyond: list = field(default_factory=list)  # arrived beyond reorder window
    delay_s: float = 0.0
    veto_channels: list = field(default_factory=list)  # channels the detector must not trust


@dataclass
class ChannelAttrib:
    channel: str
    z: float          # robust z of deviation
    direction: str    # up | down | freeze | jump
    timescale: str    # jump | ramp | spike | oscillation | freeze
    onset: float


@dataclass
class AnomalyEvent:
    id: str
    t0: float
    t1: float
    group: str
    score: float
    conformal_p: float
    quality: float = 1.0                        # 0..1 data-quality over the window
    quality_detail: dict = field(default_factory=dict)  # missing%/max_delay/veto_share
    attrib: list = field(default_factory=list)   # list[ChannelAttrib]
    origin: str = "HEALTH"                        # HEALTH | DATA
    veto_reasons: list = field(default_factory=list)


@dataclass
class CausalLink:
    src: str
    dst: str
    lag_frames: int
    strength: float   # |delta corr| during vs before event
    prior: float = 0.0  # FMECA prior boost


@dataclass
class CauseCandidate:
    fault_id: str
    name: str
    p: float
    matched: list = field(default_factory=list)
    missed: list = field(default_factory=list)
    spurious: list = field(default_factory=list)


@dataclass
class ActionProposal:
    action_id: str
    name: str
    tier: str            # auto_safe | propose | advisory
    allowed: bool
    why_safe: list = field(default_factory=list)
    blocked_by: list = field(default_factory=list)
    effect: str = ""


@dataclass
class EvidenceBundle:
    event: AnomalyEvent
    links: list = field(default_factory=list)      # CausalLink
    causes: list = field(default_factory=list)     # CauseCandidate (top-k)
    actions: list = field(default_factory=list)    # ActionProposal
    provenance: dict = field(default_factory=dict)

    def to_dict(self):
        import dataclasses as _dc
        def enc(x):
            if _dc.is_dataclass(x) and not isinstance(x, type):
                return enc(asdict(x))
            if isinstance(x, np.floating):
                return float(x)
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, (int, float, str, bool)) or x is None:
                return x
            if isinstance(x, (list, tuple)):
                return [enc(i) for i in x]
            if isinstance(x, dict):
                return {k: enc(v) for k, v in x.items()}
            return str(x)
        return enc(self)


CHANNEL_GROUPS = {
    "EPS": ["battery_voltage", "battery_current", "soc", "solar_current",
            "heater_current", "battery_temp"],
    "ADCS": ["wheel_speed", "wheel_current", "wheel_temp", "attitude_err"],
    "THERMAL": ["panel_temp", "radiator_temp"],
    "COMM": ["signal_strength"],
    "PAYLOAD": ["payload_power", "payload_temp"],
    "OBC": ["cpu_load"],
}
ALL_CHANNELS = [c for g in CHANNEL_GROUPS.values() for c in g]
GROUP_OF = {c: g for g, chs in CHANNEL_GROUPS.items() for c in chs}
