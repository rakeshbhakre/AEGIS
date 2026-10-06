"""F7 Guardian — vetted action library + deterministic safety validator.

AUTO-SAFE fires only if: origin HEALTH, dominant cause p≥0.6, ≥2 corroborating
channels, conformal p < α, action tier auto_safe, and ALL preconditions pass
against the live state snapshot. Everything else degrades to PROPOSE/ADVISORY
with explicit reasons. The validator is a pure function (unit-testable,
auditable). Plan §4.5.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from ..contracts.models import ActionProposal, AnomalyEvent, CauseCandidate

OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
       "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b}


@dataclass
class Snapshot:
    """Telemetry-derived safety state at decision time."""
    fields: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)   # action_id -> last fired sim-time
    horizon_s: float = 0.0

    def get(self, k, default=None):
        return self.fields.get(k, default)


def state_from_values(vals: dict, cooldown_s: float = 600.0, last_actions: dict | None = None):
    v = vals.get("battery_voltage", 8.0)
    solar_w = vals.get("solar_current", 0.0) * v
    load_w = vals.get("battery_current", 0.0) * v + vals.get("payload_power", 0.0)
    return Snapshot(fields={
        "power_balance_w": round(solar_w - load_w, 2),
        "battery_temp_c": round(vals.get("battery_temp", 15.0), 1),
        "soc_pct": round(vals.get("soc", 70.0), 1),
        "wheel_temp_c": round(vals.get("wheel_temp", 25.0), 1),
        "solar_output_w": round(solar_w, 2),
        "eclipse_minutes_left": 45.0 if vals.get("solar_current", 5.0) < 2.0 else 5.0,
        "heater_cmd_cooldown_s": cooldown_s,
    }, history=last_actions or {})


def check(action_id: str, spec: dict, snap: Snapshot) -> tuple[list, list]:
    why, blocked = [], []
    for pc in spec.get("preconditions", []):
        f = snap.get(pc["field"])
        ok_expr = f is not None
        if ok_expr:
            try:
                ok_expr = bool(OPS[pc["op"]](f, pc["value"]))
            except Exception:
                ok_expr = False
        line = f"{pc['field']}={f if f is not None else 'missing'} {pc['op']} {pc['value']}"
        if ok_expr:
            why.append("PASS " + line)
        else:
            blocked.append("FAIL " + line)
    if snap.history.get(action_id) is not None and \
            snap.history.get("min_gap", 0) and snap.history[action_id] > 0:
        blocked.append("cooldown active for this action")
    return why, blocked


def decide(event: AnomalyEvent, causes: list[CauseCandidate], snap: Snapshot | None,
           ont) -> list[ActionProposal]:
    """Vetted-library proposals with deterministic safety validation."""
    top = causes[0] if causes else None
    if snap is None:
        snap = Snapshot()
    if top is None:
        return [ActionProposal(action_id="ground_review", name="Open anomaly review ticket",
                               tier="propose", allowed=True,
                               why_safe=["no cause produced — human review is the only safe path"])]
    corroborated = sum(1 for a in event.attrib if a.z >= 0.8) >= 2
    gates = {
        "health-origin": event.origin == "HEALTH",
        "top confidence score >= 0.60": top.p >= 0.60,
        ">=2 corroborating channels": corroborated,
        "conformal p < 0.10": event.conformal_p < 0.10,
        "data quality >= 0.55": getattr(event, "quality", 1.0) >= 0.55,
    }
    out: list[ActionProposal] = []
    for aid in ont.mitigations(top.fault_id):
        spec = ont.actions.get(aid)
        if spec is None:
            continue
        why, blocked = check(aid, spec, snap)
        auto_eligible = spec["tier"] == "auto_safe" and all(gates.values()) and not blocked
        if auto_eligible:
            out.append(ActionProposal(
                aid, spec["name"], "auto_safe", True,
                why_safe=["inside vetted action library"] + why +
                         [f"GATE PASS {k}" for k in gates],
                effect=spec.get("effect", "")))
        else:
            gate_fails = [k for k, v in gates.items() if not v]
            why_safe = []
            if not blocked:
                why_safe = (["inside vetted action library (operator-decided)"] + why
                            + [f"GATE HOLD {k}" for k in gate_fails])
            out.append(ActionProposal(
                aid, spec["name"],
                spec["tier"] if spec["tier"] != "auto_safe" else "propose",
                not blocked,
                why_safe=why_safe,
                blocked_by=blocked,
                effect=spec.get("effect", "")))
    return out
