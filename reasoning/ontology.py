"""Ontology loader: fault graph + action library (YAML is the source of truth)."""
from __future__ import annotations
import os
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Ontology:
    def __init__(self, fault_file=None, action_file=None):
        with open(fault_file or os.path.join(HERE, "ontology", "fault_graph.yaml")) as f:
            fg = yaml.safe_load(f)
        with open(action_file or os.path.join(HERE, "ontology", "actions.yaml")) as f:
            ac = yaml.safe_load(f)
        self.faults = {f["id"]: f for f in fg["faults"]}
        self.actions = ac["actions"]
        self.priors = {f["id"]: float(f.get("prior", 0.1)) for f in fg["faults"]}
        # propagation edges: (a,b)->lag and set for prior boost
        self.prior_edges: dict[tuple, float] = {}
        for f in fg["faults"]:
            for a, b, lag in (f.get("propagation") or []):
                self.prior_edges[tuple(sorted((a, b)))] = max(
                    self.prior_edges.get(tuple(sorted((a, b))), 0.0), 1.0)

    def signature(self, fid: str) -> dict:
        return self.faults[fid]["signature"]

    def fault_ids(self):
        return list(self.faults)

    def is_data_fault(self, fid: str) -> bool:
        return bool(self.faults[fid].get("data_fault"))

    def mitigations(self, fid: str) -> list:
        return self.faults[fid].get("mitigations") or []


_ONT = None


def get() -> Ontology:
    global _ONT
    if _ONT is None:
        _ONT = Ontology()
    return _ONT
