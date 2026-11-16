import asyncio
import unittest

import numpy as np

from AEGIS.api import server
from AEGIS.contracts.models import (
    ActionProposal,
    AnomalyEvent,
    CauseCandidate,
    ChannelAttrib,
    EvidenceBundle,
)
from AEGIS.detectors.events import EventEngine
from AEGIS.reasoning.guard import Snapshot, decide
from AEGIS.reasoning.ontology import get


class ContractTests(unittest.TestCase):
    def test_evidence_bundle_serializes_nested_contracts(self):
        bundle = EvidenceBundle(
            event=AnomalyEvent(
                id="E00001", t0=1, t1=2, group="EPS", score=4,
                conformal_p=0.01,
                attrib=[ChannelAttrib("battery_temp", 2.1, "up", "ramp", 1)],
            ),
            causes=[CauseCandidate("heater_stall", "Heater stall", 0.8)],
            actions=[ActionProposal("ground_review", "Review", "propose", True)],
        )
        payload = bundle.to_dict()
        self.assertEqual(payload["event"]["id"], "E00001")
        self.assertEqual(payload["causes"][0]["fault_id"], "heater_stall")
        self.assertEqual(payload["actions"][0]["allowed"], True)


class DetectorTests(unittest.TestCase):
    def test_persistent_signal_closes_as_event(self):
        engine = EventEngine(["a", "b"], {"a": "EPS", "b": "EPS"})
        engine.conf.theta = 1.0
        emitted = None
        for t in range(20):
            emitted = engine.step(t, 10.0, np.array([2.0, 1.2]), None, None) or emitted
        for t in range(20, 50):
            emitted = engine.step(t, 0.0, np.array([0.0, 0.0]), None, None) or emitted
        self.assertIsNotNone(emitted)
        self.assertEqual(emitted.id, "E00001")
        self.assertEqual(emitted.group, "EPS")


class SafetyTests(unittest.TestCase):
    def test_auto_safe_action_is_held_when_gate_fails(self):
        event = AnomalyEvent(
            id="E1", t0=0, t1=10, group="EPS", score=8, conformal_p=0.01,
            attrib=[ChannelAttrib("battery_temp", 2, "up", "ramp", 0),
                    ChannelAttrib("heater_current", 1, "down", "ramp", 0)],
        )
        cause = CauseCandidate("heater_stall", "Heater stall", 0.9)
        proposals = decide(event, [cause], Snapshot(), get())
        self.assertTrue(proposals)
        heater_action = next(a for a in proposals if a.action_id == "switch_heater_b")
        self.assertEqual(heater_action.tier, "propose")


class ApiTests(unittest.TestCase):
    def test_health_endpoint_reports_process_health(self):
        self.assertEqual(asyncio.run(server.health()), {"ok": True, "warming": True})

    def test_invalid_fault_control_returns_client_error(self):
        response = asyncio.run(server.inject({"fault": "not_a_known_fault"}))
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.body is None)


if __name__ == "__main__":
    unittest.main()
