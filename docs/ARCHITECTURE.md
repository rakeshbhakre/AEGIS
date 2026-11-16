# AEGIS architecture

This document is the short source-navigation guide for the repository. The
full project dossier is maintained separately from the source tree so the
repository README stays focused on onboarding and engineering decisions.

## Runtime flow

```text
source -> contract -> conditioner -> detectors -> event engine
       -> RCA/causal links -> safety guard -> evidence bundle -> API/UI
```

### 1. Sources

- [`sources.py`](../sources.py) exposes the synthetic `TwinSource`, bus
  delivery model, file replay, and SMAP loading.
- [`twin/spacecraft.py`](../twin/spacecraft.py) produces physics-shaped
  spacecraft telemetry and deterministic fault/data-pathology injections.

### 2. Contracts

[`contracts/models.py`](../contracts/models.py) defines the stable dataclass
boundary: `TelemetryFrame`, `QualityFlags`, `AnomalyEvent`, `CauseCandidate`,
`ActionProposal`, and `EvidenceBundle`.

The distinction between event time (`t`) and arrival time (`t_arr`) is
intentional. Transport disorder must not change the time used for physical
reasoning.

### 3. Conditioning and detection

- [`conditioner/stream.py`](../conditioner/stream.py) performs robust scaling,
  context removal, missing-data handling, and quality flags.
- [`detectors/models.py`](../detectors/models.py) combines MAD, the custom
  NumPy autoencoder, and Isolation Forest.
- [`detectors/events.py`](../detectors/events.py) turns frame/window scores
  into persistent events using calibration, smoothing, hysteresis, and
  attribution.

### 4. Reasoning and safety

- [`reasoning/rca.py`](../reasoning/rca.py) matches observed signatures to
  ontology faults and ranks candidate causes.
- [`reasoning/graph.py`](../reasoning/graph.py) generates lagged correlation
  links for supporting evidence.
- [`reasoning/guard.py`](../reasoning/guard.py) evaluates action safety.
- [`ontology/fault_graph.yaml`](../ontology/fault_graph.yaml) stores fault
  signatures and mitigations.
- [`ontology/actions.yaml`](../ontology/actions.yaml) is the allow-listed
  action library and precondition source.

### 5. Runtime and API

[`runtime/engine.py`](../runtime/engine.py) owns the streaming pipeline,
calibration, evidence assembly, operator labels, and edge export.

The API layer is deliberately thin at the integration boundary:

- [`api/settings.py`](../api/settings.py) reads environment configuration.
- [`api/app.py`](../api/app.py) creates the FastAPI application.
- [`api/server.py`](../api/server.py) owns the current process lifecycle,
  routes, and shared runtime state.
- [`api/websocket.py`](../api/websocket.py) registers the live stream.
- [`api/feeds.py`](../api/feeds.py) isolates public-feed HTTP access.
- [`api/ai_provider.py`](../api/ai_provider.py) isolates optional
  OpenAI-compatible explanations and the deterministic fallback narrative.

The process is intentionally single-instance for the demonstration. Moving to
multiple workers requires an explicit state store and subscriber fan-out
strategy; it is not safe to scale the current in-memory engine horizontally
without that design work.

### 6. Evaluation and edge path

- [`evalharness/run_bench.py`](../evalharness/run_bench.py) runs twin,
  data-fault, SMAP, and edge checks.
- [`evalharness/metrics.py`](../evalharness/metrics.py) matches event intervals
  and summarizes precision/recall-style measures.
- [`export/edge_bundle.py`](../export/edge_bundle.py) exports a compact NumPy
  bundle and checks ground/edge forward-pass parity.

## Design principles

1. Keep module boundaries expressed through dataclass contracts.
2. Treat data quality as part of the detection decision, not as an afterthought.
3. Keep fault/action knowledge editable in YAML.
4. Produce evidence and explicit safety blockers, not opaque predictions.
5. Keep optional network/AI integrations outside the core detector path.
6. Preserve benchmark evidence exactly; do not replace weak results with claims.

## Safe extension points

- Add a new telemetry source by implementing the source contract.
- Add a detector by contributing a calibrated score to the ensemble.
- Add a fault by updating the ontology and twin injection mapping together.
- Add an action only with explicit preconditions and guard tests.
- Add an external feed through `api/feeds.py`, with timeout and degradation
  behavior.
