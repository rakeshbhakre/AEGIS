# AEGIS Project: Complete Architecture and Working Guide

**Generated:** 19 October 2026  
**Project:** AEGIS spacecraft telemetry monitoring and mission operations console  
**Repository root:** `AEGIS/`

## 1. Executive summary

AEGIS is a Python-based, streaming spacecraft-health platform. It receives telemetry from either:

1. A physics-shaped synthetic spacecraft digital twin.
2. A replay of real satellite telemetry, primarily the NASA SMAP-style `.npy` datasets.

The same runtime pipeline is used for simulation, file replay, benchmarking, the browser console, and edge export. The pipeline turns raw telemetry into:

- Cleaned and robustly scaled channel vectors.
- Data-quality flags and a quality score.
- Multiple anomaly scores.
- Event-level anomaly records with persistence and hysteresis.
- Channel attributions and subsystem classification.
- Root-cause candidates from fault signatures, priors, and correlations.
- Guarded operational actions, categorized as auto-safe, propose, or advisory.
- JSON evidence bundles for operators and downstream tools.

The project is best classified as a **full-stack scientific/operations application**:

- **Backend:** Python runtime, FastAPI HTTP API, WebSocket stream.
- **Frontend:** static HTML/CSS/JavaScript mission console with Three.js visualization.
- **Scientific/ML core:** NumPy, scikit-learn Isolation Forest, custom NumPy MLP autoencoder.
- **Domain simulation:** synthetic LEO spacecraft model with fault and data-pathology injection.
- **Evaluation:** deterministic benchmark harness for health faults, data faults, SMAP replay, and edge parity.

## 2. What was used and applied

### Languages and runtime

- Python 3 source code with type hints, dataclasses, NumPy arrays, and generator/stream-oriented processing.
- HTML, CSS, and browser JavaScript for the mission console.
- YAML for the fault ontology and action library.
- JSON/JSONL for benchmark reports, exported state, labels, and feedback.
- CSV for labeled telemetry metadata.
- NPY/NPZ for telemetry arrays and the edge model bundle.

### Core libraries and technologies

| Area | Technology used | Where it is applied |
|---|---|---|
| Web API | FastAPI | `api/server.py` REST endpoints and application lifecycle |
| Streaming transport | WebSocket | `/ws` live evidence and status updates |
| ASGI runtime | Uvicorn | Server command documented in `api/server.py` |
| Numerical computing | NumPy | Arrays, robust statistics, windows, correlations, neural-network inference |
| Classical ML | scikit-learn IsolationForest | Slow-drift and multivariate window anomaly score |
| Neural network | Custom NumPy MLP | Small autoencoder trained and inferred without PyTorch/TensorFlow |
| Configuration/data | PyYAML | Fault graph and action ontology loading |
| Concurrency | `asyncio`, `asyncio.to_thread`, `ThreadPoolExecutor` | Non-blocking warm-up, tick loop, rebuild, training, and external-feed work |
| Visualization | Browser JavaScript and Three.js vendor asset | Mission-console charts and spacecraft scene |
| Spaceflight propagation | Optional `sgp4` | ISRO satellite position feed in the sky panel |
| Data formats | JSON, JSONL, CSV, NPY, NPZ | API payloads, labels, reports, telemetry, edge bundle |

### Engineering patterns applied

- Contract-first module boundaries using dataclasses in `contracts/models.py`.
- Single streaming engine used by console, headless demo, and benchmark paths.
- Robust statistics instead of assuming Gaussian telemetry.
- Explicit data-quality handling before anomaly detection.
- Event state machine with EMA smoothing, threshold hysteresis, persistence, and event merging.
- Split-conformal-style threshold calibration against normal-operation event peaks.
- Explainable root-cause output: matched, missed, and spurious channels.
- Human-in-the-loop action policy with deterministic safety preconditions.
- Fault ontology as editable YAML rather than hard-coded business logic.
- Physics-shaped synthetic data for repeatable fault injection and ground truth.
- Edge/ground parity through shared NumPy forward math and an NPZ export.
- Graceful degradation for missing optional dependencies and unavailable external feeds.

## 3. Repository structure

```text
AEGIS/
├── api/
│   ├── server.py                 FastAPI app, REST endpoints, WebSocket loop
│   └── static/
│       ├── index.html            Mission-console UI
│       ├── craft.js              Spacecraft visualization and UI behavior
│       └── vendor/three.min.js   Three.js browser dependency
├── conditioner/
│   └── stream.py                 Cleaning, scaling, context removal, quality flags
├── contracts/
│   └── models.py                 Dataclass contracts between modules
├── detectors/
│   ├── events.py                 Windows, event state machine, calibration
│   ├── models.py                 MAD, autoencoder, Isolation Forest, ensemble
│   └── nn.py                     Pure-NumPy MLP training and inference
├── evalharness/
│   ├── metrics.py                Event matching and summary metrics
│   └── run_bench.py              Twin/data-fault/SMAP/edge benchmark runner
├── export/
│   └── edge_bundle.py            NPZ edge bundle and parity verification
├── ingest/
│   └── __init__.py               Ingestion package boundary
├── ontology/
│   ├── actions.yaml              Vetted actions and preconditions
│   └── fault_graph.yaml          Fault signatures, priors, propagation, mitigations
├── reasoning/
│   ├── graph.py                  Lagged correlation and causal links
│   ├── guard.py                  Safety decision validator
│   ├── ontology.py               YAML ontology loader
│   └── rca.py                    Fault signature extraction and ranking
├── runtime/
│   └── engine.py                 Main pipeline orchestrator and evidence builder
├── tools/
│   ├── refresh.py                Refresh/recalibration helper
│   └── robustness_matrix.py      Robustness experiments
├── twin/
│   └── spacecraft.py             LEO spacecraft model and fault effects
├── data/
│   ├── SMAP/                     Training/test arrays and labels
│   ├── aegis_edge.npz            Exported edge model
│   ├── bench.json                Existing benchmark results
│   └── edge_report.json          Existing edge parity/latency report
├── sources.py                    Twin, bus, file replay, overlays, loaders
└── AEGIS_ARCHITECTURE_REPORT.md  This guide
```

Generated/cache directories such as `__pycache__` and `.git` are not runtime architecture components.

## 4. High-level architecture

```text
                 ┌──────────────────────────────┐
                 │ Browser mission console       │
                 │ index.html + craft.js         │
                 └──────────────┬───────────────┘
                                │ HTTP + WebSocket
                 ┌──────────────▼───────────────┐
                 │ FastAPI mission console       │
                 │ api/server.py                 │
                 │ warm-up, tick loop, controls  │
                 └──────────────┬───────────────┘
                                │
       ┌────────────────────────▼────────────────────────┐
       │ sources.py                                      │
       │ TwinSource + BusModel OR FileSource/SMAP replay │
       └────────────────────────┬────────────────────────┘
                                │ TelemetryFrame
                 ┌──────────────▼───────────────┐
                 │ Conditioner                  │
                 │ gap/impulse/stale/saturation │
                 │ delay flags + robust scaling  │
                 └──────────────┬───────────────┘
                                │ vector + QualityFlags
                 ┌──────────────▼───────────────┐
                 │ Detector ensemble             │
                 │ MAD + MLP AE + IsolationForest│
                 │ slow statistic + fusion       │
                 └──────────────┬───────────────┘
                                │ fused stream
                 ┌──────────────▼───────────────┐
                 │ EventEngine                   │
                 │ EMA + threshold + hysteresis  │
                 │ persistence + attribution     │
                 └──────────────┬───────────────┘
                                │ AnomalyEvent
                 ┌──────────────▼────────────────────────┐
                 │ Evidence and reasoning                 │
                 │ graph correlation + RCA + ontology     │
                 │ guard decision + provenance             │
                 └──────────────┬────────────────────────┘
                                │ EvidenceBundle
       ┌────────────────────────▼────────────────────────┐
       │ API clients, UI, labels, reports, edge export   │
       └─────────────────────────────────────────────────┘
```

## 5. End-to-end working

### 5.1 Source generation and replay

`TwinSource` produces 16 channels grouped into EPS, ADCS, THERMAL, COMM, PAYLOAD, and OBC. The `SpacecraftTwin` models orbit phase, eclipse, solar generation, power loads, battery SOC, voltage/current, thermal nodes, reaction-wheel behavior, communications signal, payload load, and CPU load.

`FaultInjector` overlays health faults and sensor/data faults. A fault has an ID, start time, duration, and a 30-tick onset ramp. This provides exact truth for benchmark matching.

`BusModel` is a separate arrival layer. It can introduce delay and out-of-order delivery while retaining event time in `TelemetryFrame.t`. Frames are reordered inside a bounded 30-second window. Frames that fall outside the window are counted as late drops instead of being silently re-stamped.

`FileSource` replays a `(T, C)` matrix and can overlay synthetic faults. `load_smap()` reads SMAP-style arrays and optional labeled intervals.

### 5.2 Contract boundary

`TelemetryFrame` is the input contract:

- `t`: spacecraft/event time used by physics and temporal reasoning.
- `t_arr`: arrival time, allowing delayed-link simulation.
- `values`: channel values.
- `flags`: source metadata and truth markers.

`QualityFlags` records imputation, impulse noise, stale channels, saturation, delay, late arrivals, and veto channels.

`AnomalyEvent` records event ID, time interval, subsystem group, score, conformal p-value, quality, and channel attributions.

`EvidenceBundle` combines the event with causal links, cause candidates, action proposals, and provenance.

### 5.3 Conditioning and sense-check

`Conditioner.step()` performs, per channel:

1. Read the value and source flags.
2. Fill missing values when a previous value exists, while marking the channel as imputed and vetoed.
3. Detect impulse noise with a five-sample median comparison.
4. Track stale/frozen values.
5. Detect named and generic saturation rails.
6. Detect delayed and late-beyond-window telemetry.
7. Update a running median/IQR scaler.
8. Optionally subtract an orbit-phase context baseline.
9. Return a clipped robust-scaled vector and `QualityFlags`.

The engine converts these flags into a quality score from 0 to 1. The score penalizes veto channels, imputation, stale values, saturation, delay, and late arrivals. This quality information is preserved in event evidence and used by the safety guard.

### 5.4 Warm-up and calibration

`Engine.warm()` consumes normal telemetry before operational monitoring:

- Fits robust channel statistics.
- Builds 48-frame windows.
- Fits MAD and Isolation Forest models.
- Optionally trains the NumPy autoencoder with stride-4 windows.
- Computes a stride-1 fused calibration stream.
- Replays a shadow event state machine.
- Builds a threshold curve for configurable false-positive budgets per day.

The API starts warm-up in a background thread so the event loop remains responsive. The console reports a warming/calibrating status until the engine is ready.

### 5.5 Detector ensemble

The detector layer uses several complementary signals:

1. **MAD detector:** median and median absolute deviation robust z-scores per channel. It supports channel validity masking to avoid quasi-constant telemetry dominating the score.
2. **Autoencoder:** a small MLP with shape `d → 96 → 48 → 96 → d`, tanh hidden layers, and linear output. It is trained using a custom NumPy optimizer and runs without a deep-learning framework.
3. **Isolation Forest:** scikit-learn model over window features: mean, standard deviation, slope, maximum difference, and nonlinear mean magnitude.
4. **Slow statistic:** long-horizon deviation for gradual faults.
5. **Fusion:** calibrated detector streams are normalized, clipped to ±40, and combined with fixed weights. If the autoencoder is unavailable, the ensemble reweights the remaining detectors.

### 5.6 Event detection

`EventEngine` turns the fused stream into operational events:

- EMA smoothing reduces one-frame noise.
- A calibrated threshold starts an event.
- A hysteresis threshold closes it.
- Minimum persistence avoids committing short spikes.
- The peak attribution vector is retained.
- Top contributing channels are converted to `ChannelAttrib`.
- Channels map to subsystem groups.
- `Conformist.p_value()` supplies an empirical event-level p-value.

This produces an event only after a sustained signal has been observed, rather than returning every anomalous frame.

### 5.7 Root-cause analysis

`reasoning/rca.py` extracts a signature for each event:

- Direction: up, down, or flat.
- Kind: jump, ramp, spike, or freeze.
- Robust attribution magnitude.
- Veto share from data-quality flags.

Each observed signature is compared against every fault in `fault_graph.yaml`. Scores reward matching direction and kind, partially reward one-sided matches, penalize missing expected channels and spurious channels, and incorporate ontology or feedback priors using a Bayesian-style log score.

`reasoning/graph.py` adds lagged cross-correlation links between event channels and compares the event window with a preceding baseline. Ontology propagation edges add prior strength to known relationships. The resulting `CausalLink` objects make the evidence explainable.

### 5.8 Safety guard and action policy

`reasoning/guard.py` is intentionally deterministic. It never allows an arbitrary action to be invented by the model. It only evaluates actions named in `ontology/actions.yaml`.

An auto-safe action requires:

- Event origin is `HEALTH`.
- Top cause probability is at least 0.60.
- At least two corroborating channels.
- Conformal p-value is below 0.10.
- Data quality is at least 0.55.
- Every action precondition passes against a telemetry-derived `Snapshot`.

If the gates do not pass, an action becomes a proposal/advisory or is blocked with explicit reasons. Data-fault causes are routed to sensor/ground actions and are not treated like physical spacecraft faults.

### 5.9 Evidence and operator feedback

`Engine` creates an `EvidenceBundle` for committed events and publishes it to WebSocket subscribers. The API exposes recent events and labels. Operator labels can feed a feedback-prior file used by RCA ranking on later runs.

## 6. API and user workflows

### Read endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/live` | Current telemetry, fused score, event marks, quality, active injections, and latest evidence |
| `GET /api/state` | Engine statistics and recent channel history |
| `GET /api/events` | Last 50 evidence bundles |
| `GET /api/quality` | Current bus/link quality configuration |
| `GET /api/bench` | Existing benchmark JSON |
| `GET /api/sky` | External space-weather and satellite-position feed |

### Write/control endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/quality` | Inject noise, missing data, delay, or out-of-order arrival |
| `POST /api/inject` | Inject a named spacecraft fault |
| `POST /api/inject_data` | Inject a data pathology such as gap or sensor issue |
| `POST /api/storm` | Inject multiple faults together |
| `POST /api/dial` | Change false-positive budget and threshold curve selection |
| `POST /api/labels` | Label an event and optionally record its fault |
| `POST /api/refresh` | Refresh/recalibrate engine state |
| `POST /api/train` | Train the autoencoder in the background |
| `POST /api/speed` | Control replay/twin speed and run/pause state |
| `POST /api/replay_file` | Switch to SMAP file replay |
| `POST /api/twin_mode` | Return to the synthetic twin |

### WebSocket

`/ws` sends status messages during warm-up and then streams evidence/status payloads. Each client receives its own bounded queue. The server removes a client queue on disconnect.

## 7. Domain model

### Telemetry channels

- **EPS:** battery voltage/current, SOC, solar current, heater current, battery temperature.
- **ADCS:** wheel speed/current/temperature and attitude error.
- **THERMAL:** panel and radiator temperature.
- **COMM:** signal strength.
- **PAYLOAD:** payload power and temperature.
- **OBC:** CPU load.

### Fault ontology

The current ontology contains eight principal fault/data-fault classes:

1. Heater controller stall/latch-up.
2. Battery cell capacity fade/sag.
3. Reaction-wheel bearing degradation.
4. Cell thermal excursion/runaway onset.
5. Payload power overrun.
6. Sensor drift.
7. Sensor stuck/frozen.
8. Sensor step bias.

Each fault contains an ID, display name, subsystem, prior, expected channel signature, propagation edges, and mitigation IDs.

## 8. Frontend architecture

The browser console is a static single-page UI. `index.html` defines the operations layout, styling, controls, telemetry cards, charts, event panels, and action/evidence views. `craft.js` drives live polling/WebSocket updates and the 3D spacecraft visualization using the bundled Three.js asset.

The UI is deliberately connected to operational controls rather than being a passive dashboard:

- Start/pause and speed control.
- Fault injection and multi-fault storm.
- Noise/missing/delay/out-of-order controls.
- Threshold/false-positive budget dial.
- Twin/file-replay mode switch.
- Event review and operator labels.
- Current spacecraft scene and external sky information.

## 9. Evaluation and existing evidence

The repository already contains generated results in `data/bench.json` and `data/edge_report.json`. These are evidence files, not assumptions.

### Twin benchmark

- 40 injected episodes across eight fault/data-fault classes.
- 29 detections.
- Unmatched false-positive rate: **1.044 per day**.
- Best detected class in the saved run: wheel friction at **1.0 detection rate**, 36-second median TTD, and top-1/top-3 RCA of 1.0.
- Saved run reports no false auto-actions on a wrong top cause.

### Data-pathology benchmark

- 28 injected data-fault cases.
- 2 health false alarms.
- 2 data-origin alerts.
- Saved rejection rate: **0.929**.

### SMAP replay

The saved SMAP section reports zero event F-score and zero recall for the listed satellites in that run. This is important: it means the SMAP evaluation path exists and generated a report, but the current saved result does not demonstrate successful SMAP event detection. The result should be treated as a validation gap rather than as proof of performance.

### Edge export

- Bundle size: **601,187 bytes**.
- Saved parity maximum relative difference: **0.0**.
- Edge inference p50: **0.077 ms**.
- Edge inference p99: **0.163 ms**.
- Edge score uses the same NumPy forward path as the ground runtime.

The separate benchmark summary reports a single-process engine step p50 of 13.48 ms and p99 of 22.84 ms, with a 294.6 MB peak RSS for the measured run. These measurements are workload-specific and should be rerun when deployment hardware or data rates change.

## 10. Startup and operating sequence

### Console startup

The module docstring documents:

```text
uvicorn aegis.api.server:app --host 0.0.0.0 --port 8890
```

At application startup:

1. A background warm-up task creates and calibrates the engine.
2. A background tick loop advances the active source.
3. The HTTP/WebSocket server stays responsive.
4. Once warm, the tick loop advances one or more event-time frames.
5. The engine publishes evidence to subscribers.

### Headless/benchmark sequence

1. Create a `TwinSource`.
2. Create an `Engine` with the source channel contract.
3. Warm on normal frames.
4. Inject faults after the warm-up period.
5. Call `Engine.step()` for each frame.
6. Match emitted events against injected truth intervals.
7. Summarize detection, TTD, RCA, guard, and false-positive metrics.

### Edge sequence

1. Warm an engine.
2. Export detector/calibration parameters to `data/aegis_edge.npz`.
3. Load them into `EdgeLite`.
4. Compare edge and ground scores over sampled windows.
5. Measure edge latency.
6. Persist the report and fail if parity exceeds the 2% assertion.

## 11. Important design decisions

### Why dataclasses instead of API models internally?

The runtime uses lightweight dataclasses and NumPy arrays for speed and predictable internal contracts. API-specific serialization occurs at the boundary.

### Why multiple detectors?

MAD is robust and explainable, the autoencoder captures reconstruction novelty, Isolation Forest catches engineered multivariate patterns, and the slow statistic catches drift. Their combination reduces dependence on one model family.

### Why preserve event time separately from arrival time?

Delayed or reordered telemetry must not change the physical chronology used for detection and causal reasoning. The bus layer simulates communication problems without corrupting the underlying event-time semantics.

### Why an explicit action ontology?

The safety-critical action surface must be bounded and auditable. The system can recommend or allow only actions declared in YAML and must provide reasons for blocked actions.

### Why a custom NumPy autoencoder?

It removes the need for a deep-learning runtime in the edge path and guarantees that the same `mlp_forward` function is used in ground and edge inference.

## 12. Risks, limitations, and follow-up areas

- The root directory does not contain a standard dependency manifest such as `requirements.txt` or `pyproject.toml`; runtime environments must therefore be documented separately before deployment.
- Several optional integrations are runtime-dependent, including `sgp4`, external public feeds, and scikit-learn.
- Some server paths intentionally catch exceptions to keep the console alive; operators should monitor the `_STATE["err"]` and external-feed error fields.
- The saved SMAP benchmark does not show useful recall and should be investigated before claiming real-telemetry readiness.
- The saved benchmark indicates that some injected faults have modest detection/RCA performance. Thresholds, signatures, and training data should be tuned with a representative telemetry corpus.
- The API process stores engine state in process memory; horizontal scaling would require an explicit state/session strategy and subscriber fan-out design.
- The browser currently consumes a broad operational API; authentication, authorization, and network isolation should be defined before exposing the console outside a trusted environment.
- The current report describes observed source behavior and saved evidence. It does not claim that every optional external feed is available in every environment.

## 13. File-by-file reading order

For a new engineer, use this order:

1. `contracts/models.py` — learn the data contracts and channel vocabulary.
2. `sources.py` — understand twin, bus, and replay inputs.
3. `conditioner/stream.py` — understand quality and scaling.
4. `detectors/models.py` and `detectors/events.py` — understand scoring and event commitment.
5. `runtime/engine.py` — follow orchestration and evidence assembly.
6. `ontology/fault_graph.yaml` and `ontology/actions.yaml` — learn domain knowledge and allowed actions.
7. `reasoning/rca.py`, `reasoning/graph.py`, `reasoning/guard.py` — understand explanation and safety.
8. `api/server.py` — understand runtime lifecycle and operator controls.
9. `api/static/index.html` and `api/static/craft.js` — understand the user experience.
10. `evalharness/run_bench.py`, `export/edge_bundle.py` — understand validation and edge packaging.

## 14. One-sentence architecture summary

**AEGIS is a contract-driven, event-time-preserving streaming telemetry platform that combines robust statistical detection, lightweight machine learning, explainable fault reasoning, and deterministic safety guards behind a FastAPI/WebSocket mission console, with repeatable twin/replay benchmarks and a parity-checked NumPy edge bundle.**

---

# 15. Deep-search verification addendum

This addendum was produced from the checked-out repository at commit
`55fe386 Core Changes` on 17 November 2026. It supplements the architecture
guide above with the requested report sections and distinguishes source-verified
facts from information that is not present in the repository.

## 15.1 Abstract

AEGIS is a real-time spacecraft telemetry health-monitoring and decision-support
system. It generates or replays telemetry, conditions noisy or delayed
measurements, combines statistical and machine-learning anomaly scores, groups
anomalies into events, ranks likely fault causes, and proposes only ontology
approved actions after deterministic safety checks. A FastAPI/WebSocket service
exposes the runtime to a browser mission console. A benchmark harness evaluates
synthetic faults, data-quality failures, SMAP replay, and edge inference.

## 15.2 Problem statement

Spacecraft telemetry is streaming, multivariate, noisy, and subject to missing,
stale, delayed, out-of-order, or physically implausible measurements. A useful
operations tool must distinguish a spacecraft-health fault from a data-pathology
fault, identify the affected subsystem, provide evidence for the conclusion, and
avoid unsafe automated action when confidence or data quality is inadequate.

The repository does not contain the original organizer wording. The problem
statement above is a technical interpretation derived from the implemented
contracts, digital twin, fault ontology, and benchmark harness.

## 15.3 Existing-system baseline

No separate legacy application or baseline system is included. The repository
contains the AEGIS implementation, an embedded Telemanom dataset subdirectory,
and generated benchmark artifacts. Therefore, a formal comparison with an
existing system cannot be claimed from source evidence.

## 15.4 Proposed system

The proposed implementation is a single-process, modular Python pipeline:

1. `TwinSource` or `FileSource` supplies `TelemetryFrame` objects.
2. `Conditioner` performs contextual normalization and quality assessment.
3. MAD, autoencoder, and Isolation Forest detectors produce complementary
   anomaly signals.
4. `EventEngine` applies smoothing, calibration, persistence, hysteresis, and
   attribution.
5. RCA uses ontology signatures, priors, and lagged correlation links.
6. The guard checks data quality, corroboration, confidence, cooldowns, and
   ontology preconditions before classifying actions.
7. `EvidenceBundle` is serialized for the UI, API clients, labels, and reports.

## 15.5 Objectives and scope

### Implemented objectives

- Detect anomalies in synthetic and replayed telemetry.
- Model spacecraft subsystems and inject known health/data faults.
- Preserve event time separately from arrival time.
- Score data quality before trusting detector output.
- Produce explainable channel attributions and candidate causes.
- Separate health-origin alerts from data-origin alerts.
- Generate safety-tiered action proposals.
- Provide live HTTP/WebSocket access and a browser console.
- Export a lightweight edge bundle and measure parity/latency.
- Produce repeatable benchmark summaries.

### In scope

The six modeled subsystem groups are EPS, ADCS, THERMAL, COMM, PAYLOAD, and
OBC. The contract currently defines 16 telemetry channels. The ontology defines
eight primary fault entries, including health and data-fault cases, plus a
separate set of data pathologies (`gap`, `stale`, `spike`, `saturate`).

### Out of scope or not demonstrated

- Flight-qualified spacecraft control.
- Autonomous spacecraft operation.
- Production multi-user deployment.
- Formal security certification.
- A persistent application database.
- A formally validated real-telemetry accuracy result.
- A complete cloud/deployment configuration.

## 15.6 Requirements traceability

| Requirement | Evidence | Status |
|---|---|---|
| Stream telemetry | `sources.py`, `runtime/engine.py` | Implemented |
| Handle quality faults | `conditioner/stream.py`, `twin/spacecraft.py` | Implemented |
| Detect anomalies | `detectors/models.py` | Implemented |
| Form events | `detectors/events.py` | Implemented |
| Explain causes | `reasoning/rca.py`, `reasoning/graph.py` | Implemented |
| Guard actions | `reasoning/guard.py`, `ontology/actions.yaml` | Implemented |
| Live API | `api/server.py` | Implemented |
| 3D console | `api/static/index.html`, `api/static/craft.js` | Implemented |
| Benchmarking | `evalharness/run_bench.py`, `data/bench.json` | Implemented |
| Edge export | `export/edge_bundle.py`, `data/aegis_edge.npz` | Implemented |
| AuthN/AuthZ | No auth middleware or identity integration found | Not present |
| Database persistence | No DB driver/schema/migrations found | Not present |

## 15.7 Database and persistence design

No relational or document database is present. No ORM, migrations, database
connection string, or schema files were found. Runtime state is held in process
memory. Files are used for persisted artifacts:

- `data/bench.json`: benchmark results.
- `data/edge_report.json`: edge parity and latency evidence.
- `data/aegis_edge.npz`: exported model parameters.
- `data/labels.jsonl`: label/feedback records.
- `ontology/*.yaml`: editable fault/action knowledge.
- NPY/CSV files: telemetry and labels.

This is appropriate for a local demonstration and deterministic benchmark, but
not sufficient by itself for durable multi-user operations, audit retention, or
horizontal scaling.

## 15.8 API inventory

### Read endpoints

| Route | Purpose |
|---|---|
| `GET /` | Serves the mission console |
| `GET /api/live` | Current telemetry, score, event, quality, and evidence |
| `GET /api/state` | Engine statistics and recent channel history |
| `GET /api/events` | Recent evidence bundles |
| `GET /api/quality` | Current quality configuration |
| `GET /api/bench` | Saved benchmark JSON |
| `GET /api/fleet` | Fleet/space-weather view |
| `GET /api/passes` | Satellite pass calculation |
| `GET /api/ai/status` | AI provider availability |
| `GET /api/crew` | Crew/mission information |
| `GET /api/sky` | External sky/space-weather feed |

### Control endpoints

| Route | Purpose |
|---|---|
| `POST /api/quality` | Change simulated noise/missing-data quality |
| `POST /api/inject` | Inject a named spacecraft fault |
| `POST /api/inject_data` | Inject a data pathology |
| `POST /api/storm` | Inject multiple faults |
| `POST /api/dial` | Adjust false-positive budget/threshold behavior |
| `POST /api/labels` | Label an event and optionally record its fault |
| `POST /api/refresh` | Rebuild/recalibrate runtime state |
| `POST /api/train` | Train the autoencoder |
| `POST /api/speed` | Control run/pause/replay speed |
| `POST /api/replay_file` | Switch to file replay |
| `POST /api/twin_mode` | Return to synthetic twin mode |
| `POST /api/ai/explain` | Request an optional AI-generated explanation |

### Real-time channel

`WS /ws` publishes live telemetry/evidence messages to connected browser
clients. The server also uses a background warm-up/tick loop and thread
execution for blocking operations.

## 15.9 Data model

The internal contract model is dataclass-based rather than database-backed:

- `TelemetryFrame`: event time, values, flags, arrival time.
- `QualityFlags`: imputation, impulse, stale, saturation, delay, late arrival,
  veto channels, and delay duration.
- `AnomalyEvent`: interval, subsystem group, score, conformal p-value, quality,
  attribution, origin, and veto reasons.
- `CausalLink`: source/destination channel, lag, strength, and prior.
- `CauseCandidate`: fault ID, probability-style rank, matched/missed/spurious
  signature channels.
- `ActionProposal`: action, tier, allowed state, safety reasons, blockers, and
  effect.
- `EvidenceBundle`: event plus causal links, causes, actions, and provenance.

The explicit event-time/arrival-time distinction is central: out-of-order
delivery is modeled as a transport issue while temporal reasoning remains based
on spacecraft event time.

## 15.10 Algorithms and AI/ML

### Conditioning

`RobustScaler` maintains streaming robust statistics. `Conditioner` handles
missing values, impulses, stale values, saturation, delayed values, and
context-baseline removal. The goal is to prevent known data defects from being
mistaken for physical faults.

### Detectors

- MAD detector: robust channel-level deviation and attribution.
- NumPy autoencoder: custom MLP trained with NumPy/Adam and scored by
  reconstruction error.
- Isolation Forest: scikit-learn multivariate slow-drift detector.
- Ensemble: calibrated score fusion across the detector outputs.

The implementation documents an inference path that is NumPy-only after fitting,
which enables the edge bundle to share the forward computation.

### Event detection

The event engine applies EMA smoothing, a calibrated threshold/conformal
signal, persistence, hysteresis, event merging, and force-close behavior. This
turns frame-level scores into operator-meaningful anomaly intervals.

### RCA and explainability

RCA extracts direction/timescale signatures, compares them with YAML fault
signatures, applies priors/feedback, and returns ranked candidates. The graph
module calculates lagged correlation links during/around events. Evidence
contains matched, missed, and spurious channels rather than only a label.

### Safety decision layer

Actions are loaded from `ontology/actions.yaml`. The guard verifies action tier,
preconditions, quality, corroboration, confidence, and cooldown-related state.
The result is `auto_safe`, `propose`, or `advisory`, with explicit reasons or
blockers. Data-fault causes are routed toward sensor/ground actions instead of
physical actuation.

## 15.11 External integrations

The server includes keyless/public-feed integrations for ISS/space-weather/TLE
and satellite pass data, including WhereTheISS, CCMC, NOAA/SWPC, CelesTrak,
and other public mirrors visible in `api/server.py`. Optional AI provider
configuration supports local or remote OpenAI-compatible-style endpoints; API
keys are read from environment variables when configured.

These integrations are optional operational enrichments, not required for the
core twin/detector pipeline. Their availability depends on network access and
the upstream service contracts.

## 15.12 Security assessment

### Positive controls

- API-key values are intended to come from environment variables.
- The action vocabulary is allow-listed in YAML.
- Action preconditions are evaluated deterministically.
- The AI explanation path is separate from the core deterministic detector and
  guard.
- No credentials were found in the repository during this review.

### Important gaps

- No authentication or authorization middleware was found.
- Control endpoints can inject faults, alter quality, train, refresh, and
  change runtime behavior.
- The application is designed as a trusted/local console, not an Internet-facing
  service.
- External HTTP calls require timeout/error handling and should be isolated or
  cached for production use.
- AI-generated narrative should never be treated as an actuation authority.
- CORS, rate limits, audit logging, TLS termination, and secret-management
  policy are not represented in the repository.

## 15.13 Testing and validation evidence

### Static validation performed for this dossier

`python -m compileall -q .` completed successfully on the checked-out project.
No standard unit-test directory or test runner configuration was found.

### Existing benchmark evidence

The committed `data/bench.json` reports:

- Twin: 40 injected episodes and 29 detections.
- Twin unmatched false positives: 1.044/day.
- Best listed detection rate: `wheel_friction` at 1.0.
- Lowest listed health detection rate: `payload_overload` at 0.4.
- Data-fault injection: 28 cases, 92.9% rejection rate, two health false
  alarms, and two data-origin alerts.
- SMAP summary: mean event F-score, precision, and recall are all 0.0 in the
  saved run; this must not be presented as real-telemetry readiness.
- Saved runtime step timing: p50 13.48 ms, p99 22.84 ms, peak RSS 294.6 MB.
- Edge inference: p50 0.077 ms, p99 0.163 ms.
- Edge bundle size: 601,187 bytes.
- Edge ground/edge parity: maximum relative difference 0.0 in the saved report.

These are saved-run measurements, not universal product guarantees. They should
be rerun on the target machine and with a representative dataset before being
used as current performance claims.

## 15.14 Screenshots and diagrams

The repository contains the browser console implementation but no committed
screenshots, screen recording, PPT, or diagram image files. The architecture
guide above contains a text architecture diagram. The following data-flow
representation is source-grounded:

```text
TwinSource / FileSource
        |
        v
TelemetryFrame (event time + arrival time)
        |
        v
Conditioner -> QualityFlags + scaled vectors
        |
        v
MAD + NumPy AE + Isolation Forest
        |
        v
Ensemble -> EventEngine
        |
        v
AnomalyEvent -> RCA + lagged causal graph
        |
        v
Safety Guard + YAML action ontology
        |
        v
EvidenceBundle -> REST / WebSocket / browser / reports / edge tools
```

Screenshots must be captured from the running application if they are needed
for a final presentation or LinkedIn post.

## 15.15 Deployment and run information

The source docstring identifies the main command:

```text
uvicorn aegis.api.server:app --host 0.0.0.0 --port 8890
```

The repository does not include a root `requirements.txt`, `pyproject.toml`,
Dockerfile, CI workflow, cloud resource template, or deployment manifest.
The embedded Telemanom dataset has its own historical requirements file, which
is not a complete dependency manifest for the AEGIS application.

## 15.16 Limitations and future work

### Verified limitations

- SMAP saved results show no successful event recall.
- Some synthetic faults have modest detection/RCA performance.
- The API keeps live state in one process.
- There is no authentication/authorization layer.
- External feeds are network-dependent.
- There is no durable database or audit store.
- There is no formal automated test suite in the repository.
- The project has no documented production deployment topology.

### Sensible future scope

- Add a pinned dependency manifest and reproducible environment.
- Add unit, contract, integration, and browser smoke tests.
- Improve SMAP preprocessing, labeling alignment, calibration, and evaluation.
- Add persistent event/audit storage and a multi-user session model.
- Add authentication, authorization, TLS, rate limits, and control auditing.
- Add observability for latency, queue depth, feed failures, and model drift.
- Replace public-feed direct calls with adapters, caching, and circuit breakers.
- Validate edge deployment on an actual constrained target.
- Introduce versioned model/ontology artifacts and reproducible benchmark runs.

## 15.17 Missing source materials requested by the user

The following cannot be reconstructed from the repository and must be supplied
by the project team if they are to appear as facts in an interview dossier:

- Original hackathon problem statement, theme, rules, tracks, judging criteria,
  time limit, and required deliverables.
- Original pitch deck, poster, demo script, and recording.
- Official result announcement, certificate, judge feedback, and questions.
- The team's original idea, pivots, time pressure, sacrifices, and last-minute
  changes.
- Rakesh's personally implemented files/features and contribution percentage.
- Swapnali's and Omkar's actual ownership areas.
- The actual reason behind each technology choice when not recorded in source.
- A live deployment URL and safe demo credentials.

No claim about those items should be added to a public post until confirmed by
the team or an official artifact.

## 15.18 Source index

Primary source files reviewed:

- `api/server.py` — FastAPI lifecycle, REST routes, WebSocket, external feeds,
  optional AI explanation.
- `api/static/index.html`, `api/static/craft.js` — browser mission console.
- `contracts/models.py` — internal contracts and channel taxonomy.
- `sources.py` — twin, bus, replay, and overlay sources.
- `conditioner/stream.py` — streaming conditioning and quality.
- `detectors/models.py`, `detectors/nn.py`, `detectors/events.py` — detection,
  neural network, calibration, and event logic.
- `runtime/engine.py` — orchestration and evidence assembly.
- `reasoning/rca.py`, `reasoning/graph.py`, `reasoning/guard.py`,
  `reasoning/ontology.py` — explainability and safe action logic.
- `twin/spacecraft.py` — physics-shaped simulation and injection.
- `evalharness/run_bench.py`, `evalharness/metrics.py` — evaluation.
- `export/edge_bundle.py` — edge export/parity.
- `ontology/actions.yaml`, `ontology/fault_graph.yaml` — domain knowledge.
- `data/bench.json`, `data/edge_report.json` — saved measurements.

## 15.19 Final recruiter-facing technical summary

AEGIS demonstrates a broad engineering profile: Python backend development,
real-time streaming, numerical computing, applied machine learning, scientific
simulation, explainable decision systems, browser visualization, benchmark
design, and edge inference. The strongest truthful ownership narrative is that
Rakesh directed the product/architecture and integrated an AI-agent-assisted
implementation into a working, evaluated system. Personal code ownership,
team division, and hackathon context still require direct confirmation.

