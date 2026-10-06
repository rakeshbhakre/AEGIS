"""FastAPI mission console backend (F8). One process: engine + twin + WS.
Run:  uvicorn aegis.api.server:app --host 0.0.0.0 --port 8890
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import math
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..runtime.engine import Engine
from ..sources import TwinSource

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="AEGIS mission console", version="0.1")
src = TwinSource(seed=int(os.environ.get("AEGIS_SEED", "7")))
eng = None


def _new_engine(source, warm_n, labels=True):
    e = Engine(source.channels, source.group_of, train_ae=False,
               day_frames=getattr(source, "day_frames", 3600),
               labels_path=os.path.join(DATA, "labels.jsonl") if labels else None)
    e.bus_window = getattr(getattr(source, "bus", None), "window", 30.0)
    got = e.load_pretrained(os.path.join(DATA, "aegis_edge.npz"))
    if got:
        _STATE["pretrained"] = f"AE restored from aegis_edge.npz ({got['ae_layers']} layers, err-cal={got['err_cal']})"
    return e


def _rebuild_blocking(source, warm_n):
    global eng
    _STATE["rebuilding"] = True
    e = _new_engine(source, warm_n)
    frames = [source.frame(i) if isinstance(source, TwinSource) else source.frame()
              for i in range(warm_n)]
    frames = [f for f in frames if f]
    e.warm(frames, log=lambda m: _STATE.update(log=m))
    eng = e
    _K["n"] = warm_n
    _STATE["rebuilding"] = False
    _STATE["running"] = True
_STATE = {"warm": False, "clients": [], "speed": 20, "running": True,
          "mode": "twin", "file_src": None, "train_job": None}
_K = {"n": 0}


def _warm_blocking():
    global eng
    t0 = time.time()
    _rebuild_blocking(src, 7400)
    _STATE["warm"] = True
    _STATE["warm_s"] = round(time.time() - t0, 1)


@app.on_event("startup")
async def startup():
    asyncio.create_task(asyncio.to_thread(_warm_blocking))
    asyncio.create_task(_tick_loop())


def _advance_one():
    source = _current_source()
    if getattr(source, "kind", "") == "twin":
        # bus-aware: one tick may deliver 0..n frames (late arrivals reorder by
        # EVENT time; frames beyond the reorder window are counted, never stamped)
        frs = source.frames(_K["n"])
        if not frs:
            _K["n"] += 1
            return
        for fr in frs:
            eng.step(fr)
    else:
        fr = source.frame(_K["n"])
        if fr is None:
            _STATE["running"] = False                # file replay exhausted
            return
        eng.step(fr)
    _K["n"] += 1


async def _tick_loop():
    while True:
        if _STATE["warm"] and _STATE["running"] and not _STATE.get("rebuilding"):
            try:
                await asyncio.to_thread(_advance_one)   # keep the loop free for HTTP
            except Exception as e:                      # console stays alive on error
                _STATE["err"] = repr(e)
            await asyncio.sleep(1.0 / _STATE["speed"])
        else:
            await asyncio.sleep(0.25)


def _busy_response():
    return {"warming": True,
            "note": _STATE.get("log") or ("rebuilding engine…" if _STATE.get("rebuilding")
                                          else "calibrating on first orbits")}


def _current_source():
    return _STATE["file_src"] if _STATE["mode"] == "file" and _STATE["file_src"] else src


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    # wait for engine warmup before subscribing
    while eng is None:
        await ws.send_json({"type": "status", "stats": {"warming": True, "note": "engine calibrating…"}})
        await asyncio.sleep(1)
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    eng.subscribers.append(q)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=0.5)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                st = eng.stats() | {"warm": _STATE["warm"], "k": _K["n"],
                                     "mode": _STATE["mode"], "speed": _STATE["speed"]}
                await ws.send_json({"type": "status", "stats": st})
    except WebSocketDisconnect:
        pass
    finally:
        if q in eng.subscribers:
            eng.subscribers.remove(q)


def _recent(n=600):
    import numpy as np
    H = eng.history[-n:] if len(eng.history) else np.zeros((0, len(src.channels)))
    R = eng.raw_hist[-n:] if len(eng.raw_hist) else H
    sc = eng.fuse_stream[-n:]
    return {
        "channels": list(src.channels),
        "t": [int(_K["n"] - len(sc) + i) for i in range(len(sc))],
        "scaled": [[round(float(v), 3) for v in H[:, i]] for i in range(H.shape[1])] if H.shape[0] else [],
        "raw": [[round(float(v), 3) for v in R[:, i]] for i in range(R.shape[1])] if R.shape[0] else [],
        "fuse": [round(float(x), 3) for x in sc],
        "theta": eng.conf.theta,
    }


def _san(o):
    """JSON-safe: drop NaN/Inf from any payload (telemetry gaps produce them)."""
    import math as _m
    if isinstance(o, float):
        return o if _m.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    return o


@app.get("/api/live")
async def live():
    if eng is None or not _STATE["warm"]:
        return _busy_response()
    import numpy as np
    tail = eng.fuse_stream[-480:]
    marks = [{"t": int(b.event.t0), "o": b.event.origin} for b in eng.events[-24:]]
    z = eng.mad.z(eng.history[-1:])[0] if len(eng.history) else None
    vals = {}
    for i, c in enumerate(src.channels if _STATE["mode"] == "twin" else eng.channels):
        r = eng.raw_hist[-1, i] if len(eng.raw_hist) else float("nan")
        vals[c] = None if not np.isfinite(r) else round(float(r), 3)
    # --- scene-driving extras (3D craft) ---------------------------------
    day = float(getattr(eng, "day_frames", 3600) or 3600)
    phase = (_K["n"] % day) / day
    active, dp = [], []
    if _STATE["mode"] == "twin":
        try:
            for f in src.inj.faults:
                if f.start <= _K["n"] < f.start + f.dur:
                    active.append(f.fid)
            for ch, d in src.inj.pathology.items():
                if d.get("start", 1 << 30) <= _K["n"] <= d.get("until", -1):
                    dp.append([ch, d["kind"]])
        except Exception:
            pass
    last = None
    if eng.events:
        b = eng.events[-1]
        last = {"id": b.event.id, "group": b.event.group, "origin": b.event.origin,
                "parts": [a.channel for a in b.event.attrib[:5]],
                "cause": b.causes[0].fault_id if b.causes else None,
                "live": (_K["n"] - b.event.t1) < 120,
                "actions": [a.action_id for a in b.actions if a.allowed][:4]}
    return _san({"warming": False, "k": _K["n"], "mode": _STATE["mode"],
            "speed": _STATE["speed"], "running": _STATE["running"],
            "theta": round(eng.conf.theta, 3), "fp_budget": eng.conf.fp_budget,
            "lat_p99": eng.stats()["lat_p99"], "ae": bool(eng.ae.W_),
        "quality": {"now": round(eng.qf_q[-1], 3) if getattr(eng, "qf_q", None) else 1.0,
                    "mean200": round(float(sum(eng.qf_q[-200:]) / max(1, len(eng.qf_q[-200:]))), 3) if getattr(eng, "qf_q", None) else 1.0,
                    "bus": src.bus.state()},
            "fuse": [round(float(v), 3) for v in tail],
            "channels": list(eng.channels),
            "z": ([round(float(v), 2) for v in z] if z is not None else []),
            "values": vals, "marks": marks, "phase": round(phase, 5),
            "active": active, "dp": dp, "last": last,
            "events_n": eng.total_events})


@app.get("/api/state")
async def state():
    if eng is None or not _STATE["warm"]:
        return _busy_response()
    import numpy as np
    vals = {c: (float(eng.raw_hist[-1, i]) if len(eng.raw_hist) and not np.isnan(eng.raw_hist[-1, i]) else None)
            for i, c in enumerate(src.channels)}
    return _san({"values": vals,
            "stats": eng.stats() | {"warm": _STATE["warm"], "k": _K["n"],
                                    "mode": _STATE["mode"], "speed": _STATE["speed"],
                                    "log": _STATE.get("log")},
            "recent": _recent()})


@app.get("/api/events")
async def events():
    if eng is None:
        return []
    return _san([b.to_dict() for b in eng.events[-50:]])


@app.get("/api/quality")
async def quality_get():
    bus = getattr(src, "bus", None)
    inj = getattr(src, "inj", None)
    return {"bus": bus.state() if bus else {},
            "link": {"noise_gain": inj.noise_gain, "missing_rate": inj.missing_rate} if inj else {},
            "engine_quality_now": round(eng.qf_q[-1], 3) if (eng and getattr(eng, "qf_q", None)) else None}


@app.post("/api/quality")
async def quality_set(body: dict):
    """Live data-quality injection for the judge demo (organiser §12/§35):
    noise & missing hit the twin's measurement path; delay & out-of-order hit
    the bus (arrival layer). Event times are never altered — that is the point."""
    out = {}
    if "reset" in body and body["reset"]:
        out["link"] = src.inj.set_quality(noise=1.0, missing=0.0)
        out["bus"] = src.bus.set(0.0, 0.0, 0.0)
        return out
    if any(k in body for k in ("noise", "missing")):
        _m = body.get("missing")
        if _m is not None and float(_m) > 1.0:      # tolerate "25" meaning 25 %
            _m = float(_m) / 100.0
        out["link"] = src.inj.set_quality(noise=body.get("noise"), missing=_m)
    if any(k in body for k in ("delay_prob", "delay_max_s", "swap_prob")):
        out["bus"] = src.bus.set(body.get("delay_prob"), body.get("delay_max_s"), body.get("swap_prob"))
    return out


@app.post("/api/inject")
async def inject(body: dict):
    fid = body.get("fault", "heater_stall")
    dur = int(body.get("dur", 420))
    if _STATE["mode"] == "twin":
        try:
            src.inject(fid, at=_K["n"] + 5, dur=dur)
        except Exception as e:
            return JSONResponse({"ok": False, "err": repr(e)}, status_code=400)
    return {"ok": True, "fault": fid, "at": _K["n"] + 5, "dur": dur}


@app.post("/api/inject_data")
async def inject_data(body: dict):
    ch = body.get("channel", "battery_temp")
    kind = body.get("kind", "gap")
    dur = int(body.get("dur", 150))
    try:
        src.inject_data(ch or "battery_temp", kind, at=_K["n"] + 3, dur=dur)
    except Exception as e:
        return JSONResponse({"ok": False, "err": repr(e)}, status_code=400)
    return {"ok": True}


@app.post("/api/storm")
async def storm(body: dict):
    if eng is None or not _STATE["warm"]:
        return JSONResponse({"ok": False, "err": "engine still calibrating"}, status_code=409)
    fids = body.get("faults", ["heater_stall", "cell_sag", "sensor_bias"])
    src.storm(fids, _K["n"] + 5)
    return {"ok": True}


@app.post("/api/dial")
async def dial(body: dict):
    if eng is None or not _STATE["warm"]:
        return JSONResponse({"ok": False, "err": "calibrating"}, status_code=409)
    th = eng.set_fp_budget(float(body.get("fp_per_day", 2)))
    return {"theta": th, "curve": eng.conf.curve}


@app.post("/api/labels")
async def labels(body: dict):
    return eng.label(body["event"], body.get("verdict", "confirm"), body.get("fault"))


@app.post("/api/refresh")
async def refresh():
    await asyncio.to_thread(eng.refresh)
    return {"theta": eng.conf.theta, "curve": eng.conf.curve}


@app.post("/api/train")
async def train():
    """Optionally train the AE in the background and recalibrate."""
    def job():
        import numpy as np
        Xw = eng._windows(eng.history, stride=4)
        from aegis.detectors.models import AEDetector
        ae = AEDetector()
        ae.fit(Xw, log=lambda m: _STATE.update(trainlog=m))
        eng.ae = ae
        eng.ens.ae = ae
        eng._warm_done = False
        frames = None
        eng.refresh()
        eng._warm_done = True
        _STATE["trained"] = True
    asyncio.create_task(asyncio.to_thread(job))
    return {"ok": True}


@app.post("/api/speed")
async def speed(body: dict):
    _STATE["speed"] = max(1, min(500, int(body.get("speed", 20))))
    _STATE["running"] = bool(body.get("run", True))
    return {"ok": True, **{k: _STATE[k] for k in ("speed", "running")}}


@app.post("/api/replay_file")
async def replay_file(body: dict):
    """Switch stream to a real-telemetry replay (SMAP test arrays)."""
    from ..sources import load_smap
    try:
        sats = body.get("sats") or None
        lst = load_smap(os.path.join(DATA, "SMAP"), sats=sats, split="test")
        i = int(body.get("sat", 0)) % len(lst)
        fs = lst[i]
        _STATE["file_src"] = fs
        _STATE["mode"] = "file"
        _STATE["warm"] = False
        _STATE["running"] = False
        asyncio.create_task(asyncio.to_thread(_rebuild_blocking, fs, 2400))
        asyncio.create_task(_mark_warm())
        return {"ok": True, "sats": len(lst), "using": sats or "auto",
                "note": "rebuilding engine for SMAP channels (~1 min); reload console after"}
    except Exception as e:
        return JSONResponse({"ok": False, "err": repr(e)}, status_code=400)


async def _mark_warm():
    while _STATE.get("rebuilding"):
        await asyncio.sleep(2)
    _STATE["warm"] = True


@app.post("/api/twin_mode")
async def twin_mode():
    _STATE["mode"] = "twin"
    _STATE["warm"] = False
    _STATE["running"] = False
    asyncio.create_task(asyncio.to_thread(_rebuild_blocking, src, 7400))
    asyncio.create_task(_mark_warm())
    return {"ok": True}


@app.get("/api/bench")
async def bench():
    p = os.path.join(DATA, "bench.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"note": "run `make bench` to generate"}


# ---------------------------------------------------------------------------
# Live sky: NASA DONKI space-weather alerts + ISS position (wheretheiss.at)
# + ISRO satellite positions from public CelesTrak TLEs propagated locally
# with sgp4. All three feeds are public and keyless. Server-side proxy so the
# browser never fights CORS; 30 s cache; every source degrades alone.
# ---------------------------------------------------------------------------
_SKY = {"t": 0.0, "data": None}
ISRO_TLE = {"Resourcesat-2A": 41877, "Oceansat-3": 54361, "Cartosat-2C": 41599, "NVS-01": 56759}


def _get_json(url, timeout=7):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "aegis-console/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _get_text(url, timeout=7):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "aegis-console/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _sky_blocking():
    import math
    from concurrent.futures import ThreadPoolExecutor
    out = {"ts": time.strftime("%H:%M:%S UTC", time.gmtime())}

    # optional sgp4 for the ISRO leg — import once, before dispatching work
    have_sgp4 = False
    try:
        from sgp4.io import twoline2rv
        from sgp4.earth_gravity import wgs72
        from sgp4.propagation import sgp4 as _prop
        try:
            from sgp4.propagation import gstime as _gst
        except ImportError:
            def _gst(jdate):  # IAU 1982 GMST, radians
                tu = (jdate - 2451545.0) / 36525.0
                th = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tu
                      + 0.093104 * tu * tu - 6.2e-6 * tu ** 3) % 86400.0
                return 2 * math.pi / 86400.0 * th
        have_sgp4 = True
    except ImportError as e:
        out["isro_err"] = f"needs sgp4 (python -m pip install sgp4): {e}"

    def _iss():
        import datetime as _d
        d = _get_json("https://api.wheretheiss.at/v1/satellites/25544", timeout=6)
        return {"lat": d["latitude"], "lon": d["longitude"],
                "alt_km": d["altitude"], "vel_kms": round(d["velocity"] / 3600, 2),
                "visibility": d.get("visibility"), "footprint_km": d.get("footprint"),
                "solar_lat": d.get("solar_lat"), "solar_lon": d.get("solar_lon")}

    def _wx():
        import datetime as _d
        d0 = (_d.datetime.utcnow() - _d.timedelta(days=2)).strftime("%Y-%m-%d")
        d1 = _d.datetime.utcnow().strftime("%Y-%m-%d")
        qs = f"startDate={d0}&endDate={d1}"
        notifs = _get_json(f"https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications?{qs}", timeout=8)
        try:
            flares = _get_json(f"https://ccmc.gsfc.nasa.gov/DONKI-API/get/Flares?{qs}", timeout=8) or []
        except Exception:
            flares = []
        wx = {"alerts": [], "flares": []}
        for n in (notifs or []):
            wx["alerts"].append({"type": n.get("messageType"), "id": n.get("messageID"),
                                 "issued": (n.get("messageIssueTime") or "")[:16],
                                 "url": n.get("messageURL")})
        wx["alerts"] = wx["alerts"][-14:][::-1]
        big = sorted((f for f in flares if f.get("classType")),
                     key=lambda f: f.get("peakTime") or "", reverse=True)[:6]
        wx["flares"] = [{"cls": f.get("classType"), "peak": (f.get("peakTime") or "")[11:16],
                         "active_region": f.get("activeRegionNumber"), "id": f.get("eventTime")}
                        for f in big]
        return wx

    def _isro_prop(name, catnr, l1, l2, jd):
        sat = twoline2rv(l1, l2, wgs72)
        r, v = _prop(sat, (jd - (sat.jdsatepoch + sat.jdsatepochF)) * 1440.0)
        if getattr(sat, "error", 0):
            return {"name": name, "id": catnr, "err": f"propagate error {sat.error}"}
        th = _gst(jd)
        ct, st = math.cos(th), math.sin(th)
        x = r[0] * ct + r[1] * st
        y = -r[0] * st + r[1] * ct
        z = r[2]
        rho = math.hypot(x, y)
        return {"name": name, "id": catnr,
                "lat": round(math.degrees(math.atan2(z, rho)), 3),
                "lon": round((math.degrees(math.atan2(y, x)) + 540) % 360 - 180, 3),
                "alt_km": round(math.hypot(rho, z) - 6378.137, 1),
                "vel_kms": round(math.hypot(*v), 2),
                "tle_epoch": l1[18:32].strip()}

    def _isro_positions(jd):
        """Live ISRO orbits from the fleet TLE cache + local SGP4. Per-request
        CelesTrak CATNR fetches were rate-limited into timeouts (HTTP 503) — the
        TLE epoch is hours old anyway, so cached orbital elements are the honest
        input; the cache refreshes itself on its own schedule."""
        fresh = bool(FLEET_CACHE["tle"]) and time.time() - FLEET_CACHE["t"] < FLEET_TTL
        rows = {ln[0].upper(): ln for ln in ((FLEET_CACHE["tle"] or {}).get("isro") or [])}
        for _kk, _vv in _ISRO_SPARE["rows"].items():
            rows.setdefault(_kk, _vv)
        if not rows:
            _prime_isro()          # background, 10-min dedup: celestrak → tle-API fallback
            return [{"name": n, "id": c, "waiting": "orbit data refreshing"} for n, c in ISRO_TLE.items()]
        _ = fresh
        sats = []
        for name, catnr in ISRO_TLE.items():
            hit = next((v for k, v in rows.items() if name.upper()[:8] in k), None)
            if hit is None:
                sats.append({"name": name, "id": catnr, "waiting": "orbit data refreshing"})
                continue
            try:
                sats.append(_isro_prop(name, catnr, hit[1], hit[2], jd))
            except Exception as e:
                sats.append({"name": name, "id": catnr, "err": type(e).__name__})
        return sats

    jd = _SKY_JD()
    with ThreadPoolExecutor(max_workers=6) as ex:
        f_iss = ex.submit(_iss)
        f_wx = ex.submit(_wx)
        f_isro = ex.submit(_isro_positions, jd) if have_sgp4 else None

        try:
            out["iss"] = f_iss.result(9)
        except Exception as e:
            out["iss_err"] = f"iss feed unreachable: {type(e).__name__}"
        try:
            out["wx"] = f_wx.result(11)
        except Exception as e:
            out["wx_err"] = f"nasa donki feed unreachable: {type(e).__name__}"
        if f_isro is not None:
            try:
                out["isro"] = f_isro.result(9)
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                out["isro_err"] = f"orbit positions unavailable: {type(e).__name__}: {e}"
        # NOAA SWPC planetary K-index — real geomagnetic activity, updated minutely
        try:
            kp_rows = _get_json("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
                                timeout=6)
            series = []
            for r0 in (kp_rows[1:] if (isinstance(kp_rows, list) and kp_rows
                                       and isinstance(kp_rows[0], list)) else kp_rows):
                try:
                    if isinstance(r0, dict):   # noaa-planetary-k-index.json (3 h cadence)
                        t0, kp = (r0.get("time_tag") or "")[:16], float(r0.get("Kp") or 0)
                    else:                       # [[time, Kp, frac, aKp], ...] rows
                        t0, kp = r0[0][:16], float(r0[1]) + float(r0[2] or 0) / 3.0
                except Exception:
                    continue
                series.append([t0, round(min(9.0, kp), 2)])
            step = max(1, len(series) // 60)
            series = series[::step][-60:]
            out["kp"] = {"now": series[-1] if series else None,
                         "max24": max((v for _, v in series), default=0.0),
                         "storm": bool(series and series[-1][1] >= 5.0),
                         "series": series}
        except Exception as e:
            out["kp_err"] = f"swpc unreachable: {type(e).__name__}"
    return out


def _SKY_JD():  # Julian date (UTC now) — helper kept next to the sky block
    import datetime as _d
    dtx = _d.datetime.now(_d.timezone.utc)
    a = (14 - dtx.month) // 12
    yy = dtx.year + 4800 - a
    mm2 = dtx.month + 12 * a - 3
    jdn = (dtx.day + (153 * mm2 + 2) // 5 + 365 * yy + yy // 4
           - yy // 100 + yy // 400 - 32045)
    return jdn - 0.5 + (dtx.hour * 3600 + dtx.minute * 60 + dtx.second
                        + dtx.microsecond / 1e6) / 86400.0




# ---------------------------------------------------------------------------
# Fleet watch: CelesTrak group TLEs → orbital-health scoring + live positions.
# The "health metrics" here are what a ground segment can actually derive
# without privileged downlinks: ephemeris freshness, decay rate (ṅ), drag
# (B*), perigee, eccentricity anomalies, GEO station-keeping drift — the same
# evidence-card format as the telemetry engine, so explanations read alike.
# ---------------------------------------------------------------------------
FLEET_GROUPS = {"stations": "Space stations", "gps-ops": "GPS navigation",
                "weather": "Weather (NOAA/GOES)", "geo": "Geostationary comms"}
FLEET_CACHE = {"t": 0.0, "tle": {}, "errs": {}}
FLEET_TTL = 6 * 3600.0


def _tle_to_elems(l1, l2):
    """Parse the elements we score on straight from TLE columns."""
    def f(x, a, b):
        try:
            return float(x[a:b])
        except Exception:
            return 0.0
    def _exp_col(x):
        """TLE mantissa/exponent field: ' 27299-3' or '-.14965-3' → float."""
        t = x.strip().replace(" ", "")
        if not t:
            return 0.0
        try:
            if "." in t and ("e" in t or "E" in t or t.count("-") <= 1 and "+" not in t and len(t.split("-")) <= 2 and not t[-2:].lstrip("+-").isdigit()):
                return float(t)
        except Exception:
            pass
        sign = -1.0 if t.startswith("-") else 1.0
        if t[0] in "+-":
            t = t[1:]
        if t.startswith("."):
            t = t[1:]
        try:
            if len(t) <= 2:
                return 0.0
            mant = float(t[:-2]) / (10 ** len(t[:-2]))
            return sign * mant * (10.0 ** int(t[-2:]))
        except Exception:
            return 0.0
    yr = int(l1[18:20])
    yr += 2000 if yr < 57 else 1900
    try:
        mm_dot = float(l1[33:43]) * 1e-8
    except Exception:
        mm_dot = 0.0
    mm_ddot = _exp_col(l1[44:53]) * 1e-11 if "." not in l1[44:53] else float(l1[44:52]) * 1e-11
    bstar = _exp_col(l1[53:61])
    ecc = float("0." + l2[26:33].strip() if l2[26:33].strip() else 0)
    return {"epoch_yr": yr, "epoch_days": f(l1, 20, 32), "mm": f(l2, 52, 63),
            "mm_dot": mm_dot, "mm_ddot": mm_ddot, "bstar": bstar, "ecc": ecc,
            "inc": f(l2, 8, 16), "raan": f(l2, 17, 25), "argp": f(l2, 34, 42),
            "manom": f(l2, 43, 52)}


def _fleet_tles():
    """6 h-cached group TLEs + the curated ISRO birds."""
    import datetime as _d
    if FLEET_CACHE["tle"] and time.time() - FLEET_CACHE["t"] < FLEET_TTL:
        return FLEET_CACHE["tle"], FLEET_CACHE["errs"]
    from concurrent.futures import ThreadPoolExecutor

    def grab(group):
        txt = _get_text(f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle",
                        timeout=15)
        lns = [l for l in txt.splitlines() if l.strip()]
        out_ = []
        for i in range(0, len(lns) - 2, 3):
            if lns[i + 1].startswith("1 ") and lns[i + 2].startswith("2 "):
                out_.append((lns[i].strip(), lns[i + 1], lns[i + 2]))
        return group, out_

    out, errs = {}, dict(FLEET_CACHE["errs"])
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(grab, g) for g in FLEET_GROUPS]
        isro_f = {n: ex.submit(_get_text, f"https://celestrak.org/NORAD/elements/gp.php?CATNR={c}&FORMAT=tle", 12)
                  for n, c in ISRO_TLE.items()}
        for _g, fu in zip(FLEET_GROUPS, futs):
            try:
                g, rows = fu.result(20)
                out[g] = rows
            except Exception as e:
                errs[_g] = f"{type(e).__name__}"
        for n, fu in isro_f.items():
            try:
                lns = [l for l in fu.result(15).splitlines() if l.strip()]
                if len(lns) >= 3:
                    out.setdefault("isro", []).append((lns[0].strip(), lns[1], lns[2]))
            except Exception as e:
                errs["isro:" + n] = f"{type(e).__name__}"
    if out:                                  # ≥1 group succeeded → keep a disk snapshot
        try:
            import json as _j
            snap = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fleet_cache.json")
            with open(snap, "w") as fh:
                _j.dump(out, fh)
        except Exception:
            pass
    elif FLEET_CACHE["tle"] is None:          # cold + throttled: serve last-good snapshot
        try:
            import json as _j
            snap = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fleet_cache.json")
            out = _j.load(open(snap))
            errs["snapshot"] = "celestrak unreachable — serving last-good TLE snapshot"
            FLEET_CACHE["tle"], FLEET_CACHE["t"] = out, time.time() - FLEET_TTL + 1800
            return out, errs
        except Exception:
            pass
    FLEET_CACHE["tle"], FLEET_CACHE["t"], FLEET_CACHE["errs"] = out, time.time(), errs
    return out, errs


def _fleet_score(name, group, el, now):
    """Deterministic orbital-health evidence — same shape as the telemetry
    engine's evidence rows: metric, value, limit, weight, note."""
    import math as _m
    ev = []
    n = el["mm"]
    if n <= 0:
        return None, []
    # Kepler III: a = (μ/(2π/T)²)^(1/3), μ = 398600.4418 km³/s²
    a_km = (398600.4418 / (n * 2 * _m.pi / 86400.0) ** 2) ** (1.0 / 3.0)
    per = a_km * (1 - el["ecc"]) - 6378.137
    apo = a_km * (1 + el["ecc"]) - 6378.137
    def add(metric, val, limit, wt, note):
        ev.append({"metric": metric, "value": val, "limit": limit, "weight": wt, "note": note})
    if per < 140:
        add("perigee", round(per, 1), "> 140 km", 1.0, "decaying — re-entry imminent")
    elif per < 200:
        add("perigee", round(per, 1), "> 200 km", 0.5, "high drag regime, orbit shrinking fast")
    b = abs(el["bstar"])
    lim_b = 2e-4 if n < 4 else 1e-5
    if b > 10 * lim_b:
        add("B* drag", f"{b:.1e}", f"< {10*lim_b:.0e}", 0.7,
            "abnormally high — attitude tumble, debris-avoidance burn, or drag rise")
    elif b > 2 * lim_b:
        add("B* drag", f"{b:.1e}", f"< {10*lim_b:.0e}", 0.25, "elevated drag — watch decay rate")
    n_dot = abs(el["mm_dot"])
    if n_dot > 1e-2:
        add("decay rate ṅ", f"{el['mm_dot']:.1e} rev/day²", "< 1e-2", 1.0, "rapid orbital decay — deorbit or loss of control")
    elif n_dot > 1.5e-3:
        add("decay rate ṅ", f"{el['mm_dot']:.1e} rev/day²", "< 1.5e-3", 0.4, "accelerating — sustained drag or low perigee")
    if group == "geo":
        if abs(n - 1.0027) > 0.006:
            add("GEO drift", f"{n:.4f} rev/day", "1.0027 ±0.006", 0.8,
                "station-keeping off nominal — drifting out of slot")
        if per < 35400 or apo > 35950:
            add("GEO box", f"{per:.0f}/{apo:.0f} km", "35578–35786 km ±200", 0.6,
                "outside the station-keeping box")
    if el["ecc"] > (0.05 if group == "geo" else 0.02) and group != "stations":
        add("eccentricity", f"{el['ecc']:.4f}", "< 0.02", 0.35,
            "unexpectedly elliptical — maneuver residue or anomaly")
    return {"perigee": round(per, 1), "apogee": round(apo, 1),
            "period_min": round(1440.0 / n, 2)}, ev


@app.get("/api/fleet")
async def fleet(force: int = 0):
    """orbital-health table for ~all public groups + live positions"""
    import datetime as _d
    if force:
        FLEET_CACHE["t"] = 0.0
    def build():
        from sgp4.io import twoline2rv
        from sgp4.earth_gravity import wgs72
        from sgp4.propagation import sgp4 as _prop
        try:
            from sgp4.propagation import gstime as _gst
        except ImportError:
            import math as _m
            def _gst(jd):
                tu = (jd - 2451545.0) / 36525.0
                th = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tu
                      + 0.093104 * tu * tu - 6.2e-6 * tu ** 3) % 86400.0
                return 2 * _m.pi / 86400.0 * th
        tles, errs = FLEET_CACHE["tle"], FLEET_CACHE["errs"]
        jd = _SKY_JD()
        now = _d.datetime.now(_d.timezone.utc)
        th = _gst(jd); ct, st = math.cos(th), math.sin(th)
        rows = []
        for g, lst in tles.items():
            for name, l1, l2 in lst[:400]:
                try:
                    el = _tle_to_elems(l1, l2)
                    # tle age from epoch year/days (UTC)
                    try:
                        ep = _d.datetime(el["epoch_yr"], 1, 1, tzinfo=_d.timezone.utc) + _d.timedelta(days=el["epoch_days"] - 1)
                        el["epoch_yr"] = ep.timestamp() / 86400.0
                        age_h = max(0.0, (now - ep).total_seconds() / 3600.0)
                    except Exception:
                        age_h = 0.0
                    geo, ev = _fleet_score(name, g, el, now)
                    if geo is None:
                        continue
                    score = min(1.0, sum(e["weight"] for e in ev))
                    if age_h > 168:
                        ev.append({"metric": "tle_age", "value": f"{age_h/24:.1f} d", "limit": "< 7 d",
                                   "weight": 0.3, "note": "stale ephemeris — all positions approximate"})
                        score = max(score, 0.35)
                    sat = twoline2rv(l1, l2, wgs72)
                    r, v = _prop(sat, (jd - (sat.jdsatepoch + sat.jdsatepochF)) * 1440.0)
                    x = r[0] * ct + r[1] * st; y = -r[0] * st + r[1] * ct; z = r[2]
                    rho = math.hypot(x, y)
                    rows.append({"name": name, "group": g, "id": int(l1[2:7]),
                                 **geo, "inc": round(el["inc"], 2), "mm": round(el["mm"], 6),
                                 "bstar": f"{el['bstar']:.2e}", "ndot": f"{el['mm_dot']:.1e}",
                                 "ecc": round(el["ecc"], 5), "tle_age_h": round(age_h, 1),
                                 "lat": round(math.degrees(math.atan2(z, rho)), 3),
                                 "lon": round((math.degrees(math.atan2(y, x)) + 540) % 360 - 180, 3),
                                 "alt_km": round(math.hypot(rho, z) - 6378.137, 1),
                                 "score": round(score, 3),
                                 "level": "ALERT" if score >= 0.6 else ("WATCH" if score >= 0.2 else "OK"),
                                 "evidence": ev})
                except Exception:
                    continue
        rows.sort(key=lambda r_: -r_["score"])
        FLEET_CACHE["rows_last"] = rows
        return {"rows": rows, "counts": {g: len(v) for g, v in tles.items()},
                "errs": errs, "tle_age_min": round((time.time() - FLEET_CACHE["t"]) / 60, 1),
                "now": time.strftime("%H:%M:%S UTC", time.gmtime())}
    try:
        FLEET_CACHE["tle"] or await asyncio.to_thread(_fleet_tles)
    except Exception as e:
        return {"err": f"celesstrak unreachable: {type(e).__name__}", "rows": []}
    try:
        return _san(await asyncio.wait_for(asyncio.to_thread(build), 25))
    except Exception as e:
        return {"err": f"fleet build failed: {type(e).__name__}: {e}", "rows": []}


@app.get("/api/passes")
async def passes(lat: float = 19.076, lon: float = 72.878, catnr: int = 25544, hours: int = 24):
    """visible passes (max elevation ≥ 10°) for a satellite over an observer —
    i.e. the telemetry contact windows."""
    def build():
        import datetime as _d
        import math as _m
        from sgp4.io import twoline2rv
        from sgp4.earth_gravity import wgs72
        from sgp4.propagation import sgp4 as _prop
        txt = _get_text(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle", timeout=12)
        lns = [l for l in txt.splitlines() if l.strip()]
        if len(lns) < 3:
            return {"err": "satellite not found on CelesTrak"}
        sat = twoline2rv(lns[1], lns[2], wgs72)
        name = lns[0].strip()
        a_e, f = 6378.137, 1 / 298.257223563
        phi, lam = _m.radians(lat), _m.radians(lon)
        n_ = a_e / _m.sqrt(1 - f * (2 - f) * _m.sin(phi) ** 2)
        ox = n_ * _m.cos(phi) * _m.cos(lam)
        oy = n_ * _m.cos(phi) * _m.sin(lam)
        oz = n_ * (1 - f * (2 - f)) * _m.sin(phi)
        try:
            from sgp4.propagation import gstime as _gst
        except ImportError:
            def _gst(jd):
                tu = (jd - 2451545.0) / 36525.0
                th = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tu
                      + 0.093104 * tu * tu - 6.2e-6 * tu ** 3) % 86400.0
                return 2 * _m.pi / 86400.0 * th
        t0 = _d.datetime.now(_d.timezone.utc)
        jd0 = _SKY_JD()
        out_p, cur = [], None
        for k in range(0, int(hours * 120)):          # 30 s steps
            fr = k / 4320.0
            r, v = _prop(sat, ((jd0 + fr) - (sat.jdsatepoch + sat.jdsatepochF)) * 1440.0)
            th = _gst(jd0 + fr)
            ct, st = _m.cos(th), _m.sin(th)
            x = r[0] * ct + r[1] * st; y = -r[0] * st + r[1] * ct; z = r[2]
            dx, dy, dz = x - ox, y - oy, z - oz
            rng = _m.sqrt(dx * dx + dy * dy + dz * dz)
            sl = _m.sin(lam); cl = _m.cos(lam); sp = _m.sin(phi); cp = _m.cos(phi)
            top_e = sl * dy - cl * dx
            top_n = -sp * cl * dx - sp * sl * dy + cp * dz
            top_u = cp * cl * dx + cp * sl * dy + sp * dz
            elv = _m.degrees(_m.asin(max(-1, min(1, top_u / rng))))
            t = t0 + _d.timedelta(seconds=k * 30)
            if elv >= 10.0:
                if cur is None:
                    cur = {"aos": t.strftime("%H:%M:%S"), "max_el": elv, "los": None,
                           "range_km": rng}
                if elv > cur["max_el"]:
                    cur["max_el"] = round(elv, 1); cur["range_km"] = round(rng, 0)
            elif cur is not None:
                cur["los"] = t.strftime("%H:%M:%S")
                cur["max_el"] = round(cur["max_el"], 1)
                out_p.append(cur); cur = None
                if len(out_p) >= 8:
                    break
        if cur:
            cur["los"] = "…"; out_p.append(cur)
        return {"sat": name, "observer": [lat, lon], "horizon": 10,
                "passes": out_p, "window_h": hours}
    try:
        return _san(await asyncio.wait_for(asyncio.to_thread(build), 25))
    except Exception as e:
        return {"err": f"pass computation failed: {type(e).__name__}: {e}"}



# ---------------------------------------------------------------------------
# Explainable-AI layer. The DETECTION and RANKING stay deterministic (that is
# the whole point of AEGIS); the LLM only turns the evidence bundle into
# operator-language reasoning grounded in the numbers. Providers are tried in
# order — local ollama, OpenRouter, OmniRoute — through one OpenAI-compatible
# /v1/chat/completions call, so no SDKs are required. Every response states
# which provider/model actually spoke, and with what grounding.
# ---------------------------------------------------------------------------
AI_CONFIG_DEFAULTS = {
    "providers": [
        {"id": "ollama", "base": "https://splicing-wasp-germless.ngrok-free.dev/v1",
         "model": "llama3.2:3b", "api_key_env": None,
         "note": "local, offline, private"},
        {"id": "openrouter", "base": "https://openrouter.ai/api/v1",
         "model": "meta-llama/llama-3.1-8b-instruct", "api_key_env": "OPENROUTER_API_KEY",
         "note": "12k+ models, keyless = unavailable"},
        {"id": "omniroute", "base": "https://api.omniroute.ai/v1",
         "model": "omniroute/auto", "api_key_env": "OMNIROUTE_API_KEY",
         "note": "auto-routing across 20+ providers"},
    ],
    "max_tokens": 700, "temperature": 0.2,
}


def _ai_config():
    p = os.path.join(DATA, "ai_config.json")
    try:
        c = json.load(open(p))
        for k, v in AI_CONFIG_DEFAULTS.items():
            c.setdefault(k, v)
        # Fix: if api_key_env looks like an actual key (not an env var name),
        # store it directly and clear api_key_env so _ai_chat uses it
        for pr in c.get("providers", []):
            kenv = pr.get("api_key_env") or ""
            if kenv and (kenv.startswith("sk-") or kenv.startswith("key-")):
                pr["_direct_key"] = kenv
                pr["api_key_env"] = None
        return c, "data/ai_config.json"
    except Exception:
        return dict(AI_CONFIG_DEFAULTS), "defaults (ollama → openrouter → omniroute)"


def _ai_ssl_context():
    import certifi
    import ssl
    return ssl.create_default_context(cafile=certifi.where())


def _ai_chat(messages, cfg, want=None):
    """OpenAI-compatible chat with provider fallback; returns dict or raises."""
    import urllib.request
    tried = []
    provs = cfg["providers"] if not want else [p_ for p_ in cfg["providers"] if p_["id"] == want] or cfg["providers"]
    last = None
    for pr in provs:
        key = pr.get("_direct_key") or (os.environ.get(pr["api_key_env"], "") if pr.get("api_key_env") else "local")
        if pr.get("api_key_env") and not key:
            tried.append((pr["id"], "no api key in env"))
            continue
        body = json.dumps({"model": pr["model"], "messages": messages,
                           "max_tokens": cfg.get("max_tokens", 700),
                           "temperature": cfg.get("temperature", 0.2),
                           "stream": False}).encode()
        req = urllib.request.Request(pr["base"].rstrip("/") + "/chat/completions",
                                     data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        for name, value in (pr.get("headers") or {}).items():
            req.add_header(str(name), str(value))
        if key != "local":
            req.add_header("Authorization", "Bearer " + key)
        try:
            t0 = time.time()
            ctx = _ai_ssl_context() if pr["base"].startswith("https") else None
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                d = json.loads(r.read().decode())
            txt = d["choices"][0]["message"]["content"]
            return {"text": txt, "provider": pr["id"], "model": d.get("model", pr["model"]),
                    "latency_s": round(time.time() - t0, 2), "tried": tried,
                    "tokens": (d.get("usage") or {}).get("total_tokens")}
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            tried.append((pr["id"], last[:90]))
            continue
    raise RuntimeError("no AI provider reachable — " + "; ".join(f"{a}: {b}" for a, b in tried))


def _ai_narrative(bundle):
    """Deterministic fallback: the same reasoning in operator language,
    generated from the evidence bundle itself (no LLM, never fails)."""
    ev = bundle.get("evidence") or []
    if bundle.get("satellite"):
        m = bundle.get("metrics", {})
        s = f"{bundle['kind']} — {bundle['satellite']}. "
        if ev:
            s += "; ".join(f"{e['metric']} {e['value']} (limit {e['limit']} — {e['note']})"
                          for e in ev[:3]) + ". "
        else:
            s += f"Perigee {m.get('perigee')} km, B* {m.get('bstar')}, all rules within limits. "
        s += "Inference is from public ephemeris only (no onboard telemetry); " \
             "confirm with the operator's own ground station before acting."
        return s
    try:
        ch = ", ".join(str(c).replace("_", " ") for c in bundle.get("channels", [])[:3]) or "the affected channels"
        s = (f"Alert {bundle.get('group', 'ANOMALY')} at onset t={bundle.get('t0', 0):.0f} s "
             f"(peak score {bundle.get('score', 0):.1f}, conformal p={bundle.get('conformal_p', 1):.3f}) "
             f"on {ch}.")
        if ev and "channel" in (ev[0] or {}):
            e0 = ev[0]
            s += f" {len(ev)} independent indicators fired. Leading evidence: {e0['channel'].replace('_', ' ')} {e0.get('stat','')} z={float(e0.get('z', 0)):.1f} (limit {float(e0.get('theta', 3)):.1f})"
            if e0.get("lag_s"):
                s += f", preceded by {e0['lag_s']/60:.0f} min of drift in {e0.get('preceded_by','?').replace('_',' ')}"
            s += ". "
        cs = bundle.get("causes") or []
        if cs:
            s += f"Cause ranking puts {cs[0]['cause'].replace('_',' ')} first (score {cs[0].get('score', 0):.2f}"
            if len(cs) > 1:
                s += f", {cs[0].get('score',0) - cs[1].get('score',0):.2f} ahead of the runner-up"
            s += ")."
        s += " Actions are recommendation-only; nothing is uplinked without operator sign-off."
        return s
    except Exception:
        return (f"ALERT on {', '.join(bundle.get('channels', [])[:3])} — evidence rows attached; "
                "narrative generator tripped on schema, raw bundle is authoritative.")


AI_SYS = ("You are the explainability module of AEGIS, a spacecraft anomaly "
          "detection console. You NEVER invent numbers: reason ONLY from the "
          "evidence bundle in the user message. Answer in ≤140 words, plain "
          "operator language, structured as: what fired → why it is a real "
          "fault and not noise → most likely root cause → what to do. "
          "If the bundle is insufficient, say exactly what is missing.")


@app.get("/api/ai/status")
async def ai_status():
    import urllib.request
    cfg, src = _ai_config()
    out = []
    for pr in cfg["providers"]:
        st = {"id": pr["id"], "model": pr["model"], "base": pr["base"], "note": pr["note"]}
        key = os.environ.get(pr["api_key_env"], "") if pr.get("api_key_env") else "local"
        if pr.get("api_key_env") and not key:
            st["state"] = "no-key"
        else:
            try:
                rq = urllib.request.Request(
                    pr["base"].rstrip("/") + "/models",
                    headers={str(name): str(value)
                             for name, value in (pr.get("headers") or {}).items()})
                ctx = _ai_ssl_context() if pr["base"].startswith("https") else None
                with urllib.request.urlopen(rq, timeout=2.5, context=ctx) as r:
                    st["state"] = "reachable"
                    try:
                        ms = json.loads(r.read().decode()).get("data") or []
                        if ms:
                            st["available_models"] = [m.get("id") for m in ms[:6]]
                    except Exception:
                        pass
            except Exception:
                st["state"] = "down"
        out.append(st)
    return {"providers": out, "config": src,
            "model_used_last": _STATE.get("ai_last"),
            "grounding": "evidence bundle (event stats + conformal p + cause ranking) — LLM never invents numbers"}


@app.post("/api/ai/explain")
async def ai_explain(body: dict):
    eid = body.get("event_id")
    q = (body.get("question") or "").strip()[:600]
    want = body.get("provider")
    bundle = None
    if eid:
        def find():
            eng_g = globals().get("eng")
            pool = [b.to_dict() for b in eng_g.events[-80:]] if eng_g else []
            for b0 in pool:
                ev0 = b0.get("event", b0)
                if eid in (ev0.get("id"), ev0.get("event_id")):
                    b = dict(ev0)
                    if b0.get("causes"):
                        b["causes"] = b0["causes"]
                    th = eng_g.stats().get("theta") if eng_g else None
                    if "evidence" not in b and b.get("attrib"):
                        b["evidence"] = [{"channel": a.get("channel"), "stat": str(a.get("direction", "")) + " " + str(a.get("timescale", "")),
                                         "z": float(a.get("z", 0.0)), "theta": th} for a in b["attrib"]]
                    b.setdefault("channels", [a.get("channel") for a in b.get("attrib", [])])
                    b["context"] = {"n_events_total": eng_g.total_events if eng_g else len(pool),
                                     "mode": _STATE.get("mode"), "theta": th}
                    if b.get("causes"):
                        b["causes"] = [{"cause": c.get("fault_id") or c.get("name") or "unknown",
                                       "score": c.get("p", c.get("score", 0.0)),
                                       "matched": c.get("matched", [])} for c in b["causes"]]
                    return b
            return None
        bundle = await asyncio.to_thread(find)
    fname = body.get("fleet")
    if fname and not bundle:
        for r_ in (FLEET_CACHE.get("rows_last") or []):
            if r_["name"] == fname:
                bundle = {"kind": "FLEET " + r_["level"], "satellite": r_["name"],
                          "group": r_["group"], "tle": "public CelesTrak GP data",
                          "metrics": {k_: r_[k_] for k_ in ("perigee", "apogee", "period_min",
                                                            "bstar", "ndot", "ecc", "tle_age_h",
                                                            "alt_km", "score", "level")},
                          "evidence": r_["evidence"]}
                break
    if not bundle:
        bundle = {"question_only": q or "no event selected",
                  "mode": _STATE.get("mode")}
    cfg, _ = _ai_config()
    ground = json.dumps({"event": bundle, "question": q or "Explain this alert and the top cause."},
                        separators=(",", ":"))[:9000]
    msgs = [{"role": "system", "content": AI_SYS},
            {"role": "user", "content": "EVIDENCE BUNDLE (JSON, authoritative):\n" + ground +
             "\n\nAnswer ONLY from this bundle."}]
    try:
        r = await asyncio.wait_for(asyncio.to_thread(_ai_chat, msgs, cfg, want), 35)
        r["grounded_on"] = {"event_id": eid, "evidence_rows": len(bundle.get("evidence", [])),
                            "question": q or None}
        r["mode"] = "llm"
        _STATE["ai_last"] = f"{r['provider']}/{r['model']}"
    except Exception as e:
        r = {"text": _ai_narrative(bundle), "provider": "builtin", "model": "evidence-narrator",
             "note": f"LLM chain unavailable ({type(e).__name__}: {str(e)[:120]}) — deterministic fallback narrative generated directly from the evidence bundle",
             "grounded_on": {"event_id": eid, "evidence_rows": len(bundle.get("evidence", []))},
             "mode": "rule-based-fallback"}
    return _san(r)


@app.get("/api/crew")
async def crew():
    def build():
        try:
            d = _get_json("https://api.howmanyspace.com/api/v1/crew", timeout=6)
            ppl = d.get("people") or (d if isinstance(d, list) else [])
            return {"people": [{"name": p.get("name") or p.get("display_name"),
                                "role": p.get("role"), "craft": p.get("vehicle") or p.get("ship")} for p in ppl]}
        except Exception as e1:
            try:
                d = _get_json("http://api.open-notify.org/astros.json", timeout=6)
                return {"people": [{"name": p.get("name"), "role": None, "craft": p.get("craft")}
                                   for p in d.get("people", [])]}
            except Exception:
                return {"err": f"crew feeds unreachable: {type(e1).__name__}"}
    try:
        return _san(await asyncio.wait_for(asyncio.to_thread(build), 9))
    except Exception as e:
        return {"err": "crew feed timed out"}


_ISRO_PRIMING = {"t": 0.0}
_ISRO_SPARE = {"rows": {}, "t": 0.0}


def _prime_isro():
    """One-shot background fetch of ISRO TLE lines into FLEET_CACHE (max every
    10 min, never on the request path)."""
    import threading
    partial = len(_ISRO_SPARE["rows"]) < len(ISRO_TLE)
    if time.time() - _ISRO_PRIMING["t"] < (150 if partial else 600):
        return
    _ISRO_PRIMING["t"] = time.time()

    def one(name, catnr):
        try:
            txt = _get_text(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle", 8)
            lns = [l for l in txt.splitlines() if l.strip()]
            if len(lns) >= 3 and lns[1].startswith("1 "):
                return (lns[0].strip(), lns[1], lns[2])
        except Exception:
            pass
        try:  # public mirror, keyless — used when CelesTrak throttles us
            d = _get_json(f"https://tle.ivanstanojevic.me/api/tle/{catnr}", timeout=8)
            if d.get("line1") and d.get("line2"):
                return (d.get("name") or name, d["line1"], d["line2"])
        except Exception:
            pass
        return None

    def work():
        try:
            rows = []
            for n, c in ISRO_TLE.items():      # sequential: mirror rate-limit is 10 req/min
                r0 = one(n, c)
                if r0:
                    rows.append(r0)
                time.sleep(1.5)
            if rows:
                _ISRO_SPARE["rows"] = {r[0].upper(): list(r) for r in rows}
                _ISRO_SPARE["t"] = time.time()
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()


@app.get("/api/sky")
async def sky():
    now = time.time()
    if _SKY["data"] is not None and now - _SKY["t"] < 30:
        d = dict(_SKY["data"]); d["cache_s"] = round(now - _SKY["t"], 1)
        return d
    try:
        d = await asyncio.wait_for(asyncio.to_thread(_sky_blocking), 14)
    except Exception as e:
        d = {"err": f"sky fetch failed: {type(e).__name__}"}
    got = d if any(k in d for k in ("iss", "wx", "isro")) else None
    if got is None:                      # total failure → serve last good, flagged stale
        d = dict(_SKY["data"] or {}); d["stale"] = True; d.setdefault("err", "all sky feeds unreachable")
        return d
    _SKY["t"], _SKY["data"] = now, d
    return d


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
