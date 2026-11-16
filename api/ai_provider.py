"""Optional OpenAI-compatible explanation provider integration."""
from __future__ import annotations

import json
import logging
import os
import time

from .settings import settings

DATA = str(settings.data_dir)
logger = logging.getLogger("aegis.api.ai")

AI_CONFIG_DEFAULTS = {
    "providers": [
        {"id": "ollama", "base": "http://127.0.0.1:11434/v1",
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
    except FileNotFoundError:
        return dict(AI_CONFIG_DEFAULTS), "defaults (ollama → openrouter → omniroute)"
    except Exception as exc:
        logger.warning("AI provider config unavailable; using defaults: %s", exc)
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


