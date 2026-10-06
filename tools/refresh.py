"""`make refresh` companion: folds operator labels into feedback priors.

labels.jsonl verdicts (confirm/override) shift fault priors used by RCA.
Run after a day of operation or right before re-deploy.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aegis.reasoning import ontology

def main(path="data/labels.jsonl", out="data/feedback_priors.json"):
    ont = ontology.get(); pri = dict(ont.priors)
    if not os.path.exists(path):
        print("no labels yet"); return
    for ln in open(path):
        try: d = json.loads(ln)
        except Exception: continue
        fid = d.get("fault")
        if fid in pri:
            pri[fid] = max(0.01, pri[fid] * (1.30 if d.get("verdict") == "confirm" else 0.70))
    tot = sum(pri.values())
    pri = {k: round(v / tot, 4) for k, v in pri.items()}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(pri, open(out, "w"), indent=1)
    print("wrote", out, pri)

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
