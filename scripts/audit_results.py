"""Dump provenance and per-benchmark numbers of every results/benchmarks/*.json for paper auditing."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHES = ["mmstar", "hallusion", "mathvista", "chartqa", "ocrbench", "geometry3k"]


def main():
    for p in sorted((ROOT / "results" / "benchmarks").glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        a = d.get("arch", {})
        flags = [k for k in ("multifreq", "hybrid", "tokrot", "qcond", "direct_hyb") if a.get(k)]
        print(f"== {d['name']}  mode={d.get('mode')}  front={a.get('front_mode')} K={a.get('K')} flags={flags}")
        print(f"   ckpt={d.get('ckpt')}")
        print(f"   model={d.get('model')}  max_pixels={d.get('args', {}).get('max_pixels')} force_gen={d.get('args', {}).get('force_gen')}")
        b = d["benches"]
        accs = {k: 100 * b[k]["acc"] for k in BENCHES if k in b}
        mc2 = (accs["mmstar"] + accs["hallusion"]) / 2
        core3 = (accs["mmstar"] + accs["hallusion"] + accs["mathvista"]) / 3
        avg6 = sum(accs.values()) / len(accs)
        print("   " + "  ".join(f"{k}={v:.2f}" for k, v in accs.items()) + f"  | MC2={mc2:.3f} Core3={core3:.3f} Avg6={avg6:.3f}")
        for k in ("mmstar", "hallusion", "mathvista"):
            x = b[k]
            ci = x.get("ci95")
            cats = ", ".join(f"{c}={100 * v['acc']:.1f}" for c, v in x.get("by_cat", {}).items())
            print(f"   {k}: ci95=[{100 * ci[0]:.1f},{100 * ci[1]:.1f}] cal={100 * x.get('cal_acc', float('nan')):.1f} | {cats}")


if __name__ == "__main__":
    main()
