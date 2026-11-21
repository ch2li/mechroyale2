"""lab/autopsy.py - classify how the bot died, from BlackBox summaries.

Input: a lab_runs_*.json written by lab/calibrate.py (variant -> list of match results).
Every death gets one primary cause plus tags; the report ranks causes by how often
they happen and how many points (16 - place) those matches scored.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

WINDOW = 10.0


def classify(r):
    if r["survived"]:
        return "survived", []
    dmg = [d for d in r["damage"] if d["t"] >= r["t_end"] - WINDOW]
    tags = []
    if r["zone_death"]:
        return "zone", tags
    tot = sum(d["amt"] for d in dmg) or 1.0
    attackers = Counter()
    for d in dmg:
        if d["by"] is not None:
            attackers[d["by"]] += d["amt"]
    rear = sum(d["amt"] for d in dmg if d["rear"]) / tot
    unseen = sum(d["amt"] for d in dmg if not d["seen"] and d["w"] != "zone") / tot
    by_w = Counter()
    for d in dmg:
        by_w[d["w"]] += d["amt"]
    if rear > 0.3:
        tags.append("flanked(rear>30%)")
    if unseen > 0.4:
        tags.append("shot-by-unseen")
    if by_w.get("laser", 0) / tot > 0.5:
        tags.append("laser-burst")
    if by_w.get("npc", 0) / tot > 0.25:
        tags.append("npc-dmg")
    eng = r.get("engage_hp") or []
    if eng and eng[-1][1] < 0.7:
        tags.append(f"entered-last-fight<70%hp")
    modes = Counter(x["mode"].split("+")[0] for x in r["last"][-20:])
    top_mode = modes.most_common(1)[0][0] if modes else "?"
    tags.append(f"mode:{top_mode}")
    n = len([a for a, v in attackers.items() if v > 0.1 * tot])
    cause = {0: "unknown", 1: "lost-1v1", 2: "2-on-1"}.get(n, "3+-on-1")
    return cause, tags


def report(runs):
    for name, res in runs.items():
        causes = defaultdict(list)
        tagc = Counter()
        for r in res:
            c, tags = classify(r)
            causes[c].append(r["pts"])
            for t in tags:
                tagc[t] += 1
        n = len(res)
        print(f"\n=== {name}: {n} matches, mean pts {statistics.mean(r['pts'] for r in res):.2f}")
        for c, pts in sorted(causes.items(), key=lambda kv: -len(kv[1])):
            print(f"  {c:12s} {len(pts):4d} ({100 * len(pts) / n:4.1f}%)  mean pts {statistics.mean(pts):5.2f}")
        print("  tags:", ", ".join(f"{t} {100 * v / n:.0f}%" for t, v in tagc.most_common(12)))
        eng = [e[1] for r in res for e in (r.get("engage_hp") or [])[1:]]
        if eng:
            low = sum(1 for e in eng if e < 0.7)
            print(f"  fights started after the first: {len(eng)}, {100 * low / len(eng):.0f}% at <70% HP, median HP {statistics.median(eng):.2f}")
        print(f"  regen seconds/match {statistics.mean(r['regen_s'] for r in res):.1f}   "
              f"fighting<60%HP seconds/match {statistics.mean(r['low_hp_fight_s'] for r in res):.1f}   "
              f"death time median {statistics.median(r['t_end'] for r in res):.0f}s")
        kt = Counter(r.get("killer_tier") for r in res if not r["survived"])
        print("  killed by tier:", dict(kt))


if __name__ == "__main__":
    report(json.load(open(sys.argv[1])))
