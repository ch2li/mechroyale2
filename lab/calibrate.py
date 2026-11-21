"""lab/calibrate.py - does the practice sim reproduce our real practice scores?

Each variant is the current bot with FLAGS overrides that recreate a version we
actually submitted; REAL holds the mean practice score that version got.
Usage: python3 lab/calibrate.py N_SEEDS '{"skill_scale":1.0}' [variant,...]
"""
import json
import multiprocessing as mp
import statistics
import sys

BOT = "/home/claude/mechroyale2"
K1 = {"risk_nav": True, "risk_k": 1.0}
K2 = {"risk_nav": True, "risk_k": 2.0}
VARIANTS = {   # name: (flags, real mean practice score, real runs)
    "old_base":     ({"risk_nav": False}, 3.97, 6),
    "v12_k1":       (K1, 4.19, 3),
    "k2":           (K2, 4.32, 3),
    "k3":           ({**K2, "risk_k": 3.0}, 4.08, 1),
    "k2_tmap":      ({**K1, "tmap": True}, 3.94, 1),
    "k1_isolate":   ({**K1, "isolate": True}, 3.67, 1),
    "k1_lightning": ({**K1, "lightning_first": True}, 3.88, 1),
    "k1_riskfight": ({**K1, "risk_fight": True}, 3.92, 1),
    "k2_standoff":  ({**K2, "standoff": True}, 4.06, 1),
    "no_npc":       ({"risk_nav": False, "hunt_npcs": False}, 3.75, 1),
    "k2_regen":     ({**K2, "regen_gate": True}, None, 0),
    "pacifist":     ({"risk_nav": False, "pacifist": True, "hunt_npcs": False}, 0.56, 1),
}


def one(args):
    seed, flags, params = args
    sys.path.insert(0, BOT + "/lab")
    from practice_sim import run_match
    r = run_match(seed, BOT, flags=flags, **params)
    bb = r.pop("bb")
    r["engage_hp"] = bb["engage_hp"]
    r["regen_s"] = bb["regen_s"]
    r["low_hp_fight_s"] = bb["low_hp_fight_s"]
    r["damage"] = bb["damage"]
    r["last"] = bb["last"][-30:]
    r["mode_s"] = bb["mode_s"]
    return r


def run_variant(name, n, params, start=1000, procs=2):
    flags = VARIANTS[name][0]
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=4) as pool:
        res = pool.map(one, [(start + i, flags, params) for i in range(n)])
    return res


if __name__ == "__main__":
    n = int(sys.argv[1])
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
    out = {}
    for name in names:
        res = run_variant(name, n, params)
        pts = [r["pts"] for r in res]
        m = statistics.mean(pts)
        se = statistics.pstdev(pts) / max(1, len(pts)) ** 0.5
        real = VARIANTS[name][1] or 0.0
        print(f"{name:14s} sim {m:5.2f} ±{se:4.2f}   real {real:4.2f}   kills {statistics.mean(r['kills'] for r in res):4.1f}  "
              f"t_end {statistics.mean(r['t_end'] for r in res):5.0f}s  zone_deaths {sum(r["zone_death"] for r in res)} house_zone {sum(r["house_zone_dead"] for r in res)} survived {sum(r["survived"] for r in res)}", flush=True)
        out[name] = res
    tag = json.dumps(params, sort_keys=True).replace(" ", "")
    json.dump(out, open(f"/home/claude/lab_runs_{n}_{abs(hash(tag)) % 10**6}.json", "w"), default=str)
