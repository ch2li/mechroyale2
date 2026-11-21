"""lab/practice_sim.py - local practice match that follows the official practice rules.

Differences from the old sim.py (which this extends):
  * all 15 house bots exist from t=0 (they count as alive for placement), dormant until released;
  * release order is weakest first: 4 easy, 4 medium, 4 hard, 3 expert;
  * release is kill-triggered: `first_at` s one hunter, +`per_kill` per kill, everyone after `all_at`;
  * a dormant bot wakes up if we shoot it or come within `wake_r` m;
  * placement = 16 - (house bots dead before us); survivors at the clock are ranked by HP;
  * the bot's BlackBox summary is returned for lab/autopsy.py.

Usage: from lab.practice_sim import run_match; run_match(seed, botdir, **params)
"""
import json
import math
import os
import sys
import time

TIERS = [("easy", 4), ("medium", 4), ("hard", 4), ("expert", 3)]


def tier_of(i):
    k = 0
    for name, n in TIERS:
        if i < k + n:
            return name
        k += n
    return "expert"


def run_match(seed, botdir, first_at=20.0, per_kill=1.0, all_at=300.0, skill_scale=1.0,
              laser_frac=0.6, light_frac=0.18, wake_r=60.0, tmax=480.0, flags=None):
    os.environ["BOTDIR"] = botdir
    sys.path.insert(0, botdir)
    sys.path.insert(0, "/home/claude")
    import sim as S  # noqa: E402  (base sim, imports bot from BOTDIR)
    B = S.B
    if flags:
        B.FLAGS.update(flags)
    s = S.Sim(seed, hunters_at=[])
    rng = s.rng
    me = s.me
    zc = S.CFG["zone"]
    house = []
    skills = {"easy": 4.0, "medium": 3.0, "hard": 2.2, "expert": 1.5}
    leads = {"easy": 0.5, "medium": 0.35, "hard": 0.25, "expert": 0.15}
    for i in range(15):
        for _ in range(60):
            x, y = s.rand_free(zc["start_center"], 430)
            if math.dist((x, y), me.pos[:2]) > 160:
                break
        h = S.M(i + 4 if i + 4 != me.id else 19, x, y, s.w)
        h.id = [k for k in range(16) if k != me.id][i]
        t_ = tier_of(i)
        h.tier = t_
        h.skill = math.radians(skills[t_] * skill_scale)
        h.lead_err = leads[t_]
        h.active = False
        r = rng.random()
        if r < laser_frac:
            h.weapons[1] = {"kind": "laser", "cooldown": 0, "charge": 0, "stats": dict(S.CFG["weapons"]["laser"])}
        elif r < laser_frac + light_frac:
            h.weapons[1] = {"kind": "lightning", "cooldown": 0, "stats": dict(S.CFG["weapons"]["lightning"])}
        house.append(h)
        s.mechs.append(h)

    house_ids = {id(h) for h in house}
    orig_damage = s.damage

    def squad_damage(src, tgt, amount, kind, origin_yaw):
        if src is not None and id(src) in house_ids and id(tgt) in house_ids:
            return  # the squad never hurts itself
        orig_damage(src, tgt, amount, kind, origin_yaw)
    s.damage = squad_damage

    def n_should_be_active():
        if s.t >= all_at:
            return 15
        if s.t < first_at:
            return 0
        return int(1 + per_kill * s.kills)

    death = None
    hp_at_end = None
    while s.t < tmax and me.alive:
        # ---- release / wake
        act_n = sum(1 for h in house if h.active or not h.alive)
        want = n_should_be_active()
        for h in house:
            if act_n >= want:
                break
            if h.alive and not h.active:
                h.active = True
                h.active_t = s.t
                act_n += 1
        for h in house:
            if h.alive and not h.active and (math.dist(h.pos[:2], me.pos[:2]) < wake_r or h.hp < h.max_hp):
                h.active = True
                h.active_t = s.t
        if s.tick % 2 == 0:
            o = s.obs_for_me()
            S.mf.draw.clear()
            t0 = time.perf_counter()
            a = s.bot.act(json.loads(json.dumps(o)))
            s.max_act_ms = max(s.max_act_ms, (time.perf_counter() - t0) * 1000)
            me.cmd = a
            if "take" in a:
                for p in list(s.pickups):
                    if p["id"] == a["take"] and math.dist(p["pos"][:2], me.pos[:2]) < 4:
                        if me.weapons[1]:
                            old = me.weapons[1]
                            s.pickups.append({"id": 9500 + s.tick, "kind": old["kind"], "pos": list(me.pos)})
                        me.weapons[1] = {"kind": p["kind"], "cooldown": 1.0, "charge": 0, "stats": dict(S.CFG["weapons"][p["kind"]])}
                        me.events.append({"type": "took", "id": p["id"], "kind": p["kind"], "slot": 1})
                        s.pickups.remove(p)
            for h in house:
                if not h.alive:
                    continue
                if h.active:
                    h.cmd = s.hunter_cmd(h)
                else:  # dormant: drift toward the next circle, hold fire
                    z = s.zone()
                    nc = z["next_center"]
                    dd = math.hypot(nc[0] - h.pos[0], nc[1] - h.pos[1])
                    if dd > max(20.0, z["next_radius"] * 0.6):
                        h.cmd = {"move": [(nc[0] - h.pos[0]) / dd * 0.6, (nc[1] - h.pos[1]) / dd * 0.6],
                                 "aim": [math.atan2(nc[1] - h.pos[1], nc[0] - h.pos[0]), 0], "fire": [False, False]}
                    else:
                        h.cmd = {"move": [0, 0], "aim": h.aim, "fire": [False, False]}
        for m in s.mechs:
            if m.alive:
                s.apply(m, m.cmd)
        for c in list(s.cubes):
            if math.dist(c["pos"][:2], me.pos[:2]) < 3.5:
                s.cubes.remove(c)
                me.cubes += 1
                me.max_hp = S.MC["hp"] * (1 + 0.04 * min(10, me.cubes))
                me.hp += S.MC["hp"] * 0.04
                me.events.append({"type": "pickup", "cubes": me.cubes})
        s.step_bullets()
        s.npc_step()
        for h in house:
            if h.alive:
                izh, dpsh = s.in_zone(h)
                if not izh:
                    h.hp -= dpsh * S.DT
                    if h.hp <= 0:
                        h.alive = False
                        h.zone_death = True
        iz, dps = s.in_zone(me)
        if not iz:
            s.out_zone_s += S.DT
            me.hp -= dps * S.DT
            me.last_dmg = s.t
            if me.hp <= 0:
                me.alive = False
                death = "zone"
        s.t += S.DT
        s.tick += 1
    dead_house = sum(1 for h in house if not h.alive)
    if me.alive:
        alive_house = [h for h in house if h.alive]
        place = 1 + sum(1 for h in alive_house if h.hp > me.hp)
        hp_at_end = me.hp
    else:
        place = 16 - dead_house
    bb = s.bot.bb.summary()
    killer = None
    if not me.alive and bb["damage"]:
        last = [d for d in bb["damage"] if d["t"] >= bb["t_end"] - 3.0]
        by = {}
        for d in last:
            by[d["by"]] = by.get(d["by"], 0) + d["amt"]
        killer = max(by, key=by.get) if by else None
    tiers = {h.id: h.tier for h in house}
    return {
        "seed": seed, "pts": 16 - place, "place": place, "kills": s.kills, "t_end": round(s.t, 1),
        "survived": me.alive, "zone_death": death == "zone", "out_zone_s": round(s.out_zone_s, 1),
        "house_dead": dead_house, "house_zone_dead": sum(1 for h in house if getattr(h, "zone_death", False)),
        "killer_tier": tiers.get(killer), "tiers": tiers,
        "active_at_death": sum(1 for h in house if h.alive and h.active),
        "release_t": {h.id: round(getattr(h, "active_t", -1), 1) for h in house},
        "hp_at_end": hp_at_end, "max_act_ms": round(s.max_act_ms, 1), "bb": bb,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("botdir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--flags", default="{}")
    a = ap.parse_args()
    r = run_match(a.seed, a.botdir, flags=json.loads(a.flags), **json.loads(a.params))
    bb = r.pop("bb")
    r["bb_brief"] = {k: bb[k] for k in ("engage_hp", "regen_s", "low_hp_fight_s", "mode_s", "kills")}
    r["bb_dmg_n"] = len(bb["damage"])
    print(json.dumps(r, default=str))
