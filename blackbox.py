"""blackbox.py - per-match flight recorder for death-cause analysis.

The bot calls BlackBox.tick() once per decision. Nothing is written to disk and
nothing changes what the bot does; the lab harness (lab/) reads `bot.bb` after a
simulated match and lab/autopsy.py turns it into a cause of death.
"""
from collections import deque
import math

RING_S = 12.0  # seconds of full-detail history kept before death


class BlackBox:
    def __init__(self):
        self.ring = deque()          # last RING_S seconds of per-tick state
        self.damage = []             # every damaged event, enriched
        self.hits = []               # (t, amount, target)
        self.kills = []              # (t, target)
        self.first_seen = {}         # enemy id -> (t, hp, weapons)
        self.last_taken = -1e9       # last time we took weapon damage
        self.last_dealt = -1e9       # last time we dealt damage
        self.engage_hp = []          # (t, hp_frac) when a fight starts after >= 3 s of peace
        self._in_fight = False
        self._peace_since = 0.0
        self.regen_s = 0.0           # seconds spent actually regenerating
        self.low_hp_fight_s = 0.0    # seconds fighting below 60% HP
        self.mode_s = {}             # seconds per behaviour mode
        self.t = 0.0

    def tick(self, obs, mode, fire, enemies_vis, n_los):
        t = float(obs["t"])
        me = obs["self"]
        hp, mhp = float(me["hp"]), float(me["max_hp"])
        dt = 0.1
        self.t = t
        x, y = me["pos"][0], me["pos"][1]
        by_pos = {m["id"]: m for m in obs.get("mechs", []) if m.get("kind") == "mech"}
        for m in by_pos.values():
            if m["id"] not in self.first_seen:
                self.first_seen[m["id"]] = (t, m.get("hp"), list(m.get("weapons") or []))
        for e in obs.get("events", []):
            et = e.get("type")
            if et == "damaged":
                w = e.get("weapon")
                a = by_pos.get(e.get("by"))
                dist = math.hypot(a["pos"][0] - x, a["pos"][1] - y) if a else None
                self.damage.append({
                    "t": t, "amt": float(e.get("amount", 0)), "w": w, "by": e.get("by"),
                    "rear": bool(e.get("rear")), "dist": None if dist is None else round(dist, 1),
                    "seen": a is not None, "mode": mode, "hp": round(hp), "nvis": len(enemies_vis),
                    "nlos": n_los, "in_zone": bool(me.get("in_zone", True)),
                })
                if w != "zone":
                    self.last_taken = t
            elif et == "hit":
                self.hits.append((t, float(e.get("amount", 0)), e.get("target")))
                self.last_dealt = t
            elif et == "kill":
                self.kills.append((t, e.get("target")))
        fighting = bool(enemies_vis)
        if fighting and not self._in_fight and t - self._peace_since >= 3.0:
            self.engage_hp.append((round(t, 1), round(hp / mhp, 2)))
        if not fighting and self._in_fight:
            self._peace_since = t
        self._in_fight = fighting
        quiet = t - max(self.last_taken, self.last_dealt)
        if quiet > 5.0 and hp < mhp and me.get("in_zone", True):
            self.regen_s += dt
        if fighting and hp / mhp < 0.6:
            self.low_hp_fight_s += dt
        self.mode_s[mode] = self.mode_s.get(mode, 0.0) + dt
        self.ring.append({
            "t": t, "hp": round(hp), "mode": mode, "pos": [round(x, 1), round(y, 1)],
            "fire": bool(fire[0] or fire[1]), "nvis": len(enemies_vis), "nlos": n_los,
            "quiet": round(min(quiet, 99.0), 1), "alive": obs.get("alive"),
            "en": [[m["id"], round(math.hypot(m["pos"][0] - x, m["pos"][1] - y)), m.get("hp")]
                   for m in enemies_vis[:6]],
        })
        while self.ring and t - self.ring[0]["t"] > RING_S:
            self.ring.popleft()

    def summary(self):
        return {
            "t_end": self.t, "kills": self.kills, "damage": self.damage, "hits_n": len(self.hits),
            "first_seen": self.first_seen, "engage_hp": self.engage_hp, "regen_s": round(self.regen_s, 1),
            "low_hp_fight_s": round(self.low_hp_fight_s, 1),
            "mode_s": {k: round(v, 1) for k, v in self.mode_s.items()}, "last": list(self.ring),
        }
