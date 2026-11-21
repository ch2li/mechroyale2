"""buckeye v12.0 — mechfight/1 bot.

Practice matches are 1 vs a 15-bot squad that hunts you, so points come from (a) not dying to the
zone and (b) killing hunters one at a time before they overwhelm you. This bot:
  * loads the map into a walk grid + A* planner (nav.py) so it does not get stuck on buildings;
  * loots early (better weapon, cubes, crates, NPC drones) while the zone allows it;
  * moves to a spot inside the next zone circle with a wall at its back and watches the open side;
  * fights with lead-aimed gun + charged laser / lightning / lobbed mortar, always facing the
    target (front armour), strafing randomly, kiting to its preferred range, and retreating to
    cover to regenerate when hurt or outnumbered;
  * dodges telegraphed lasers and incoming mortar shells, and unsticks itself with jump-jets.
"""
from __future__ import annotations

import gc
import math
import random
import time
from collections import deque

import mechfight as mf

try:
    from nav import Planner, World
except Exception:  # pragma: no cover
    World = None
    Planner = None

PI = math.pi
TAU = 2 * math.pi
MECH_SOUNDS = {"gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet", "reload"}
# ==================================================================================
# EXPERIMENT SWITCHES - change one line, commit, submit, write down the score.
# Defaults reproduce v1.0 (scored 4.02 / 3.94).
# ==================================================================================
FLAGS = {
    "kite_lightning": False,  # v1.1: back off out of enemy lightning range (45 m)
    "evade_unseen": False,    # v1.1: side-step when hit by an enemy we cannot see
    "zone_guard": False,      # v1.1: fight footwork may not carry us out of the zone
    "hunt_npcs": True,        # shoot NPC drones for cubes early on
    "shoot_crates": True,     # shoot loot crates for cubes / weapons
    "supply_drops": True,     # go capture supply drops
    "retreat_cover": True,    # retreat to cover when hurt or outnumbered
    "range_scale": 1.0,       # multiply preferred fighting distance (e.g. 0.7 closer, 1.4 farther)
    "peek": False,            # v2.0 (scored 3.88): reload behind cover, peek out only to shoot
    "lanchester": False,      # v3.0: fight or disengage by Lanchester square-law strength ratio
    "farm": False,            # v3.0 (scored 3.98): explore early and farm NPC drones / crates
    "ffa_detect": True,       # v4.0: switch to free-for-all doctrine once others die without us
    "force_ffa": False,       # testing only: act as if the match is a free-for-all from tick 0
    "standoff": False,        # tested in the real testbed: worse (10.8 pts vs 12.0 base)
    "no_mortar": False,       # v5: never carry a mortar (lasers cook it off); testbed 13.56 vs 11.5 (n=18/12)
    "dodge_timing": False,    # dodge an enemy laser only in the last moment (house bots fire at exactly 2.75 s)
    "retreat_early": False,   # v6: disengage earlier when outnumbered/losing; no looting while hurt (regen first)
    "lightning_first": False, # v11: prefer lightning over laser (chains through bunched-up hunter groups)
    "isolate": False,         # v11: with 2+ enemies in sight, move to a tactical-map cell only the target can hit
    "tmap": False,            # v10: use the precomputed tactical map (tmap.bin) for cover and camp spots
    "pursue": False,          # v9: no enemy in sight -> go meet the hunters (last seen / heard) instead of camping
    "drop_priority": True,    # v8: supply drops (full HP + 3 cubes) beat almost anything, even mid-fight
    "risk_nav": True,         # v12: routes priced by exposure/zone/chokes; route-length zone timing; vetted overrides; loot only if evac in time
    "pacifist": False,        # v7 probe: never start fights (kills bring more hunters); evade and survive to the end
}

WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
UPGRADE_PRIO = [
    "armor", "plating", "nanites", "laser_damage", "laser_charge", "gun_rate", "gun_damage",
    "lightning_efficiency", "gun_mag", "servos", "mortar_reload", "lightning_range", "gyros",
    "reactor", "capacitor", "jets", "gun_velocity", "lightning_chain", "mortar_splash",
    "mortar_velocity", "laser_pierce", "sensors",
]


def angdiff(a: float, b: float) -> float:
    return (a - b + PI) % TAU - PI


def hyp(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def unit(dx: float, dy: float):
    L = hyp(dx, dy)
    if L < 1e-9:
        return 0.0, 0.0
    return dx / L, dy / L


if FLAGS["no_mortar"]:
    WEAPON_RANK["mortar"] = -1
if FLAGS["lightning_first"]:
    WEAPON_RANK["lightning"], WEAPON_RANK["laser"] = 3, 2
    for _u in ("lightning_chain", "lightning_efficiency", "lightning_range"):
        UPGRADE_PRIO.remove(_u)
        UPGRADE_PRIO.insert(3, _u)


class Bot:
    name = "buckeye-v12.0"

    def __init__(self, hello: mf.Hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.me_id = hello.get("slot", -1)
        self.rng = random.Random(int(hello.get("seed", 0)) & 0xFFFFFFFF)
        self.mech = cfg.get("mech", {})
        self.wcfg = cfg.get("weapons", {})
        self.loot_r = float(cfg.get("loot", {}).get("pickup_radius", 4.0))
        self.size_per_cube = float(cfg.get("cubes", {}).get("size_per_cube", 0.05))
        self.world = None
        self.planner = None
        try:
            if World is not None:
                self.world = World(hello["map"]["dir"], self.mech)
                self.planner = Planner(self.world)
                mf.log(f"map loaded in {self.world.load_s:.2f}s")
        except Exception as e:  # keep playing without the map
            mf.log(f"map load failed: {e!r}")
            self.world = None
            self.planner = None
        # state
        self.t = 0.0
        self.path = None
        self.path_goal = None
        self.path_t = -1e9
        self.force_replan = False
        self._wanted_move = 0.0
        self.want_climb = False
        self.deadline = 0.0
        self._tleft, self._ws, self._zone, self._sd = 1e9, 11.7, None, (0.0, 0.0, 1.0)
        self.ffa = bool(FLAGS.get("force_ffa"))
        self.my_kills = 0
        self.emem = {}          # enemy id -> (t, x, y)
        self.sound_pts = []     # (t, x, y) estimated positions of heard mech sounds
        self._safe = None
        self.strafe = 1.0
        self.strafe_until = 0.0
        self.hist = deque(maxlen=16)  # (t, x, y, move_x, move_y)
        self.unstuck_until = -1.0
        self.unstuck_dir = (0.0, 0.0)
        self.last_hurt_t = -1e9
        self.last_hurt_bearing = None
        self.last_hurt_by = None
        self.hurt_by = {}  # mech id -> last t it hurt us
        self.last_fire_t = -1e9
        self.look = None  # (yaw, until)
        self.target_id = None
        self.camp = None  # (x, y, face_yaw, zone_key)
        self.mode = ""
        self.blacklist = {}  # item id -> until t (unreachable loot)
        self.loot_goal = None  # (id, since_t, start_dist)
        gc.collect()
        try:
            gc.freeze()
        except AttributeError:
            pass

    # ================================================================== helpers
    def ground(self, x, y, fallback):
        if self.world is None:
            return fallback
        return self.world.ground_at(x, y)

    def on_ground_level(self, p) -> bool:
        """Item is at ground level (not on a roof)."""
        if self.world is None:
            return True
        return p[2] - self.world.ground_at(p[0], p[1]) < 2.5

    def los(self, a, b) -> bool:
        if self.world is None:
            return True
        return self.world.los(a, b)

    def free_dir(self, x, y, dx, dy, look=5.0):
        """Rotate a move direction until a short step that way is walkable."""
        if self.world is None or (dx == 0 and dy == 0):
            return dx, dy
        L = hyp(dx, dy)
        ux, uy = dx / L, dy / L
        for rot in (0, 0.45, -0.45, 0.9, -0.9, 1.35, -1.35, 1.9, -1.9, 2.5, -2.5):
            c, s = math.cos(rot), math.sin(rot)
            rx, ry = ux * c - uy * s, ux * s + uy * c
            if self.world.seg_walkable(x, y, x + rx * look, y + ry * look):
                return rx * L, ry * L
        return dx, dy

    # ------------------------------------------------------------------ zone
    @staticmethod
    def zone_at(z, t):
        s0, s1 = z["shrink_start"], z["shrink_end"]
        if t <= s0 or s1 <= s0:
            return z["center"][0], z["center"][1], z["radius"]
        if t >= s1:
            return z["next_center"][0], z["next_center"][1], z["next_radius"]
        f = (t - s0) / (s1 - s0)
        c, n = z["center"], z["next_center"]
        return (c[0] + (n[0] - c[0]) * f, c[1] + (n[1] - c[1]) * f,
                z["radius"] + (z["next_radius"] - z["radius"]) * f)

    def zone_time_left(self, z, x, y, now, margin=4.0):
        """Seconds until (x, y) is outside the zone (inf if never this phase)."""
        cx, cy, r = self.zone_at(z, now)
        if hyp(x - cx, y - cy) > r - margin:
            return 0.0
        end = max(z["shrink_end"], now)
        tt = max(now, z["shrink_start"])
        while tt <= end + 0.01:
            cx, cy, r = self.zone_at(z, tt)
            if hyp(x - cx, y - cy) > r - margin:
                return tt - now
            tt += 1.0
        return 1e9

    def safe_disc(self, z):
        zr = z["next_radius"]
        cx, cy = z["next_center"]
        if zr <= 1:
            return cx, cy, 3.0
        r = max(zr - max(12.0, 0.3 * zr), min(zr * 0.5, 6.0))
        return cx, cy, r

    # ------------------------------------------------------------------ navigation
    def nav_move(self, x, y, gx, gy, gr, now):
        """Unit-ish move vector toward the goal region using the planner."""
        d = hyp(gx - x, gy - y)
        if d <= gr:
            return 0.0, 0.0
        w = self.world
        self.want_climb = False
        if w is None or d < 7.0:
            return unit(gx - x, gy - y)
        mc = w.comp_at(x, y)
        if mc < 0 or w.comp_at(gx, gy) != mc:
            # not reachable on foot (or we are on a roof): head straight there, jet over walls
            ux, uy = unit(gx - x, gy - y)
            if not w.seg_walkable(x, y, x + ux * 6, y + uy * 6):
                self.want_climb = True
            return ux, uy
        risky = False
        if FLAGS["risk_nav"] and w.tmap is not None:
            thr = self.threat_points(now, x, y)
            risky = bool(thr)
            self.planner.risk = self.make_risk(thr, now) if (risky or self._tleft < 60) else None
        # direct if clear and short
        if d < (25 if risky else 70) and w.seg_walkable(x, y, gx, gy):
            return unit(gx - x, gy - y)
        goal = (gx, gy, gr)
        pl = self.planner
        if pl.path is not None and pl.done_t > self.path_t:
            self.path, self.path_goal, self.path_t = pl.path, pl.goal, pl.done_t
        need = (self.path is None or self.path_goal is None or self.force_replan
                or hyp(self.path_goal[0] - gx, self.path_goal[1] - gy) > max(6.0, 0.2 * d)
                or now - self.path_t > (3.0 if risky else 6.0))
        if need and (not pl.busy() or pl.goal is None
                     or hyp(pl.goal[0] - gx, pl.goal[1] - gy) > max(15.0, 0.3 * d)):
            pl.request(x, y, goal, now)
            self.force_replan = False
            pl.run(min(self.deadline, time.perf_counter() + 0.006), now)
            if pl.path is not None and pl.done_t > self.path_t:
                self.path, self.path_goal, self.path_t = pl.path, pl.goal, pl.done_t
        if FLAGS["risk_nav"] and self.path and self.path_t == pl.done_t and self.path_goal is not None:
            sx_, sy_, sr_ = self._sd
            pg = self.path_goal
            st_ = hyp(pg[0] - self.path[0][0], pg[1] - self.path[0][1])
            if hyp(pg[0] - sx_, pg[1] - sy_) < sr_ + 8 and st_ > 25:
                ln = sum(hyp(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.path, self.path[1:]))
                self.route_k = (min(3.0, ln / st_ * 1.1), (round(sx_), round(sy_)))
        path = self.path
        if not path or self.path_goal is None or hyp(self.path_goal[0] - gx, self.path_goal[1] - gy) > max(15.0, 0.3 * d):
            return self.free_dir(x, y, *unit(gx - x, gy - y))
        # closest waypoint
        bi, bd = 0, 1e18
        for i, (px, py) in enumerate(path):
            dd = (px - x) ** 2 + (py - y) ** 2
            if dd < bd:
                bi, bd = i, dd
        if bd > 18 ** 2:
            self.force_replan = True
        # farthest reachable waypoint ahead
        tgt = path[min(bi + 1, len(path) - 1)]
        for k in range(min(len(path) - 1, bi + (3 if risky else 12)), bi, -1):
            px, py = path[k]
            if w.seg_walkable(x, y, px, py):
                tgt = path[k]
                break
        if bi >= len(path) - 1:
            if w.seg_walkable(x, y, gx, gy):
                tgt = (gx, gy)
        return self.free_dir(x, y, *unit(tgt[0] - x, tgt[1] - y))

    # ------------------------------------------------------------------ aiming
    def aim_solution(self, me, tgt, speed=None):
        """[yaw, pitch] from my muzzle to the target's centre, leading for projectile speed."""
        mx, my, mz = me["pos"]
        mz += float(me["stats"].get("muzzle_height", 7))
        tx, ty, tz = tgt["pos"]
        scale = 1 + self.size_per_cube * float(tgt.get("cubes", 0) or 0)
        h = (3.5 if tgt.get("kind") == "npc" else 9.0) * scale
        tz += h * 0.5
        vx, vy, vz = tgt.get("vel", [0, 0, 0])
        lat = 0.06  # our decision latency
        px, py, pz = tx + vx * lat, ty + vy * lat, tz
        if speed:
            T = 0.0
            for _ in range(3):
                T = math.sqrt((px + vx * T - mx) ** 2 + (py + vy * T - my) ** 2 + (pz - mz) ** 2) / speed
            px, py = px + vx * T, py + vy * T
            if abs(vz) > 1.5:
                pz += max(-4.0, min(4.0, vz * T - 0.5 * 22 * T * T))
        dx, dy, dz = px - mx, py - my, pz - mz
        dh = hyp(dx, dy)
        return math.atan2(dy, dx), math.atan2(dz, max(dh, 0.1)), dh, (px, py, pz)

    def mortar_solution(self, me, tgt, high):
        st = self.wstats(me, "mortar")
        v = float(st.get("speed", 72))
        g = float(st.get("gravity", 22))
        mx, my, mz = me["pos"]
        mz += float(me["stats"].get("height", 9)) + 0.5
        tx, ty, tz = tgt["pos"]
        vx, vy, _ = tgt.get("vel", [0, 0, 0])
        T = 1.0
        for _ in range(3):
            px, py = tx + vx * T, ty + vy * T
            d = hyp(px - mx, py - my)
            h = tz + 1.0 - mz
            disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
            if disc < 0:
                return None
            sq = math.sqrt(disc)
            th = math.atan2(v * v + sq, g * d) if high else math.atan2(v * v - sq, g * d)
            T = d / max(0.1, v * math.cos(th))
        return math.atan2(py - my, px - mx), th, d

    def wstats(self, me, kind):
        for wp in me["weapons"]:
            if wp and wp["kind"] == kind:
                return wp.get("stats", {})
        return self.wcfg.get(kind, {})

    # ================================================================== main
    def act(self, obs: mf.Observation) -> mf.Action:
        t_start = time.perf_counter()
        deadline = t_start + 0.010
        self.deadline = deadline
        now = float(obs["t"])
        self.t = now
        me = obs["self"]
        x, y, z = me["pos"]
        hp, mhp = float(me["hp"]), float(me["max_hp"])
        hpf = hp / max(1.0, mhp)
        energy = float(me.get("energy", 0))
        my_yaw, my_pitch = me["aim"]
        stats = me.get("stats", {})
        walk_speed = float(stats.get("walk_speed", 11.7))
        act: dict = {"tick": obs["tick"]}
        zone = obs["zone"]
        rng = self.rng

        # ---- events
        for e in obs.get("events", []):
            et = e.get("type")
            if et == "damaged":
                if e.get("weapon") != "zone":
                    self.last_hurt_t = now
                    self.last_hurt_bearing = e.get("bearing")
                    self.last_hurt_by = e.get("by")
                    if e.get("by") is not None:
                        self.hurt_by[e["by"]] = now
            elif et == "take_failed":
                self.blacklist[e.get("id")] = now + 8.0
            elif et == "took":
                self.loot_goal = None
            elif et == "kill":
                self.my_kills += 1

        # ---- offer
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            best, bi = 1e9, 0
            for i, ch in enumerate(offer["choices"]):
                try:
                    r = UPGRADE_PRIO.index(ch.get("id"))
                except ValueError:
                    r = 50
                if r < best:
                    best, bi = r, i
            act["pick"] = bi

        # ---- weapons
        slots = me.get("weapons", [])
        held = {}
        for i, wp in enumerate(slots):
            if wp:
                held[wp["kind"]] = i
        laser_charging = "laser" in held and float(slots[held["laser"]].get("charge", 0) or 0) > 0
        fire = [False, False]

        # ---- perception
        enemies = [m for m in obs.get("mechs", []) if m.get("kind") == "mech" and m.get("id") != me.get("id")]
        muzzle = (x, y, z + float(stats.get("muzzle_height", 7)))
        for s in obs.get("sounds", []):
            if s.get("kind") in MECH_SOUNDS and s.get("dist") in ("near", "mid") and s.get("bearing") is not None:
                if self.look is None or self.look[1] < now + 1.0:
                    self.look = (float(s["bearing"]), now + 2.0)
        for b in obs.get("beams", []):
            p0 = b.get("path", [[x, y, z]])[0]
            self.look = (math.atan2(p0[1] - y, p0[0] - x), now + 2.5)
        if self.last_hurt_bearing is not None and now - self.last_hurt_t < 0.3:
            self.look = (float(self.last_hurt_bearing), now + 3.0)

        # ---- zone
        zcx, zcy, zr_now = self.zone_at(zone, now)
        scx, scy, sr = self.safe_disc(zone)
        d_safe = hyp(x - scx, y - scy) - sr
        tleft = self.zone_time_left(zone, x, y, now)
        self._ws, self._zone, self._sd = walk_speed, zone, (scx, scy, sr)
        rk = 1.35
        if FLAGS["risk_nav"]:
            rr = getattr(self, "route_k", None)
            if rr is not None and rr[1] == (round(scx), round(scy)):
                rk = max(rk, rr[0])
        travel = max(0.0, d_safe) * rk / max(1.0, walk_speed)
        zone_urgent = (not me.get("in_zone", True)) or tleft < travel + 12.0
        self._tleft = tleft

        # ---- stuck detection: we asked to move a fair distance but barely moved
        lm = getattr(self, "_last_move", (0.0, 0.0))
        self.hist.append((now, x, y, lm[0], lm[1]))
        stuck = False
        if len(self.hist) >= 12 and me.get("grounded", True):
            t0_, x0_, y0_ = self.hist[0][:3]
            ex_ = sum(h[3] for h in list(self.hist)[1:]) * walk_speed * 0.1 * 0.5
            ey_ = sum(h[4] for h in list(self.hist)[1:]) * walk_speed * 0.1 * 0.5
            want = hyp(ex_, ey_)
            got = hyp(x - x0_, y - y0_)
            if now - t0_ > 1.0 and want > 4.0 and got < 0.25 * want:
                stuck = True
        if stuck and now > self.unstuck_until + 0.5:
            a = rng.uniform(-PI, PI)
            self.unstuck_dir = (math.cos(a), math.sin(a))
            if FLAGS["risk_nav"] and hasattr(self, "_zone"):
                self.unstuck_dir = self.pick_move(x, y, [(math.cos(a + k * PI / 4), math.sin(a + k * PI / 4)) for k in range(8)], now)
            self.unstuck_until = now + 0.9
            self.force_replan = True
            self.hist.clear()

        # ---- free-for-all detection: in practice the squad never kills each other
        if (not self.ffa and FLAGS["ffa_detect"] and 2.0 < now < 150
                and int(obs.get("alive", 16)) + self.my_kills < int(self.cfg.get("match", {}).get("mechs", 16))):
            self.ffa = True
            mf.log(f"FFA detected at t={now:.1f}")
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech" and m.get("id") != me.get("id"):
                self.emem[m["id"]] = (now, m["pos"][0], m["pos"][1])
        for s in obs.get("sounds", []):
            if s.get("kind") in MECH_SOUNDS and s.get("bearing") is not None and s.get("dist") in ("near", "mid"):
                r_ = 45.0 if s["dist"] == "near" else 110.0
                self.sound_pts.append((now, x + r_ * math.cos(s["bearing"]), y + r_ * math.sin(s["bearing"])))
        if len(self.sound_pts) > 40:
            self.sound_pts = self.sound_pts[-40:]

        # ================================================================ decide
        self.want_climb = False
        move = (0.0, 0.0)
        boost = False
        jump = False
        aim = None
        mode = ""
        target = None

        if enemies:
            mode = "fight"
            # ---- target selection
            best, bs = None, 1e18
            for m in enemies:
                mp = m["pos"]
                d = math.dist((x, y), mp[:2])
                vis = self.los(muzzle, (mp[0], mp[1], mp[2] + 4.5))
                m["_d"], m["_los"] = d, vis
                s = d / 120.0 + 1.6 * float(m.get("hp", 1400)) / max(1.0, float(m.get("max_hp", 1400)))
                if vis:
                    s -= 1.5
                if m["id"] == self.target_id:
                    s -= 0.4
                if now - self.hurt_by.get(m["id"], -1e9) < 2.0:
                    s -= 0.5
                if s < bs:
                    bs, best = s, m
            target = best
            self.target_id = target["id"]
            td, tlos = target["_d"], target["_los"]
            n_los = sum(1 for m in enemies if m["_los"])

            # ---- aim + fire
            gun_spd = float(self.wstats(me, "gun").get("speed", 252)) if "gun" in held else None
            ayaw, apitch, dh, _ = self.aim_solution(me, target, gun_spd)
            lyaw, lpitch, _, _ = self.aim_solution(me, target, None)
            scale = 1 + self.size_per_cube * float(target.get("cubes", 0) or 0)
            rad = 2.5 * scale
            want = (ayaw, apitch)
            if "laser" in held:
                lw = slots[held["laser"]]
                lst = lw.get("stats", {})
                full = float(lst.get("charge", 0.75))
                ch = float(lw.get("charge", 0) or 0)
                if ch >= full - 0.2 or "gun" not in held:
                    want = (lyaw, lpitch)
                lerr_y = abs(angdiff(my_yaw, lyaw)) * dh
                lerr_p = abs(my_pitch - lpitch) * dh
                laser_ok = lerr_y < rad * 0.8 and lerr_p < 3.5 * scale
                if ch > 0:
                    forced = ch >= full + float(lst.get("hold_s", 2.0)) - 0.25
                    if (ch >= full - 0.04 and laser_ok and tlos) or forced:
                        fire[held["laser"]] = False  # release -> fires
                    else:
                        fire[held["laser"]] = True
                elif tlos and td < float(lst.get("range", 280)) * 0.95 and float(lw.get("cooldown", 0)) <= 0.1:
                    fire[held["laser"]] = True
            if "gun" in held:
                gw = slots[held["gun"]]
                gst = gw.get("stats", {})
                err_y = abs(angdiff(my_yaw, ayaw)) * dh
                err_p = abs(my_pitch - apitch) * dh
                if tlos and td < float(gst.get("range", 230)) * 0.97 and err_y < rad * 1.25 and err_p < 3.5 * scale:
                    if float(gw.get("ammo", 1)) > 0 and float(gw.get("reload", 0)) <= 0:
                        fire[held["gun"]] = True
                ammo = float(gw.get("ammo", 1))
                if float(gw.get("reload", 0)) <= 0 and (ammo <= 0 or (not tlos and ammo < float(gst.get("mag", 12)) * 0.5)):
                    act["reload"] = True
            if "lightning" in held:
                lw = slots[held["lightning"]]
                lst = lw.get("stats", {})
                cone = math.radians(float(lst.get("cone_deg", 30))) * 0.45
                if (td < float(lst.get("range", 45)) * 0.95 and abs(angdiff(my_yaw, lyaw)) < cone and tlos
                        and energy > float(lst.get("energy", 3)) + 8):
                    fire[held["lightning"]] = True
                if "gun" not in held and "laser" not in held:
                    want = (lyaw, lpitch)
            if "mortar" in held:
                mw = slots[held["mortar"]]
                sol = self.mortar_solution(me, target, high=not tlos)
                if sol is not None:
                    myaw, mpitch, md = sol
                    if not tlos or ("gun" not in held and "laser" not in held):
                        want = (myaw, mpitch)
                    if (float(mw.get("cooldown", 0)) <= 0 and 22 < md < 230
                            and abs(angdiff(my_yaw, myaw)) * md < 4.0 and abs(my_pitch - mpitch) < 0.03):
                        fire[held["mortar"]] = True
            aim = want
            if fire[0] or fire[1]:
                self.last_fire_t = now

            # ---- movement: kite + strafe
            has_light = "lightning" in held
            light_rng = float(self.wstats(me, "lightning").get("range", 45)) if has_light else 0.0
            if has_light and energy > 25 and n_los <= 1:
                pref = light_rng * 0.55  # close in: lightning is the best dps we have
            elif "laser" in held:
                pref = 90.0
            elif "mortar" in held:
                pref = 110.0
            else:
                pref = 60.0
            if FLAGS["standoff"] and "laser" in held and n_los <= 2:
                pref = 165.0  # house bots: gun hits 19% at 150-180 m, laser <=10%; our laser is hitscan
            pref *= float(FLAGS["range_scale"])
            thp = float(target.get("hp", 1400))
            winning = thp <= hp * 1.1 or thp < 260
            hurt = (not winning) and (hpf < 0.35 or (n_los >= 2 and hpf < 0.6) or (len(enemies) >= 3 and hpf < 0.8))
            if n_los >= 3 and hpf < 0.5:
                hurt = True
            if FLAGS["retreat_early"] and not winning:
                if hpf < 0.55 or (n_los >= 2 and hpf < 0.85) or len(enemies) >= 3:
                    hurt = True
            if FLAGS["lanchester"]:
                ratio = self.lanchester(me, held, enemies, hp)
                hurt = ratio < 0.75 and not (thp < 200 and target["_los"])
                mode = "fight" if not hurt else "fight-" + ("%.1f" % ratio)
            if hurt:
                pref = 140.0
            ex, ey = unit(target["pos"][0] - x, target["pos"][1] - y)
            radial = 0.0
            if td < pref - 15:
                radial = -1.0
            elif td > pref + (8 if pref < 40 else 30) and not hurt and (len(enemies) <= 2 or winning):
                radial = 0.8 if pref < 40 else 0.6
            if now > self.strafe_until:
                self.strafe = rng.choice((-1.0, 1.0))
                self.strafe_until = now + rng.uniform(0.45, 1.3)
            tx_, ty_ = -ey * self.strafe, ex * self.strafe
            side = 0.85
            # keep out of an enemy lightning's reach unless we are clearly winning a close fight
            l_rng = float(self.wcfg.get("lightning", {}).get("range", 45))
            zap = [m for m in enemies if "lightning" in (m.get("weapons") or []) and m["_d"] < l_rng + 25]
            if not FLAGS["kite_lightning"]:
                zap = []
            if zap and not (has_light and winning and energy > 25):
                zd = min(m["_d"] for m in zap)
                if zd < l_rng + 12:
                    zm = min(zap, key=lambda m: m["_d"])
                    ex, ey = unit(zm["pos"][0] - x, zm["pos"][1] - y)
                    tx_, ty_ = -ey * self.strafe, ex * self.strafe
                    radial, side = -1.0, 0.35
                    boost = energy > 20 and zd < l_rng + 4
                    mode += "+kite"
                elif radial > 0:
                    radial = 0.0
            if FLAGS["standoff"] and "laser" in held and pref >= 150 and td < 110 and energy > 35:
                boost = True  # open the range quickly
            mvx = ex * radial + tx_ * side
            mvy = ey * radial + ty_ * side
            if hurt and self.world is not None and FLAGS["retreat_cover"]:
                cov = self.find_cover(x, y, enemies, zone, now)
                if cov is not None:
                    cx_, cy_ = self.nav_move(x, y, cov[0], cov[1], 2.0, now)
                    mvx, mvy = cx_ + tx_ * 0.25, cy_ + ty_ * 0.25
                    mode = "cover"
                boost = energy > (45 if has_light else 20)
            elif has_light and pref < 40 and tlos and light_rng < td < 130 and energy > 70:
                boost = True  # close the gap fast
            elif (FLAGS["isolate"] and n_los >= 2 and self.world is not None and self.world.tmap is not None
                  and (iso := self.isolate_cell(x, y, target, enemies, zone, now)) is not None):
                ix_, iy_ = self.nav_move(x, y, iso[0], iso[1], 2.0, now)
                mvx, mvy = ix_ + tx_ * 0.2, iy_ + ty_ * 0.2
                mode = "isolate"
            elif not tlos and not hurt and len(enemies) == 1 and hpf > 0.55:
                # hold the angle: stay put near cover, aim at where it will appear
                mvx, mvy = tx_ * 0.3, ty_ * 0.3
                mode = "hold-angle"
            # ---- peek & duck: expose ourselves only while a weapon is ready to fire
            if FLAGS["peek"] and self.world is not None and not mode.startswith("cover"):
                g_ok = l_ok = z_ok = False
                if "gun" in held:
                    gw_ = slots[held["gun"]]
                    g_ok = float(gw_.get("ammo", 0)) >= 2 and float(gw_.get("reload", 0)) <= 0
                if "laser" in held:
                    lw_ = slots[held["laser"]]
                    l_ok = float(lw_.get("charge", 0) or 0) > 0 or float(lw_.get("cooldown", 0)) <= 0.4
                if "lightning" in held:
                    z_ok = energy > 20 and td < light_rng + 10
                ready = g_ok or l_ok or z_ok
                seen_by = [m for m in enemies if m["_los"] or m["_d"] < 45]
                spot = None
                if not ready and seen_by:
                    spot = self.find_spot(x, y, None, seen_by, zone, now, "duck")
                    if spot is not None:
                        mode = "duck"
                elif ready and not tlos and target["_d"] < 160:
                    tv = target.get("vel", [0, 0, 0])
                    closing = -(tv[0] * ex + tv[1] * ey)  # >0: it is walking toward us
                    if closing < 3.0:
                        others = [m for m in enemies if m["id"] != target["id"]]
                        spot = self.find_spot(x, y, target, others, zone, now, "peek")
                        if spot is not None:
                            mode = "peek"
                if spot is not None:
                    px_, py_ = self.nav_move(x, y, spot[0], spot[1], 1.0, now)
                    mvx, mvy = px_, py_
                    if mode == "duck" and "gun" in held:
                        gw_ = slots[held["gun"]]
                        if float(gw_.get("reload", 0)) <= 0 and float(gw_.get("ammo", 12)) < float(gw_.get("stats", {}).get("mag", 12)):
                            act["reload"] = True  # reload behind cover
            # fight our way onto a supply drop: full HP + 3 cubes is worth more than holding ground
            if FLAGS["drop_priority"] and hpf < 0.85 and min(m["_d"] for m in enemies) > 45:
                for dr in obs.get("drops", []):
                    dd = hyp(dr["pos"][0] - x, dr["pos"][1] - y)
                    hold_ = dr.get("holder")
                    if dd < 200 and dr.get("lands_in", 0) < dd / walk_speed + 2 and (hold_ is None or hold_ == me.get("id")):
                        gx_, gy_ = self.nav_move(x, y, dr["pos"][0], dr["pos"][1], float(dr.get("radius", 8)) * 0.4, now)
                        mvx, mvy = gx_ + tx_ * 0.3, gy_ + ty_ * 0.3
                        mode = "fight-drop"
                        boost = energy > 30 and dd > 20
                        break
            # stay inside zone
            if tleft < 10 or not me.get("in_zone", True):
                zx, zy = self.nav_move(x, y, scx, scy, sr, now)
                mvx, mvy = mvx * 0.4 + zx, mvy * 0.4 + zy
            if self.ffa:
                mvx, mvy, boost, mode = self.ffa_fight(x, y, me, hpf, energy, target, enemies, held, zone, now,
                                                       scx, scy, sr, tleft, mvx, mvy, tx_, ty_, boost)
            move = self.free_dir(x, y, mvx, mvy)
            # dodge a laser that is being charged at us
            cs = self.__dict__.setdefault("charge_start", {})
            for m in enemies:
                if m.get("charging"):
                    cs.setdefault(m["id"], now)
                else:
                    cs.pop(m["id"], None)
            for m in enemies:
                if m.get("charging"):
                    if FLAGS["dodge_timing"]:
                        el = now - cs.get(m["id"], now)
                        if not (2.4 <= el <= 3.0):
                            continue  # house bots always release at 2.75 s: only dodge then
                    by = math.atan2(y - m["pos"][1], x - m["pos"][0])
                    if abs(angdiff(m["aim"][0], by)) < 0.12:
                        bx, byy = unit(-math.sin(by) * self.strafe, math.cos(by) * self.strafe)
                        move = self.free_dir(x, y, bx, byy)
                        boost = energy > 15
                        mode += "+dodge"
                        break
        elif (FLAGS["evade_unseen"] and self.last_hurt_by is not None and self.last_hurt_by < 100 and now - self.last_hurt_t < 2.0
              and self.last_hurt_bearing is not None):
            # shot by someone we cannot see: do not sit still, side-step out of the line, face the shooter
            self.target_id = None
            mode = "evade"
            bq = float(self.last_hurt_bearing)
            if now > self.strafe_until:
                self.strafe = rng.choice((-1.0, 1.0))
                self.strafe_until = now + rng.uniform(0.6, 1.4)
            ax_, ay_ = math.cos(bq), math.sin(bq)
            mv = (-ay_ * self.strafe - ax_ * 0.5, ax_ * self.strafe - ay_ * 0.5)
            if tleft < 10 or not me.get("in_zone", True):
                zx, zy = self.nav_move(x, y, scx, scy, sr, now)
                mv = (mv[0] * 0.4 + zx, mv[1] * 0.4 + zy)
            move = self.free_dir(x, y, *mv)
            aim = (bq, 0.0)
            boost = energy > 40
        elif self.ffa:
            self.target_id = None
            near_threat = any(hyp(px - x, py - y) < 110 for (px, py, wt) in self.threat_points(now, x, y) if wt >= 1.0)
            if near_threat:
                mode, move, aim = self.ffa_roam(obs, me, x, y, z, hpf, energy, held, slots, act, zone, zone_urgent,
                                                scx, scy, sr, tleft, now)
            else:
                mode, move, aim, fire = self.peaceful(obs, me, x, y, z, hpf, held, slots, act, zone,
                                                      zone_urgent, scx, scy, sr, now)
                if mode in ("camp", "to-camp"):
                    mode, move, aim = self.ffa_roam(obs, me, x, y, z, hpf, energy, held, slots, act, zone, zone_urgent,
                                                    scx, scy, sr, tleft, now)
            boost = (zone_urgent and d_safe > 20 and energy > 45) or (mode == "ffa-hurt" and energy > 40)
        else:
            self.target_id = None
            mode, move, aim, fire_loot = self.peaceful(obs, me, x, y, z, hpf, held, slots, act, zone,
                                                       zone_urgent, scx, scy, sr, now)
            fire = fire_loot
            if zone_urgent and d_safe > 20 and energy > 45 and not laser_charging:
                boost = True

        # ---- never let fight / evade footwork carry us out of the zone
        if FLAGS["zone_guard"] and mode.startswith(("fight", "cover", "hold", "evade")) and hyp(*move) > 0.1:
            lx, ly = x + move[0] * 12, y + move[1] * 12
            if hyp(lx - zcx, ly - zcy) > zr_now - 10 or (tleft < 15 and hyp(lx - scx, ly - scy) > hyp(x - scx, y - scy)):
                ix, iy = unit(zcx - x, zcy - y) if hyp(lx - zcx, ly - zcy) > zr_now - 10 else unit(scx - x, scy - y)
                dot = move[0] * ix + move[1] * iy
                if dot < 0.3:
                    move = (move[0] - (dot - 0.3) * ix, move[1] - (dot - 0.3) * iy)
                    move = self.free_dir(x, y, *move)

        # ---- grab a better weapon we are standing on (any mode)
        if "take" not in act and not laser_charging and self.ffa:
            for p in obs.get("pickups", []):
                ds = self.ffa_swap(p.get("kind"), slots, held)
                if ds is None:
                    continue
                if hyp(p["pos"][0] - x, p["pos"][1] - y) < self.loot_r - 0.3 and abs(p["pos"][2] - z) < 3.8:
                    act["take"] = int(p["id"])
                    if ds >= 0:
                        act["drop"] = ds
                    break
        if "take" not in act and not laser_charging and not self.ffa:
            s1 = slots[1] if len(slots) > 1 else None
            rank1 = WEAPON_RANK.get(s1["kind"], 0) if s1 else -1
            for p in obs.get("pickups", []):
                k = p.get("kind")
                if k in held or WEAPON_RANK.get(k, 0) <= rank1:
                    continue
                if hyp(p["pos"][0] - x, p["pos"][1] - y) < self.loot_r - 0.3 and abs(p["pos"][2] - z) < 3.8:
                    act["take"] = int(p["id"])
                    if s1 is not None:
                        act["drop"] = 1
                    break

        # ---- pacifist doctrine: do not hunt; slip away from every mech, shoot only in self-defence
        if FLAGS["pacifist"] and enemies:
            near = min(m["_d"] for m in enemies)
            if near > 30:
                fire = [False, False]
            if not laser_charging or near > 30:
                sp = self.safe_point(x, y, zone, now, scx, scy, sr, tleft < 15,
                                     force=(self._safe is None or now - self._safe[2] > 1.5))
                mv = self.nav_move(x, y, sp[0], sp[1], 4.0, now)
                move = self.free_dir(x, y, *mv) if hyp(*mv) > 0.05 else move
                boost = energy > 35 and near < 140
                mode = "evade"
            if target is not None:
                aim = (math.atan2(target["pos"][1] - y, target["pos"][0] - x), aim[1] if aim else 0.0)

        # ---- mortar shell dodge (always)
        dodge = self.shell_dodge(obs, x, y, z)
        if dodge is not None:
            if FLAGS["risk_nav"]:
                a0 = math.atan2(dodge[1], dodge[0])
                dodge = self.pick_move(x, y, [dodge] + [(math.cos(a0 + da), math.sin(a0 + da)) for da in (1.0, -1.0)], now)
            move = self.free_dir(x, y, *dodge)
            boost = energy > 10
            mode += "+shell"

        # ---- climb over a wall toward an unreachable goal (jump, then jet while pushing on)
        if getattr(self, "want_climb", False) and (mode in ("zone", "to-camp", "drop") or tleft < 15):
            if me.get("grounded", False):
                jump = energy > 35
            else:
                boost = energy > 3
            mode += "+climb"

        # ---- unstuck override
        if now < self.unstuck_until:
            ux, uy = self.unstuck_dir
            base = move if hyp(*move) > 0.1 else (ux, uy)
            move = unit(base[0] + ux, base[1] + uy)
            jump = me.get("grounded", False)
            boost = energy > 8
            mode += "+unstuck"

        # ---- aim default
        if aim is None:
            yaw = my_yaw
            if self.look is not None and now < self.look[1]:
                yaw = self.look[0]
            elif hyp(*move) > 0.2:
                yaw = math.atan2(move[1], move[0])
            elif self.camp is not None:
                yaw = self.camp[2] + 0.7 * math.sin(now * 0.6)
            aim = (yaw, 0.0)
        pmin = float(stats.get("min_pitch", -0.7))
        pmax = float(stats.get("max_pitch", 1.3))
        aim = (aim[0], max(pmin, min(pmax, aim[1])))

        L = hyp(*move)
        if L > 1:
            move = (move[0] / L, move[1] / L)
        self._wanted_move = hyp(*move)
        self._last_move = move
        act["move"] = [round(move[0], 4), round(move[1], 4)]
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)
        act["fire"] = fire
        self.mode = mode

        # ---- planner gets the leftover time
        if self.planner is not None and self.planner.busy():
            self.planner.run(deadline, now)
        mf.draw.label(f"{mode} hp{hp:.0f} e{energy:.0f} {1000 * (time.perf_counter() - t_start):.0f}ms")
        if target is not None:
            tp = target["pos"]
            mf.draw.line([x, y, z + 7], [tp[0], tp[1], tp[2] + 4.5], "#ff5040")
        if self.path:
            for (px, py) in self.path[:: max(1, len(self.path) // 20)]:
                mf.draw.point([px, py, z], "#40c0ff", 3)
        return act

    # ================================================================== peaceful behaviour
    def peaceful(self, obs, me, x, y, z, hpf, held, slots, act, zone, zone_urgent, scx, scy, sr, now):
        fire = [False, False]
        aim = None
        cands = []  # (cost, kind, obj)
        slot1 = slots[1] if len(slots) > 1 else None
        cur_rank = WEAPON_RANK.get(slot1["kind"], 0) if slot1 else -1
        max_detour = 25.0 if zone_urgent else 140.0
        for p in obs.get("pickups", []):
            if self.blacklist.get(p["id"], -1) > now:
                continue
            k = p.get("kind")
            if self.ffa:
                if self.ffa_swap(k, slots, held) is None:
                    continue
            elif k in held or WEAPON_RANK.get(k, 0) <= cur_rank:
                continue
            if not self.on_ground_level(p["pos"]):
                continue
            d = hyp(p["pos"][0] - x, p["pos"][1] - y)
            lim = max_detour
            if FLAGS["standoff"] and k == "laser" and "laser" not in held and not zone_urgent:
                lim = 260.0  # the whole doctrine depends on having a laser
            if d < lim:
                cands.append((d - 60 * WEAPON_RANK.get(k, 0) - 40, "pickup", p))
        for c in obs.get("cubes", []):
            if self.blacklist.get(c["id"], -1) > now or not self.on_ground_level(c["pos"]):
                continue
            d = hyp(c["pos"][0] - x, c["pos"][1] - y)
            if d < max_detour:
                cands.append((d - 70, "cube", c))
        if FLAGS["shoot_crates"]:
            for c in obs.get("crates", []):
                if self.blacklist.get(c["id"], -1) > now or not self.on_ground_level(c["pos"]):
                    continue
                d = hyp(c["pos"][0] - x, c["pos"][1] - y)
                if d < (40 if zone_urgent else (170 if FLAGS["farm"] else 110)):
                    cands.append((d - 25, "crate", c))
        if FLAGS["hunt_npcs"]:
            if hpf > 0.6 and now < 260:
                for n in npcs_of(obs):
                    d = hyp(n["pos"][0] - x, n["pos"][1] - y)
                    if d < (50 if zone_urgent else (170 if FLAGS["farm"] else 90)):
                        cands.append((d - 20, "npc", n))
        # NPC that is shooting us
        if self.last_hurt_by is not None and self.last_hurt_by >= 100 and now - self.last_hurt_t < 2:
            for n in npcs_of(obs):
                if n["id"] == self.last_hurt_by:
                    cands.append((-200, "npc", n))
        # supply drop
        if FLAGS["supply_drops"]:
            for dr in obs.get("drops", []):
                d = hyp(dr["pos"][0] - x, dr["pos"][1] - y)
                eta = d / max(1.0, float(me["stats"].get("walk_speed", 11.7)))
                holder = dr.get("holder")
                rng_ = 350 if FLAGS["drop_priority"] else 220
                if d < rng_ and dr.get("lands_in", 0) < eta + 6 and (holder is None or holder == me.get("id")):
                    cands.append((d - (300 if FLAGS["drop_priority"] else 120), "drop", dr))
        if FLAGS["retreat_early"] and hpf < 0.6:
            cands = [c for c in cands if c[1] in ("cube", "pickup") and hyp(c[2]["pos"][0] - x, c[2]["pos"][1] - y) < 25]
        if FLAGS["risk_nav"] and self._tleft < 90:
            ws = max(1.0, self._ws)
            ok = []
            for c in cands:
                if c[0] <= -200:
                    ok.append(c)
                    continue
                px, py = c[2]["pos"][0], c[2]["pos"][1]
                eta = hyp(px - x, py - y) * 1.3 / ws
                back = max(0.0, hyp(px - scx, py - scy) - sr) * 1.35 / ws
                if self.zone_time_left(zone, px, py, now) > eta + 0.5 * back + 6.0:
                    ok.append(c)
            cands = ok
        cands.sort(key=lambda c: c[0])
        # stick with current loot goal (avoid dithering), give up if no progress
        if self.loot_goal is not None:
            gid, since, d0 = self.loot_goal
            if now - since > 25:
                self.blacklist[gid] = now + 30
                self.loot_goal = None
        if cands:
            _, kind, o = cands[0]
            if self.loot_goal is None or self.loot_goal[0] != o["id"]:
                self.loot_goal = (o["id"], now, 0)
            ox, oy, oz = o["pos"]
            d = hyp(ox - x, oy - y)
            if kind == "pickup":
                if d < self.loot_r - 0.4 and not ("laser" in held and float(slots[held["laser"]].get("charge", 0) or 0) > 0):
                    act["take"] = int(o["id"])
                    if self.ffa:
                        ds = self.ffa_swap(o.get("kind"), slots, held)
                        if ds is not None and ds >= 0:
                            act["drop"] = ds
                    elif all(slots[:2]) and len(slots) >= 2:
                        act["drop"] = 1
                mv = self.nav_move(x, y, ox, oy, 1.0, now) if d > 1.0 else (0.0, 0.0)
                return "loot-" + o["kind"], mv, aim, fire
            if kind == "cube":
                mv = self.nav_move(x, y, ox, oy, 0.5, now)
                return "cube", mv, aim, fire
            if kind == "drop":
                r = float(o.get("radius", 8)) * 0.5
                mv = self.nav_move(x, y, ox, oy, r, now)
                return "drop", mv, aim, fire
            # shoot crate / npc
            tgt = dict(o)
            tgt.setdefault("kind", "crate" if kind == "crate" else "npc")
            if kind == "crate":
                tgt["kind"] = "npc"  # small target height
            muzzle = (x, y, z + float(me["stats"].get("muzzle_height", 7)))
            vis = self.los(muzzle, (ox, oy, oz + 1.2))
            gun_spd = float(self.wstats(me, "gun").get("speed", 252))
            ayaw, apitch, dh, _ = self.aim_solution(me, tgt, gun_spd)
            if kind == "crate":
                apitch = math.atan2(oz + 1.2 - muzzle[2], max(0.1, dh))
            aim = (ayaw, apitch)
            my_yaw, my_pitch = me["aim"]
            ok = abs(angdiff(my_yaw, ayaw)) * dh < 1.3 and abs(my_pitch - apitch) * dh < 1.5
            if vis and ok and dh < 150:
                if "gun" in held:
                    gw = slots[held["gun"]]
                    if float(gw.get("ammo", 1)) > 0:
                        fire[held["gun"]] = True
                    elif float(gw.get("reload", 0)) <= 0:
                        act["reload"] = True
                if "lightning" in held and dh < 40 and float(me.get("energy", 0)) > 40:
                    fire[held["lightning"]] = True
                if "laser" in held and kind == "npc":
                    lw = slots[held["laser"]]
                    ch = float(lw.get("charge", 0) or 0)
                    full = float(lw.get("stats", {}).get("charge", 0.75))
                    fire[held["laser"]] = not (ch >= full - 0.04)
            standoff = 8.0 if kind == "crate" else (60.0 if FLAGS["farm"] else 30.0)
            if d > standoff or not vis:
                mv = self.nav_move(x, y, ox, oy, standoff if vis else 3.0, now)
            else:
                mv = (0.0, 0.0)
            return "shoot-" + kind, mv, aim, fire
        self.loot_goal = None
        # reload in peace
        if "gun" in held:
            gw = slots[held["gun"]]
            if float(gw.get("ammo", 12)) < float(gw.get("stats", {}).get("mag", 12)) * 0.6 and float(gw.get("reload", 0)) <= 0:
                act["reload"] = True
        # explore to find drones / crates while the zone allows it
        if FLAGS["farm"] and not zone_urgent and now < 210 and hpf > 0.5 and self.world is not None:
            wp = self.explore_goal(x, y, scx, scy, sr, now)
            if wp is not None:
                mv = self.nav_move(x, y, wp[0], wp[1], 8.0, now)
                if hyp(*mv) > 0.1:
                    yaw = math.atan2(mv[1], mv[0]) + 1.1 * math.sin(now * 1.8)  # sweep the view
                    return "explore", mv, (yaw, 0.0), fire
        # pursue: meet the next hunter early instead of waiting for it (practice score ~ kills per minute)
        if FLAGS["pursue"] and not zone_urgent and hpf > 0.55:
            pts = [(px, py, wt) for (px, py, wt) in self.threat_points(now, x, y)]
            best_p, bd = None, 1e9
            zx_, zy_, zr_ = self.zone_at(zone, now + 15)
            for (px, py, wt) in pts:
                d = hyp(px - x, py - y)
                if 25 < d < 260 and hyp(px - zx_, py - zy_) < zr_ - 10 and d < bd:
                    best_p, bd = (px, py), d
            if best_p is not None:
                mv = self.nav_move(x, y, best_p[0], best_p[1], 15.0, now)
                return "pursue", mv, (math.atan2(best_p[1] - y, best_p[0] - x), 0.0), fire
        # go to (or hold) camp spot inside the safe disc
        key = (round(scx), round(scy), round(sr))
        if self.camp is None or self.camp[3] != key:
            self.camp = self.pick_camp(x, y, scx, scy, sr, key)
        cx, cy, face, _ = self.camp
        d = hyp(cx - x, cy - y)
        if d > 2.5:
            mv = self.nav_move(x, y, cx, cy, 2.0, now)
            return ("zone" if zone_urgent else "to-camp"), mv, aim, fire
        return "camp", (0.0, 0.0), aim, fire

    def explore_goal(self, x, y, cx, cy, r, now):
        """Waypoint toward the nearest not-yet-seen part of the safe disc."""
        vis = getattr(self, "_visited", None)
        if vis is None:
            vis = self._visited = set()
        ci, cj = int(x // 40), int(y // 40)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                vis.add((ci + di, cj + dj))
        g = getattr(self, "_explore", None)
        if g is not None and now - g[2] < 20 and hyp(g[0] - x, g[1] - y) > 15 and (int(g[0] // 40), int(g[1] // 40)) not in vis:
            return g
        w = self.world
        my_comp = w.comp_at(x, y)
        best, bs = None, 1e18
        rng = self.rng
        for _ in range(60):
            a = rng.uniform(-PI, PI)
            rr = r * math.sqrt(rng.random())
            px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
            if (int(px // 40), int(py // 40)) in vis or not w.walkable(px, py):
                continue
            if my_comp >= 0 and w.comp_at(px, py) != my_comp:
                continue
            d = hyp(px - x, py - y)
            score = abs(d - 110) + 0.2 * rr
            if score < bs:
                bs, best = score, (px, py, now)
        self._explore = best
        return best

    def pick_camp(self, x, y, cx, cy, r, key):
        """A walkable spot inside the safe disc with a wall close behind it."""
        w = self.world
        if w is None:
            return (cx, cy, math.atan2(y - cy, x - cx), key)
        best, bs = None, 1e18
        rng = self.rng
        tries = 260 if r > 15 else 60
        my_comp = w.comp_at(x, y)
        # bias toward the side of the disc nearest to us
        for _ in range(tries):
            a = rng.uniform(-PI, PI)
            rr = r * math.sqrt(rng.random())
            px, py = cx + rr * math.cos(a), cy + rr * math.sin(a)
            if not w.walkable(px, py):
                continue
            if my_comp >= 0 and w.comp_at(px, py) != my_comp:
                continue
            walls = []
            for k in range(8):
                ang = k * PI / 4
                qx, qy = px + 4.5 * math.cos(ang), py + 4.5 * math.sin(ang)
                kk = w.fidx(qx, qy)
                if kk >= 0 and w.blocked[kk]:
                    walls.append(ang)
            dme = hyp(px - x, py - y)
            score = dme * 0.6 + rr * 0.25
            if walls:
                score -= 60
            if FLAGS["tmap"] and w.tmap is not None:
                cc = w.cidx(px, py)
                if cc >= 0:
                    score += 3.0 * w.exposure(cc)
            if score < bs:
                sx = sum(math.cos(a2) for a2 in walls)
                sy = sum(math.sin(a2) for a2 in walls)
                face = math.atan2(-sy, -sx) if walls else math.atan2(cy - py, cx - px)
                bs, best = score, (px, py, face, key)
        if best is None:
            return (cx, cy, math.atan2(y - cy, x - cx), key)
        return best

    # ================================================================== free-for-all doctrine
    # Learned from the 10 PM Top-16 showdown replays: bots that kept moving, stayed out of crowds,
    # did not open fire early and ended with laser + lightning placed far higher.
    FFA_RANK = {"laser": 3.0, "lightning": 2.5, "mortar": 0.3, "gun": 0.5}

    def ffa_swap(self, kind, slots, held):
        """Slot to drop for this pickup (-1 = an empty slot), or None if not worth taking."""
        if kind is None or kind in held or (FLAGS["no_mortar"] and kind == "mortar"):
            return None
        for i, wp in enumerate(slots[:2]):
            if wp is None:
                return -1
        worst = min(range(min(2, len(slots))), key=lambda i: self.FFA_RANK.get(slots[i]["kind"], 0))
        if self.FFA_RANK.get(kind, 0) > self.FFA_RANK.get(slots[worst]["kind"], 0) + 0.4:
            return worst
        return None

    def threat_points(self, now, x, y):
        pts = [(px, py, 1.0) for (t, px, py) in self.emem.values() if now - t < 12]
        pts += [(px, py, 0.5) for (t, px, py) in self.sound_pts if now - t < 6]
        if self.last_hurt_bearing is not None and now - self.last_hurt_t < 4 and (self.last_hurt_by or 0) < 100:
            b_ = float(self.last_hurt_bearing)
            pts.append((x + 90 * math.cos(b_), y + 90 * math.sin(b_), 1.0))
        return pts

    def make_risk(self, thr, now):
        """Per-metre extra route cost: open to a known threat (tactical map), outside the zone on arrival, chokepoints."""
        w, zone, spd = self.world, self._zone, max(1.0, self._ws)
        cache = {}
        nbr = w.nbr
        za = self.zone_at

        def risk(c, g):
            e = cache.get(c)
            if e is None:
                cx, cy = w.ccenter(c)
                r = 0.0
                for tx, ty, wt in thr:
                    dd = hyp(cx - tx, cy - ty)
                    if dd < 140 and not w.covered_from(c, tx, ty):
                        r += (1.5 if dd < 90 else 0.7) * wt
                if bin(nbr[c]).count("1") < 6:
                    r += 0.3
                e = cache[c] = (r, cx, cy)
            r, cx, cy = e
            zx, zy, zr = za(zone, now + g / spd)
            out = hyp(cx - zx, cy - zy) - (zr - 4.0)
            if out > 0:
                r += 3.0 + 0.1 * out
            return r
        return risk

    def pick_move(self, x, y, cands, now):
        """Choose among candidate moves (first = preferred): stay walkable, in the zone, and do not close on threats."""
        zx, zy, zr = self.zone_at(self._zone, now + 2.0)
        inz = hyp(x - zx, y - zy) < zr - 2.0
        thr = [t for t in self.threat_points(now, x, y) if hyp(t[0] - x, t[1] - y) < 70]
        w = self.world
        best, bs = cands[0], 1e18
        for i, (mx, my) in enumerate(cands):
            ux, uy = unit(mx, my)
            px, py = x + ux * 5.0, y + uy * 5.0
            sc = 0.6 * i
            if w is not None and not w.walkable(px, py):
                sc += 6.0
            if inz and hyp(px - zx, py - zy) > zr - 2.0:
                sc += 10.0
            for tx, ty, wt in thr:
                sc += 0.4 * wt * (hyp(tx - x, ty - y) - hyp(tx - px, ty - py))
            if sc < bs:
                bs, best = sc, (mx, my)
        return best

    def safe_point(self, x, y, zone, now, scx, scy, sr, urgent, force=False):
        """Low-threat walkable point 35-110 m away, inside where the zone will be."""
        c = self._safe
        if (not force and c is not None and now - c[2] < 4.0 and hyp(c[0] - x, c[1] - y) > 8):
            return c
        w = self.world
        pts = self.threat_points(now, x, y)
        zx, zy, zr = self.zone_at(zone, now + 20)
        best, bs = None, 1e18
        my_comp = w.comp_at(x, y) if w is not None else -1
        for rad in (35, 70, 110):
            for k in range(16):
                a = k * PI / 8 + (0.2 if rad == 70 else 0.0)
                px, py = x + rad * math.cos(a), y + rad * math.sin(a)
                if w is not None:
                    if not w.walkable(px, py) or (my_comp >= 0 and w.comp_at(px, py) != my_comp):
                        continue
                thr = sum(wt * 900.0 / (hyp(px - tx, py - ty) + 20.0) for (tx, ty, wt) in pts)
                out_now = max(0.0, hyp(px - zx, py - zy) - (zr - 15))
                if out_now > 0 and hyp(x - zx, y - zy) < zr - 15:
                    continue  # never trade a safe position for one the zone will cover
                out_safe = max(0.0, hyp(px - scx, py - scy) - sr)
                score = thr + 10.0 * out_now + (2.0 if urgent else 0.15) * out_safe + 0.02 * rad
                if score < bs:
                    bs, best = score, (px, py, now)
        if best is None:
            best = (scx, scy, now)
        self._safe = best
        return best

    def ffa_fight(self, x, y, me, hpf, energy, target, enemies, held, zone, now, scx, scy, sr, tleft,
                  mvx, mvy, tx_, ty_, boost):
        close = [m for m in enemies if m["_d"] < 150]
        thf = float(target.get("hp", 1400)) / max(1.0, float(target.get("max_hp", 1400)))
        td = target["_d"]
        ex, ey = unit(target["pos"][0] - x, target["pos"][1] - y)
        urgent = tleft < 15
        if len(close) >= 2 or (hpf < 0.45 and thf > hpf - 0.05):
            sp = self.safe_point(x, y, zone, now, scx, scy, sr, urgent, force=(now - self._safe[2] > 1.5) if self._safe else True)
            nx_, ny_ = self.nav_move(x, y, sp[0], sp[1], 4.0, now)
            return nx_ + tx_ * 0.3, ny_ + ty_ * 0.3, energy > 25, "ffa-disengage"
        if thf < 0.4 and td < 170 and target["_los"]:
            radial = 1.0 if td > 55 else 0.0
            return ex * radial + tx_ * 0.6, ey * radial + ty_ * 0.6, boost, "ffa-finish"
        if "laser" in held:
            pref = 165.0 if FLAGS["standoff"] else 110.0
        elif "lightning" in held and "gun" not in held:
            pref = 35.0
        else:
            pref = 85.0
        if not (target["_los"] and td < pref - 15):
            # nothing to change: keep the proven v1.0 footwork, just never walk toward a healthy enemy
            dot = mvx * ex + mvy * ey
            if dot > 0 and thf > 0.4:
                mvx, mvy = mvx - dot * ex, mvy - dot * ey
            return mvx, mvy, boost, "ffa-hold"
        mx_, my_ = -ex + tx_ * 0.85, -ey + ty_ * 0.85  # back off while firing
        if urgent or not me.get("in_zone", True):
            zx_, zy_ = self.nav_move(x, y, scx, scy, sr, now)
            mx_, my_ = mx_ * 0.4 + zx_, my_ * 0.4 + zy_
        return mx_, my_, boost, "ffa-kite"

    def ffa_roam(self, obs, me, x, y, z, hpf, energy, held, slots, act, zone, urgent, scx, scy, sr, tleft, now):
        """No enemy in sight: keep moving to low-threat ground, grab only nearby useful loot."""
        near_threat = any(hyp(px - x, py - y) < 100 for (px, py, wt) in self.threat_points(now, x, y) if wt >= 1.0)
        if "gun" in held:
            gw = slots[held["gun"]]
            if float(gw.get("ammo", 12)) < float(gw.get("stats", {}).get("mag", 12)) * 0.7 and float(gw.get("reload", 0)) <= 0:
                act["reload"] = True
        if not near_threat and not urgent:
            best, bd = None, 1e9
            for p in obs.get("pickups", []):
                if self.ffa_swap(p.get("kind"), slots, held) is None or not self.on_ground_level(p["pos"]):
                    continue
                if self.blacklist.get(p["id"], -1) > now:
                    continue
                d = hyp(p["pos"][0] - x, p["pos"][1] - y)
                if d < 90 and d < bd:
                    best, bd = p, d
            if best is None:
                for c in obs.get("cubes", []):
                    if not self.on_ground_level(c["pos"]) or self.blacklist.get(c["id"], -1) > now:
                        continue
                    d = hyp(c["pos"][0] - x, c["pos"][1] - y)
                    if d < 45 and d < bd:
                        best, bd = c, d
            if best is not None:
                if self.loot_goal is None or self.loot_goal[0] != best["id"]:
                    self.loot_goal = (best["id"], now, 0)
                elif now - self.loot_goal[1] > 15:
                    self.blacklist[best["id"]] = now + 30
                mv = self.nav_move(x, y, best["pos"][0], best["pos"][1], 0.8, now)
                return "ffa-loot", mv, None
        sp = self.safe_point(x, y, zone, now, scx, scy, sr, urgent)
        mv = self.nav_move(x, y, sp[0], sp[1], 4.0, now)
        if hyp(*mv) < 0.1:
            self._safe = None
        mode = "ffa-hurt" if (hpf < 0.5 and near_threat) else "ffa-roam"
        yaw = None
        if self.look is not None and now < self.look[1]:
            yaw = self.look[0]
        elif hyp(*mv) > 0.1:
            yaw = math.atan2(mv[1], mv[0]) + 1.0 * math.sin(now * 1.6)
        return mode, mv, (yaw, 0.0) if yaw is not None else None

    def weapon_dps(self, kinds, dist, cubes):
        """Rough sustained damage per second of a loadout at a distance (accuracy folded in)."""
        W = self.wcfg
        dps = 0.0
        g = W.get("gun", {})
        if "gun" in kinds and dist < g.get("range", 230):
            cyc = g.get("mag", 12) * g.get("cooldown", 0.24) + g.get("reload", 2.4)
            acc = max(0.25, 0.85 - dist / 400.0)
            dps += g.get("damage", 45) * g.get("mag", 12) / cyc * acc
        l = W.get("laser", {})
        if "laser" in kinds and dist < l.get("range", 280):
            dps += l.get("damage", 315) / (l.get("charge", 0.75) + l.get("cooldown", 1.65)) * 0.7
        z = W.get("lightning", {})
        if "lightning" in kinds and dist < z.get("range", 45) + 5:
            dps += z.get("damage", 60) / z.get("cooldown", 0.25) * 0.9
        m = W.get("mortar", {})
        if "mortar" in kinds and 20 < dist < 235:
            dps += (m.get("damage", 200) + m.get("splash", 160)) * 0.5 / m.get("cooldown", 2.6) * 0.4
        return dps * (1 + 0.04 * min(cubes, 10))

    def lanchester(self, me, held, enemies, hp):
        """Square-law strength ratio: (our hp*dps) / (sum of sqrt(their hp*dps))^2. >1 = we should win."""
        armor = float(me.get("stats", {}).get("armor", 0) or 0)
        my_eff_hp = hp / max(0.2, 1.0 - armor)
        near = [m for m in enemies if m["_d"] < 170]
        if not near:
            return 9.0
        dmean = sum(m["_d"] for m in near) / len(near)
        mine = my_eff_hp * self.weapon_dps(list(held), dmean, float(me.get("cubes", 0) or 0))
        theirs = 0.0
        for m in near:
            kinds = m.get("weapons") or ["gun"]
            e = float(m.get("hp", 1400)) * self.weapon_dps(kinds, m["_d"], float(m.get("cubes", 0) or 0))
            # an enemy that cannot see us right now only counts half
            theirs += math.sqrt(max(0.0, e)) * (1.0 if m["_los"] else 0.7)
        theirs = theirs ** 2
        return mine / max(1.0, theirs)

    def find_spot(self, x, y, see_target, hide_from, zone, now, key):
        """Closest walkable point within ~16 m that is hidden from every enemy in hide_from and,
        if see_target is given, has a clear shot at it."""
        cache = getattr(self, "_spot", {})
        ck = (key, see_target["id"] if see_target else None)
        hit = cache.get(ck)
        if hit is not None and now - hit[0] < 0.5 and hyp(hit[2] - x, hit[3] - y) < 4:
            return hit[1]
        w = self.world
        cx_, cy_, zr = self.zone_at(zone, now + 6)
        best = None
        for rad in (4, 7, 11, 16):
            cands = []
            for k in range(16):
                a = k * PI / 8
                px, py = x + rad * math.cos(a), y + rad * math.sin(a)
                if not w.walkable(px, py) or hyp(px - cx_, py - cy_) > zr - 6:
                    continue
                if not w.seg_walkable(x, y, px, py, 1.5):
                    continue
                gz = w.ground_at(px, py)
                if see_target is not None:
                    tp = see_target["pos"]
                    if not w.los((px, py, gz + 7), (tp[0], tp[1], tp[2] + 4.5)):
                        continue
                if any(w.los((m["pos"][0], m["pos"][1], m["pos"][2] + 8), (px, py, gz + 6)) for m in hide_from[:4]):
                    continue
                cands.append((px, py))
            if cands:
                best = cands[0]
                break
        cache[ck] = (now, best, x, y)
        self._spot = cache
        return best

    def find_cover(self, x, y, enemies, zone, now):
        """Nearby walkable point that no visible enemy can see."""
        cache = getattr(self, "_cover", None)
        if cache is not None and now - cache[0] < 0.6:
            return cache[1]
        w = self.world
        if FLAGS["tmap"] and w is not None and w.tmap is not None:
            best = self.find_cover_tmap(x, y, enemies, zone, now)
            self._cover = (now, best)
            return best
        cx_, cy_, zr = self.zone_at(zone, now + 8)
        ecx = sum(m["pos"][0] for m in enemies) / len(enemies)
        ecy = sum(m["pos"][1] for m in enemies) / len(enemies)
        away = math.atan2(y - ecy, x - ecx)
        best, bs = None, 1e18
        for rad in (10, 18, 28, 40):
            for k in range(12):
                a = away + (k - 5.5) * (PI / 7)
                px, py = x + rad * math.cos(a), y + rad * math.sin(a)
                if not w.walkable(px, py) or hyp(px - cx_, py - cy_) > zr - 6:
                    continue
                gz = w.ground_at(px, py) + 5.0
                seen = 0
                for m in enemies[:4]:
                    mp = m["pos"]
                    if w.los((mp[0], mp[1], mp[2] + 8), (px, py, gz)):
                        seen += 1
                        break
                if seen:
                    continue
                score = rad + 12 * abs(angdiff(a, away))
                if score < bs:
                    bs, best = score, (px, py)
            if best is not None:
                break
        self._cover = (now, best)
        return best

    def isolate_cell(self, x, y, target, enemies, zone, now):
        """Closest reachable cell where the target can be shot but every other enemy is behind cover."""
        cache = getattr(self, "_isoc", None)
        if cache is not None and now - cache[0] < 0.5 and cache[2] == target["id"]:
            return cache[1]
        w = self.world
        tp = target["pos"]
        others = [(m["pos"][0], m["pos"][1]) for m in enemies if m["id"] != target["id"]][:5]
        cx_, cy_, zr = self.zone_at(zone, now + 8)
        best, bs = None, 1e18
        for c, pl in w.reachable_cells(x, y, 8).items():
            px, py = w.ccenter(c)
            if hyp(px - cx_, py - cy_) > zr - 6 or w.covered_from(c, tp[0], tp[1]):
                continue
            if any(not w.covered_from(c, ox, oy) for (ox, oy) in others):
                continue
            if pl < bs:
                bs, best = pl, (px, py)
        self._isoc = (now, best, target["id"])
        return best

    def find_cover_tmap(self, x, y, enemies, zone, now):
        """Tactical-map cover: the closest reachable cell hidden from every known enemy (lookup, no raycasts)."""
        w = self.world
        threats = [(m["pos"][0], m["pos"][1]) for m in enemies[:6]]
        threats += [(px, py) for (t, px, py) in self.emem.values() if now - t < 8]
        if not threats:
            return None
        cx_, cy_, zr = self.zone_at(zone, now + 8)
        ecx = sum(t[0] for t in threats) / len(threats)
        ecy = sum(t[1] for t in threats) / len(threats)
        reach = w.reachable_cells(x, y, 13)
        best, bs = None, 1e18
        for c, pl in reach.items():
            px, py = w.ccenter(c)
            if hyp(px - cx_, py - cy_) > zr - 6:
                continue
            seen = sum(1 for (tx, ty) in threats if not w.covered_from(c, tx, ty))
            # fully hidden cells first, then fewest exposures; prefer short paths that open distance from the enemies
            score = seen * 120.0 + pl - 0.35 * hyp(px - ecx, py - ecy) + 0.5 * w.exposure(c)
            if score < bs:
                bs, best = score, (px, py)
        return best

    def shell_dodge(self, obs, x, y, z):
        st = self.wcfg.get("mortar", {})
        g = float(st.get("gravity", 22))
        sr = float(st.get("splash_radius", 7)) + 4.0
        for p in obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            tt = 0.0
            while tt < 6.0:
                tt += 0.1
                qx, qy = px + vx * tt, py + vy * tt
                qz = pz + vz * tt - 0.5 * g * tt * tt
                if qz < self.ground(qx, qy, z) + 0.3:
                    if hyp(qx - x, qy - y) < sr:
                        return unit(x - qx + 1e-3, y - qy)
                    break
        return None


def npcs_of(obs):
    return [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]


if __name__ == "__main__":
    mf.run(Bot)
