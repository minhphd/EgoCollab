"""A walking person for BEHAVIOR scenes: an Isaac Sim character skinned in numpy every frame.

Isaac Sim's skeletal animation aborts the app under OmniGibson's Fabric scene delegate, so no skeleton reaches the
renderer. At load the biped set's walk and idle clips are retargeted onto the character (omni.anim.retarget) and read
into arrays with its skin weights; the skeletal prims then leave the stage, and every frame plain copies of the
character's meshes get skinned points written to Fabric. The caller moves the body; each clip plays in place, looped
over its steadiest stretch, so the walking speed to move the body at is the clip's (Human.speed).

    patch_app()                       # before omnigibson launches: loads the retargeting extension with the app
    h = Human(og.sim, "M_Medical_01")
    head = h.update(xy, yaw, moving=True)
"""

from __future__ import annotations

import numpy as np

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters"
WALKS = [f"stand_walk_{i}_skelanim" for i in (1, 2, 3, 4, 5, 7)]
IDLE = "stand_idle_loop_skelanim"
FPS = 30  # the clips' sample rate


def patch_app() -> None:
    """Make Isaac Sim's app load omni.anim.retarget.core at launch (OmniGibson passes no extra arguments)."""
    import isaacsim
    base = isaacsim.SimulationApp
    if getattr(base, "_with_retarget", False):
        return

    class App(base):
        _with_retarget = True

        def __init__(self, config=None, experience=""):
            config = dict(config or {})
            config["extra_args"] = list(config.get("extra_args", [])) + ["--enable", "omni.anim.retarget.core"]
            super().__init__(config, experience=experience)

    isaacsim.SimulationApp = App


def qangle(a, b) -> float:
    """Angle between two Gf quaternions, rad."""
    d = abs(a.GetReal() * b.GetReal() + float(np.dot(np.array(a.GetImaginary()), np.array(b.GetImaginary()))))
    return 2 * float(np.arccos(min(1.0, d)))


class Clip:
    """A retargeted clip on the character's joints. A walk is looped over its steadiest stretch of 1 to 3 s (steady
    speed and heading, ends that match) with the root's travel removed; speed and heading are that stretch's."""

    def __init__(self, anim, joints: list, rest: tuple, walk: bool):
        from pxr import Gf
        cj = list(anim.GetJointsAttr().Get())
        times = list(anim.GetRotationsAttr().GetTimeSamples())
        idx = [cj.index(j) if j in cj else -1 for j in joints]
        r = [anim.GetRotationsAttr().Get(t) for t in times]
        tr = anim.GetTranslationsAttr()
        t = [tr.Get(x) for x in tr.GetTimeSamples()] or [tr.Get()] * len(times)
        a, b, self.speed, self.heading, self.gap = 0, len(times), 0.0, 0.0, 0.0
        if walk:
            root = np.array([np.array(x[0])[:2] for x in t])
            v = np.linalg.norm(np.diff(root, axis=0), axis=1) * FPS
            head = np.unwrap(np.arctan2(*np.diff(root, axis=0).T[::-1]))
            med, best = float(np.median(v[v > 0.3])), None
            for a0 in range(0, len(v) - 95, 3):
                for n in range(30, 91, 2):
                    if v[a0:a0 + n].min() < 0.8 * med or np.ptp(head[a0:a0 + n]) > np.radians(12):
                        continue
                    err = sum(qangle(r[a0][i], r[a0 + n][i]) for i in range(1, len(cj)))
                    best = min(best, (err, a0, n)) if best else (err, a0, n)
            self.gap, a, n = best
            b = a + n
            self.speed, self.heading = float(np.mean(v[a:b])), float(np.mean(head[a:b]))
        rest_t, rest_r = rest
        self.frames = []  # (rotations, translations) per frame, on the character's joints
        for k in range(a, b):
            rr = [r[k][i] if i >= 0 else rest_r[j] for j, i in enumerate(idx)]
            tt = [t[k][i] if i >= 0 else rest_t[j] for j, i in enumerate(idx)]
            tt[0] = Gf.Vec3f(rest_t[0][0], rest_t[0][1], tt[0][2])  # in place at the rest root: the caller moves the body
            self.frames.append((rr, tt))

    def face(self, yaws: list) -> None:
        """Turn every frame by minus the clip's mean facing (yaws: each frame's facing in skeleton space), so all
        clips face skeleton +x and switching clips does not turn the body."""
        from pxr import Gf
        f = float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))
        turn = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(np.degrees(-f)))
        for rr, _ in self.frames:
            rr[0] = Gf.Quatf((Gf.Rotation(Gf.Quatd(rr[0])) * turn).GetQuat())

    @property
    def seconds(self) -> float:
        return len(self.frames) / FPS


class Human:
    def __init__(self, og_sim, character: str = "M_Medical_01", walk_speed: float = 1.1, prim: str = "/World/Characters"):
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdShade, UsdSkel, Vt
        self.og_sim, self.Vt, self.Gf, self.UsdSkel = og_sim, Vt, Gf, UsdSkel
        stage = omni.usd.get_context().get_stage()
        with og_sim.editing_usd():
            biped = stage.DefinePrim(f"{prim}/Biped_Setup", "Xform")
            biped.GetReferences().AddReference(f"{ROOT}/Biped_Setup.usd")
            UsdGeom.Imageable(biped).MakeInvisible()
            body = stage.DefinePrim(f"{prim}/body", "Xform")
            body.GetReferences().AddReference(f"{ROOT}/{character}/{character}.usd")
        skel_prim = next(p for p in Usd.PrimRange(body) if p.IsA(UsdSkel.Skeleton))
        skelroot = next(p for p in Usd.PrimRange(body) if p.IsA(UsdSkel.Root))
        skel = UsdSkel.Skeleton(skel_prim)
        self.joints = joints = list(skel.GetJointsAttr().Get())
        src = {p.GetName(): p for p in Usd.PrimRange(biped) if p.IsA(UsdSkel.Animation)}
        src_skel = next(p for p in Usd.PrimRange(biped) if p.IsA(UsdSkel.Skeleton))
        with og_sim.editing_usd():
            stage.DefinePrim(f"{prim}/retargeted", "Scope")
            omni.kit.commands.execute("CreateRetargetAnimationsCommand", source_skeleton_path=str(src_skel.GetPath()),
                                      target_skeleton_path=str(skel_prim.GetPath()),
                                      source_animation_paths=[str(src[n].GetPath()) for n in WALKS + [IDLE] if n in src],
                                      target_animation_parent_path=f"{prim}/retargeted", set_root_identity=False)
        made = {p.GetName(): UsdSkel.Animation(p) for p in Usd.PrimRange(stage.GetPrimAtPath(f"{prim}/retargeted"))
                if p.IsA(UsdSkel.Animation)}
        rest = skel.GetRestTransformsAttr().Get()
        rest = ([Gf.Vec3f(m.ExtractTranslation()) for m in rest], [Gf.Quatf(m.ExtractRotationQuat()) for m in rest])
        walks = [Clip(made[n], joints, rest, True) for n in WALKS if n in made]
        self.walk = min(walks, key=lambda c: abs(c.speed - walk_speed) + 0.05 * c.gap)  # nearest pace, then the cleanest loop
        self.idle = Clip(made[IDLE], joints, rest, False)
        self.speed = self.walk.speed
        cache = UsdSkel.Cache()
        cache.Populate(UsdSkel.Root(skelroot), Usd.PrimDefaultPredicate)
        binding = next(b for b in cache.ComputeSkelBindings(UsdSkel.Root(skelroot), Usd.PrimDefaultPredicate)
                       if b.GetSkeleton().GetPrim() == skel_prim)
        self.parents = np.array(cache.GetSkelQuery(skel).GetTopology().GetParentIndices())
        hip = lambda fr: self._skeleton(fr)[1][:3, :3]  # the hips' orientation in skeleton space (row vectors)
        go = np.array([np.cos(self.walk.heading), np.sin(self.walk.heading), 0.0])
        fwd = np.mean([go @ np.linalg.inv(hip(fr)) for fr in self.walk.frames], axis=0)  # the hips' forward axis
        for clip in (self.walk, self.idle):
            clip.face([float(np.arctan2(*(fwd @ hip(fr))[1::-1])) for fr in clip.frames])
        self.bind_inv = np.array([np.array(m.GetInverse()) for m in skel.GetBindTransformsAttr().Get()])
        self.skel_world = np.array(UsdGeom.Xformable(skel_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        self.meshes = []
        with og_sim.editing_usd():
            for i, sq in enumerate(binding.GetSkinningTargets()):
                self.meshes.append(self._copy(stage, sq, f"{prim}/skin/m{i:02d}", joints))
            for sq in binding.GetSkinningTargets():
                sq.GetPrim().SetActive(False)
            skel_prim.SetActive(False)
            stage.RemovePrim(f"{prim}/retargeted")
            stage.RemovePrim(f"{prim}/Biped_Setup")
        og_sim.update_handles()  # the stage edits above invalidate PhysX's views of the scene's objects
        last = [j.split("/")[-1].lower() for j in joints]
        find = lambda *names: next((i for i, n in enumerate(last) if n in names), None)
        self.head_i = find("head")
        self.hand_i = find("r_hand", "righthand", "hand_r")
        self.arm = (find("r_upperarm", "rightarm", "upperarm_r"), find("r_forearm", "rightforearm", "lowerarm_r"), self.hand_i)
        self.spine_i, self.neck_i = find("spine01", "spine", "spine_01"), find("necktwist01", "neck", "neck_01")
        self.eyes = [i for i in (find("r_eye", "eye_r", "righteye"), find("l_eye", "eye_l", "lefteye")) if i is not None]
        print("arm", [joints[i].split("/")[-1] if i is not None else None for i in self.arm],
              "spine", self.spine_i is not None and last[self.spine_i], "neck", self.neck_i is not None and last[self.neck_i], flush=True)
        kids = {i: [] for i in range(len(joints))}
        for i, pj in enumerate(self.parents):
            if pj >= 0:
                kids[int(pj)].append(i)
        def subtree(i):
            out, todo = [], [i]
            while todo:
                k = todo.pop()
                out.append(k)
                todo += kids[k]
            return np.array(out)
        self.subtree = {i: subtree(i) for i in (*self.arm, self.spine_i, self.neck_i) if i is not None}
        self.reset()
        x = np.array([1.0, 0.0, 0.0]) @ self.skel_world[:3, :3]
        self.yaw0 = float(np.arctan2(x[1], x[0]))  # the world heading of skeleton +x, where every clip faces

    def reset(self) -> None:
        """Standing, at the start of the idle clip: an episode begins the same whatever the last one did."""
        self.phase, self.moving, self.sk, self.world, self._frame = 0.0, False, None, None, None
        self.t, self.last_move, self.blend = 0.0, -9.0, None  # blend: (pose blended from, seconds into the blend)

    def _copy(self, stage, sq, path: str, joints: list) -> tuple:
        """A plain copy of a skinned mesh (topology, texture coordinates, materials) and its skin as arrays."""
        from pxr import Gf, UsdGeom, UsdShade, UsdSkel, Vt
        src = UsdGeom.Mesh(sq.GetPrim())
        dst = UsdGeom.Mesh.Define(stage, path)
        dst.CreateFaceVertexCountsAttr(src.GetFaceVertexCountsAttr().Get())
        dst.CreateFaceVertexIndicesAttr(src.GetFaceVertexIndicesAttr().Get())
        dst.CreateSubdivisionSchemeAttr("none")
        for pv in UsdGeom.PrimvarsAPI(src).GetPrimvars():
            if pv.GetPrimvarName().startswith(("skel:", "display")) or pv.GetPrimvarName() == "normals":
                continue
            try:  # one that will not copy is left out
                npv = UsdGeom.PrimvarsAPI(dst).CreatePrimvar(pv.GetPrimvarName(), pv.GetTypeName(), pv.GetInterpolation())
                npv.Set(pv.Get())
                if pv.IsIndexed():
                    npv.SetIndices(pv.GetIndices())
            except Exception:
                pass
        mat = UsdShade.MaterialBindingAPI(src.GetPrim()).ComputeBoundMaterial()[0]
        if mat:
            UsdShade.MaterialBindingAPI.Apply(dst.GetPrim()).Bind(mat)
        for sub in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(src.GetPrim())):
            ns = UsdGeom.Subset.Define(stage, dst.GetPath().AppendChild(sub.GetPrim().GetName()))
            ns.CreateIndicesAttr(sub.GetIndicesAttr().Get())
            ns.CreateElementTypeAttr(sub.GetElementTypeAttr().Get())
            ns.CreateFamilyNameAttr(sub.GetFamilyNameAttr().Get())
            m2 = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
            if m2:
                UsdShade.MaterialBindingAPI.Apply(ns.GetPrim()).Bind(m2)
        pts = np.asarray(src.GetPointsAttr().Get(), float)
        dst.CreatePointsAttr(src.GetPointsAttr().Get())
        dst.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-100, -100, -100), Gf.Vec3f(100, 100, 100)]))
        api = UsdSkel.BindingAPI(src.GetPrim())
        k = api.GetJointIndicesPrimvar().GetElementSize()
        ji = np.asarray(api.GetJointIndicesPrimvar().Get(), int).reshape(-1, k)
        jw = np.asarray(api.GetJointWeightsPrimvar().Get(), float).reshape(-1, k)
        if len(ji) == 1:  # one influence set for the whole mesh
            ji, jw = np.repeat(ji, len(pts), 0), np.repeat(jw, len(pts), 0)
        own = api.GetJointsAttr().Get()
        if own:  # the mesh's joint order onto the skeleton's
            ji = np.array([joints.index(j) for j in own])[ji]
        gb = api.GetGeomBindTransformAttr().Get()
        bind_pts = np.c_[pts, np.ones(len(pts))] @ (np.array(gb) if gb is not None else np.eye(4))
        return bind_pts, ji, jw / np.maximum(jw.sum(1, keepdims=True), 1e-9), path

    def _skeleton(self, frame) -> np.ndarray:
        """Skeleton-space joint matrices (row vectors) of one clip frame."""
        Vt, Gf = self.Vt, self.Gf
        rr, tt = frame
        local = np.array([np.array(m) for m in self.UsdSkel.MakeTransforms(
            Vt.Vec3fArray(tt), Vt.QuatfArray(rr), Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(self.joints)))])
        out = np.empty_like(local)
        for j, pj in enumerate(self.parents):
            out[j] = local[j] if pj < 0 else local[j] @ out[pj]
        return out

    def update(self, xy, yaw: float, moving: bool, dt: float = 1 / 15, reach=None, look=None) -> np.ndarray:
        """Pose the body at xy facing yaw, walking or standing, one dt on; returns the head joint's world position.
        Walking lasts until 0.3 s after the last move (turns in place do not flicker), and clips blend over 0.25 s."""
        import usdrt
        Gf = self.Gf
        self.t += dt
        if moving:
            self.last_move = self.t
        walking = self.t - self.last_move < 0.3
        if walking != self.moving:
            self.blend = (self._frame, 0.0) if getattr(self, "_frame", None) else None
            self.phase, self.moving = 0.0, walking
        clip = self.walk if walking else self.idle
        self.phase += dt / clip.seconds
        rr, tt = clip.frames[int(self.phase * len(clip.frames)) % len(clip.frames)]
        if self.blend is not None:
            (r0, t0), b = self.blend
            w = min(1.0, b / 0.25)
            rr = [Gf.Slerp(w, qa, qb) for qa, qb in zip(r0, rr)]
            tt = [Gf.Vec3f(*[float(x) for x in np.array(a) * (1 - w) + np.array(c) * w]) for a, c in zip(t0, tt)]
            self.blend = None if w >= 1.0 else ((r0, t0), b + dt)
        self._frame = (rr, tt)
        self.sk = self._skeleton(self._frame)
        c, s = np.cos(yaw - self.yaw0), np.sin(yaw - self.yaw0)
        self.world = self.skel_world @ np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [xy[0], xy[1], 0, 1]])
        if look is not None:
            self._look(np.asarray(look, float))
        if reach is not None and reach[1] > 0:
            self._reach(np.asarray(reach[0], float), float(reach[1]))
        skin = self.bind_inv @ self.sk
        rt = self.og_sim.usdrt_stage
        for bp, ji, jw, path in self.meshes:
            p = np.einsum("nk,nkd->nd", jw, np.einsum("nc,nkcd->nkd", bp, skin[ji]))
            rt.GetPrimAtPath(path).GetAttribute("points").Set(usdrt.Vt.Vec3fArray((p @ self.world)[:, :3].astype(np.float32)))
        return self.joint(self.head_i)

    # -- pose edits in skeleton space (row vectors: a child's matrix is its local one times its parent's) ---------
    def _to_skel(self, p) -> np.ndarray:
        return (np.array([*p, 1.0]) @ np.linalg.inv(self.world))[:3]

    def _turn(self, i: int, pivot: np.ndarray, R: np.ndarray) -> None:
        """Rotate joint i and everything below it by R (column-vector rotation) about pivot."""
        G = np.eye(4)
        G[:3, :3] = R.T
        G[3, :3] = pivot - pivot @ R.T
        self.sk[self.subtree[i]] = self.sk[self.subtree[i]] @ G

    @staticmethod
    def _aim(a, b, most: float = np.pi) -> np.ndarray:
        """The rotation taking direction a to direction b, by at most `most` rad."""
        a, b = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
        axis, cosang = np.cross(a, b), float(np.clip(a @ b, -1, 1))
        n = np.linalg.norm(axis)
        if n < 1e-8:
            return np.eye(3)
        k, ang = axis / n, min(np.arccos(cosang), most)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

    def _pos(self, i: int) -> np.ndarray:
        return self.sk[i][3, :3].copy()

    def _look(self, target) -> None:
        """Turn the neck (and head) toward target, by up to 60 degrees, most of the way."""
        if self.neck_i is None:
            return
        n, h = self._pos(self.neck_i), self._pos(self.head_i)
        want = self._to_skel(target) - h
        R = self._aim(np.array([1.0, 0.0, -0.15]), want, most=np.radians(60))  # the head faces skeleton +x
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        self._turn(self.neck_i, n, self._aim(np.array([1.0, 0.0, -0.15]), want, most=0.7 * ang) if ang > 0 else np.eye(3))

    def _reach(self, target, w: float) -> None:
        """Bend at the spine for low targets and bring the right hand a share w of the way to target."""
        up, fo, ha = self.arm
        if None in (up, fo, ha):
            return
        T = self._to_skel(target)
        S = self._pos(up)
        if self.spine_i is not None:  # bend forward (about skeleton y) for what lies below the shoulder
            bend = w * np.radians(30) * float(np.clip((S[2] - T[2] - 0.6) / 0.5, 0.0, 1.0))  # more puts the head into wall cabinets
            if bend > 0:
                c, s_ = np.cos(bend), np.sin(bend)
                self._turn(self.spine_i, self._pos(self.spine_i), np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]))
        S, E, W = self._pos(up), self._pos(fo), self._pos(ha)
        T = W + w * (T - W)  # part of the way
        a, b = np.linalg.norm(E - S), np.linalg.norm(W - E)
        d = float(np.clip(np.linalg.norm(T - S), abs(a - b) + 1e-3, a + b - 1e-3))
        u = (T - S) / (np.linalg.norm(T - S) + 1e-9)
        pole = (E - S) - ((E - S) @ u) * u  # keep the elbow on its side
        pole = pole / np.linalg.norm(pole) if np.linalg.norm(pole) > 1e-4 else np.array([0.0, 0.0, -1.0])
        ca = (a * a + d * d - b * b) / (2 * a * d)
        E2 = S + a * (ca * u + np.sqrt(max(0.0, 1 - ca * ca)) * pole)
        self._turn(up, S, self._aim(E - S, E2 - S))
        W1 = self._pos(ha)
        self._turn(fo, E2, self._aim(W1 - E2, S + d * u - E2))

    def joint(self, i: int) -> np.ndarray:
        """World position of joint i in the last pose."""
        return (np.array([*self.sk[i][3, :3], 1.0]) @ self.world)[:3]

    def eye(self) -> np.ndarray:
        """Where the eye camera goes: just in front of the eyes (5 cm out from the head joint's line through them), so
        it turns and bends with the head and never sits inside it."""
        h = self.joint(self.head_i)
        if not self.eyes:
            return h + np.array([0.0, 0.0, 0.08])
        m = np.mean([self.joint(i) for i in self.eyes], axis=0)
        out = (m - h) / (np.linalg.norm(m - h) + 1e-9)
        return m + 0.05 * out

    def hand(self) -> np.ndarray:
        return self.joint(self.hand_i)
