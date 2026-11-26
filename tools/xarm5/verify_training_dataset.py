#!/usr/bin/env python3
"""¿Está un dataset xArm5 listo para entrenar, y es consistente con el que entrenó un modelo que funcionó?

Compara ``--dataset`` con ``--reference`` (p. ej. el dataset de ``train_real_v7x5v2_relative_ee_e300``, el modelo
probado con éxito en el xArm5) en tres niveles:
1. **Estructura y convenciones**: arreglos por episodio (nombre, tipo, forma), atributos de la conversión
   (``xarm5_conversion``: coordenadas relativas, hemisferio del cuaternión, home, extrínseca, modo de orientación,
   filtro, versión del conversor).
2. **Física**: largo de los episodios, tramo done (al final, contiguo, >= ``--min-done-frames``), pinza, norma y
   hemisferio del cuaternión, saltos entre frames, velocidad, posiciones del TCP (al inicio, en todo y al final).
3. **Entrenamiento**: el MISMO ``ZarrDataset`` que usa ``visuomotor_core/train.py`` (ventanas, copy-padding,
   ``relative_ee``), sus estadísticas de normalización, un lote del DataLoader y, con ``--forward``, un paso de
   entrenamiento (forward + backward) de ``DiffusionPolicy`` en la GPU.
Solo lectura. Los nombres de los datasets se pasan siempre por argumento (AGENTS.md §16).

    python3 tools/xarm5/verify_training_dataset.py --dataset datasets/<nuevo>.zarr \\
        --reference datasets/<referencia>.zarr --forward --out <informe>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_core"))

CONV_KEYS = ("converter_version", "relative_coordinates", "quaternion_hemisphere", "quaternion_reference_xyzw",
             "orientation_mode", "home_xyz_m", "home_rpy_rad", "yaw_limits_rad", "tcp_to_camera_xyz_m", "tcp_to_camera_rpy_rad",
             "extrinsic_calibrated", "trajectory_filter", "start_poses")


def q(v, ps=(5, 50, 95)):
    v = np.asarray(v, dtype=np.float64)
    return [round(float(np.percentile(v, p)), 4) for p in ps] if len(v) else []


def structure(g) -> dict:
    eps = sorted(g.group_keys(), key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0)
    arrays = {}
    for k in eps[:3] + eps[-3:]:
        for name in g[k].array_keys():
            a = g[k][name]
            arrays.setdefault(name, set()).add((str(a.dtype), tuple(a.shape[1:])))
    conv = dict(g.attrs).get("xarm5_conversion", {})
    return {"episodes": len(eps), "arrays": {n: sorted(map(str, v)) for n, v in sorted(arrays.items())},
            "conversion": {k: conv.get(k) for k in CONV_KEYS}, "episode_keys": eps}


def physics(g, eps, min_done: int) -> dict:
    conv = dict(g.attrs).get("xarm5_conversion", {})
    home = np.asarray(conv.get("home_xyz_m", [0, 0, 0]), dtype=np.float64)
    rel = bool(conv.get("relative_coordinates", True))
    q_ref = np.asarray(conv.get("quaternion_reference_xyzw") or [0, 0, 0, 1], dtype=np.float64)
    rate = 20.0
    L, done_n, done_bad, grip, qnorm, hemi, step, vmax, p_all, p0, pf = [], [], 0, [], [], [], [], [], [], [], []
    for k in eps:
        e = g[k]
        s, a = np.asarray(e["state"], dtype=np.float64), np.asarray(e["action"], dtype=np.float64)
        rate = float(dict(e.attrs).get("sampling_rate_hz", rate))
        L.append(len(s) / rate)
        d = a[:, 8] > 0.5 if a.shape[1] > 8 else np.zeros(len(a), bool)
        done_n.append(int(d.sum()))
        if d.any():
            i0 = int(np.argmax(d))
            if not d[i0:].all() or d.sum() < min_done:
                done_bad += 1
        else:
            done_bad += 1
        grip.append(float(np.abs(np.r_[s[:, 7], a[:, 7]]).max()))
        qn = np.linalg.norm(s[:, 3:7], axis=1)
        qnorm.append(float(np.abs(qn - 1).max()))
        hemi.append(float((s[:, 3:7] @ q_ref).min()))
        xyz = s[:, :3] + (home if rel else 0.0)
        st = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        step.append(float(st.max()) if len(st) else 0.0)
        w = max(1, int(round(0.25 * rate)))
        vmax.append(float((np.linalg.norm(xyz[w:] - xyz[:-w], axis=1) * rate / w).max()) if len(xyz) > w else 0.0)
        p_all.append(xyz)
        p0.append(xyz[0])
        pf.append(xyz[-1])
    P, P0, PF = np.concatenate(p_all), np.array(p0), np.array(pf)
    return {"episodes": len(eps), "frames": int(len(P)), "duration_s_p5_50_95": q(L),
            "done_frames_min_p50": [int(min(done_n)), int(np.median(done_n))],
            "episodes_with_bad_done": done_bad,
            "gripper_max_abs": round(max(grip), 6), "quat_norm_max_dev": round(max(qnorm), 6),
            "hemisphere_min_dot": round(min(hemi), 4), "max_step_mm_p50_95_max": [round(1000 * x, 1) for x in q(step, (50, 95, 100))],
            "max_speed_m_s_p50_95": q(vmax, (50, 95)),
            "tcp_xyz_min": np.round(P.min(0), 3).tolist(), "tcp_xyz_max": np.round(P.max(0), 3).tolist(),
            "start_offset_from_home_mm_p50_95": [round(1000 * x, 1) for x in q(np.linalg.norm(P0 - home, axis=1), (50, 95))],
            "final_tcp_min": np.round(PF.min(0), 3).tolist(), "final_tcp_max": np.round(PF.max(0), 3).tolist()}


def training_check(root: str, eps, forward: bool, batch_size: int) -> dict:
    import torch
    import zarr
    from visuomotor_core.dataset.zarr_dataset import ZarrDataset
    from visuomotor_core.models.config import DiffusionConfig
    conv = dict(zarr.open_group(root, mode="r").attrs).get("xarm5_conversion", {})
    q_mode = conv.get("quaternion_hemisphere") or "qw_positive"
    q_ref = conv.get("quaternion_reference_xyzw") if q_mode == "home" else None
    cfg = DiffusionConfig(action_representation="relative_ee", quaternion_hemisphere=q_mode,
                          quaternion_reference=(list(map(float, q_ref)) if q_ref else None))
    ds = ZarrDataset(root=root, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon, episodes=eps,
                     action_representation="relative_ee", color_jitter=None)
    stats = ds.stats
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    batch = next(iter(dl))
    from visuomotor_core.training.evaluation import estimate_action_offset
    out = {"windows": len(ds.frames), "constant_state_dims": getattr(ds, "constant_state_dims", []),
           "action_offset": int(estimate_action_offset(root, eps, max_episodes=len(eps))),
           "batch_shapes": {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")},
           "stats": {k: {m: np.round(np.asarray(stats[k][m]).reshape(-1)[:9], 4).tolist() for m in ("mean", "std", "min", "max")
                         if m in stats[k]} for k in ("observation.state", "action")}}
    if forward:
        from visuomotor_core.models.diffusion_policy import DiffusionPolicy
        dev = torch.device("cuda")
        cfg.input_shapes["observation.state"] = [stats["observation.state"]["mean"].shape[0]]
        cfg.output_shapes["action"] = [stats["action"]["mean"].shape[0]]
        policy = DiffusionPolicy(cfg, dataset_stats=stats).to(dev)
        policy.train()
        b = {k: v.to(dev) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = policy.forward(b)["loss"]
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1e9))
        out["forward_backward"] = {"loss": round(float(loss), 4), "grad_norm": round(gn, 3),
                                   "finite": bool(np.isfinite(float(loss)) and np.isfinite(gn)),
                                   "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    return out


def main(argv=None) -> int:
    import zarr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--min-done-frames", type=int, default=40)
    ap.add_argument("--forward", action="store_true", help="un paso de entrenamiento en la GPU")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = {}
    for tag, path in (("dataset", a.dataset), ("reference", a.reference)):
        root = str(REPO / path) if not Path(path).is_absolute() else path
        g = zarr.open_group(root, mode="r")
        st = structure(g)
        res[tag] = {"path": path, "structure": {k: v for k, v in st.items() if k != "episode_keys"},
                    "physics": physics(g, st["episode_keys"], a.min_done_frames),
                    "training": training_check(root, st["episode_keys"], a.forward and tag == "dataset", a.batch_size)}
    d, r = res["dataset"], res["reference"]
    issues = []
    missing = sorted(set(r["structure"]["arrays"]) - set(d["structure"]["arrays"]))
    if missing:                                     # los arreglos EXTRA (sellos, inliers...) el cargador los ignora
        issues.append(f"faltan arreglos de la referencia: {missing}")
    res["extra_arrays"] = sorted(set(d["structure"]["arrays"]) - set(r["structure"]["arrays"]))
    for name in ("state", "action", "image"):
        if d["structure"]["arrays"].get(name) != r["structure"]["arrays"].get(name):
            issues.append(f"{name}: {d['structure']['arrays'].get(name)} vs {r['structure']['arrays'].get(name)}")
    for k in ("relative_coordinates", "quaternion_hemisphere", "orientation_mode", "home_xyz_m", "home_rpy_rad"):
        if d["structure"]["conversion"].get(k) != r["structure"]["conversion"].get(k):
            issues.append(f"convención {k}: {d['structure']['conversion'].get(k)} vs {r['structure']['conversion'].get(k)}")
    if d["training"]["action_offset"] != r["training"]["action_offset"]:
        issues.append(f"desfase acción/estado {d['training']['action_offset']} vs {r['training']['action_offset']}")
    if d["training"]["batch_shapes"] != r["training"]["batch_shapes"]:
        issues.append(f"lote: {d['training']['batch_shapes']} vs {r['training']['batch_shapes']}")
    p = d["physics"]
    if p["episodes_with_bad_done"]:
        issues.append(f"{p['episodes_with_bad_done']} episodios con done ausente, corto o no contiguo")
    if p["gripper_max_abs"] > 1e-6:
        issues.append(f"pinza distinta de 0 (máx {p['gripper_max_abs']})")
    if p["quat_norm_max_dev"] > 1e-3 or p["hemisphere_min_dot"] < 0:
        issues.append(f"cuaternión: norma ±{p['quat_norm_max_dev']}, hemisferio mín {p['hemisphere_min_dot']}")
    if "forward_backward" in d["training"] and not d["training"]["forward_backward"]["finite"]:
        issues.append("paso de entrenamiento con pérdida o gradiente no finito")
    # Demos marcadas por el conversor (no son errores de formato: quedan en el dataset, ver el informe de conversión)
    rp = Path(str(REPO / a.dataset) if not Path(a.dataset).is_absolute() else a.dataset)
    rp = rp.with_name(rp.stem + "_conversion_report.json")
    if rp.is_file():
        eps_rep = json.loads(rp.read_text()).get("episodes", {})
        by = {}
        for k, e in eps_rep.items():
            for fl in e.get("flags") or []:
                key = fl.split(" > ")[0].split(":")[0].strip()
                by.setdefault(key, []).append(k)
        res["conversion_flags"] = {k: len(v) for k, v in by.items()}
        res["conversion_flagged_episodes"] = sorted({k for v in by.values() for k in v})
        print(f"\nMarcadas por el conversor (se conservan): {len(res['conversion_flagged_episodes'])} de {len(eps_rep)} | "
              + "; ".join(f"{k}: {n}" for k, n in res["conversion_flags"].items()))
    if res.get("extra_arrays"):
        print(f"Arreglos extra (el cargador de entrenamiento los ignora): {res['extra_arrays']}")
    res["issues"] = issues
    res["verdict"] = "LISTO para entrenar (consistente con la referencia)" if not issues else "REVISAR"
    w = 30
    print(f"{'':{w}s} {'DATASET':>34s} {'REFERENCIA':>34s}")
    for sec, keys in (("structure", ("episodes",)), ("physics", ("frames", "duration_s_p5_50_95", "done_frames_min_p50",
                      "episodes_with_bad_done", "gripper_max_abs", "quat_norm_max_dev", "hemisphere_min_dot",
                      "max_step_mm_p50_95_max", "max_speed_m_s_p50_95", "tcp_xyz_min", "tcp_xyz_max",
                      "start_offset_from_home_mm_p50_95", "final_tcp_min", "final_tcp_max")),
                      ("training", ("windows", "action_offset", "batch_shapes"))):
        for k in keys:
            print(f"{k:{w}s} {str(d[sec][k]):>34s} {str(r[sec][k]):>34s}")
    for k in ("converter_version", "quaternion_hemisphere", "orientation_mode", "tcp_to_camera_xyz_m", "start_poses"):
        print(f"{k:{w}s} {str(d['structure']['conversion'].get(k))[:34]:>34s} {str(r['structure']['conversion'].get(k))[:34]:>34s}")
    if "forward_backward" in d["training"]:
        print(f"{'paso de entrenamiento (GPU)':{w}s} {str(d['training']['forward_backward'])}")
    print("\nPROBLEMAS:" if issues else "\nSin problemas de estructura, convenciones, física ni entrenamiento.")
    for i in issues:
        print(" -", i)
    print("VEREDICTO:", res["verdict"])
    if a.out:
        out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        print("->", out)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
