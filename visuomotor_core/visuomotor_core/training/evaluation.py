"""Evaluación offline de una Diffusion Policy por muestreo completo (DDIM) sobre ventanas de episodios.

Mide lo que el robot ejecutaría, no la pérdida de difusión (la pérdida de validación NO predice el éxito de la
política: Mandlekar et al. 2021, §4.5; Chi et al. 2023). Métricas en la representación absoluta del estado
(``base - home``), comparando el horizonte ejecutable predicho (pasos 1..H-1) con la demostración:

* ``ade_mm`` / ``fde_mm``: error medio / final de posición del horizonte completo.
* ``ade_exec_mm``: error medio sobre los ``n_action_steps`` primeros pasos (lo que realmente se ejecuta).
* ``rot_deg``: error geodésico medio de orientación.
* ``done_*``: exactitud, precisión, exhaustividad y F1 del flag de término en el primer paso ejecutado.
* ``hold_ade_mm`` / ``hold_ade_exec_mm`` / ``hold_fde_mm``: línea base trivial "quedarse quieto" (predicción = pose
  actual). Un modelo útil debe batirla con holgura, pero es una vara BAJA.
* ``cv_ade_mm`` / ``cv_ade_exec_mm`` / ``cv_fde_mm``: línea base cinemática "velocidad constante" (extrapola la
  velocidad de los dos últimos estados observados; no usa la imagen). Es la vara EXIGENTE: con trayectorias suaves
  predice muy bien el corto plazo, aunque no es una política (no arranca desde el reposo ni sabe dónde parar). En
  ``train_real_v7x5v2_relative_ee_e300`` (época 200, val) el modelo la batía al final del horizonte (FDE 18.6 vs.
  21.3 mm) pero no en los primeros pasos (3.6 vs. 1.3 mm en 3 pasos): ver ``tools/diagnose_policy_baselines.py``.
* ``progress_ratio``: desplazamiento predicho / demostrado al final del horizonte (1 = avance correcto; << 1 indica
  política "perezosa" que se detendría en el robot).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Subset


def estimate_action_offset(root, episodes: Sequence[str], max_episodes: int = 8) -> int:
    """Desfase temporal entre la acción y el estado del dataset (posición), para alinear la línea base cinemática.

    Devuelve 0 si ``action[t] ~ state[t]`` (datasets reales VIO: la acción es la pose registrada en t) o 1 si
    ``action[t] ~ state[t+1]`` (la acción es el objetivo del paso siguiente, p. ej. el experto PID simulado).
    """
    group = zarr.open_group(root, mode="r") if isinstance(root, str) else root
    err_same = err_next = 0.0
    for name in list(episodes)[:max(1, int(max_episodes))]:
        s = np.asarray(group[name]["state"][:, :3], dtype=np.float64)
        a = np.asarray(group[name]["action"][:, :3], dtype=np.float64)
        if len(s) < 2:
            continue
        err_same += float(np.linalg.norm(a[:-1] - s[:-1], axis=1).mean())
        err_next += float(np.linalg.norm(a[:-1] - s[1:], axis=1).mean())
    return 0 if err_same <= err_next else 1


def constant_velocity_prediction(states: torch.Tensor, horizon: int, action_offset: int = 0) -> torch.Tensor:
    """Línea base "velocidad constante" para los pasos ejecutables ``t+1 .. t+horizon`` de la acción.

    ``states [B, T_obs, >=3]`` (los dos últimos para la velocidad). El paso ``j`` (0-based, acción ``t+1+j``)
    corresponde a la pose ``t+1+j+action_offset`` (ver :func:`estimate_action_offset`).
    """
    cur = states[:, -1, :3].float()
    if states.shape[1] < 2:
        return cur[:, None, :].expand(-1, horizon, -1).clone()
    vel = cur - states[:, -2, :3].float()
    k = torch.arange(horizon, device=states.device, dtype=torch.float32)[None, :, None] + 1.0 + float(action_offset)
    return cur[:, None, :] + k * vel[:, None, :]


def _geodesic_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def evaluate_sampling(policy, dataset, indices: Sequence[int], device: torch.device, n_action_steps: int,
                      batch_size: int = 128, num_workers: int = 2, seed: int = 0,
                      done_threshold: float = 0.5, action_offset: int = 0) -> Dict[str, float]:
    """``dataset`` debe entregar acciones ABSOLUTAS, sin aumentos ni dropout propioceptivo.

    ``action_offset``: desfase acción/estado del dataset (:func:`estimate_action_offset`) para la línea base cinemática.
    """
    if len(indices) == 0:
        return {}
    was_training = policy.training
    policy.eval()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    sums = {"ade": 0.0, "fde": 0.0, "ade_exec": 0.0, "rot": 0.0, "hold_ade": 0.0, "hold_ade_exec": 0.0,
            "hold_fde": 0.0, "cv_ade": 0.0, "cv_ade_exec": 0.0, "cv_fde": 0.0, "prog_num": 0.0, "prog_den": 0.0}
    has_velocity = True
    tp = fp = fn = tn = 0
    n = 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            obs = {"observation.image": batch["observation.image"].to(device, non_blocking=True),
                   "observation.state": batch["observation.state"].to(device, non_blocking=True)}
            gt = batch["action"].to(device)[:, 1:]                       # pasos ejecutables t+1 .. t+H-1
            pred = policy.predict_chunk(obs).float()                      # [B, H-1, A]
            h = min(pred.shape[1], gt.shape[1])
            pred, gt = pred[:, :h], gt[:, :h]
            pos_err = (pred[..., :3] - gt[..., :3]).norm(dim=-1) * 1000.0  # mm
            cur = obs["observation.state"][:, -1, :3].float()
            hold_err = (cur[:, None, :] - gt[..., :3]).norm(dim=-1) * 1000.0
            cv = constant_velocity_prediction(obs["observation.state"], h, action_offset)
            cv_err = (cv - gt[..., :3]).norm(dim=-1) * 1000.0
            has_velocity = has_velocity and obs["observation.state"].shape[1] >= 2
            k = max(1, min(int(n_action_steps), h))
            sums["ade"] += pos_err.mean(1).sum().item()
            sums["fde"] += pos_err[:, -1].sum().item()
            sums["ade_exec"] += pos_err[:, :k].mean(1).sum().item()
            sums["hold_ade"] += hold_err.mean(1).sum().item()
            sums["hold_ade_exec"] += hold_err[:, :k].mean(1).sum().item()
            sums["hold_fde"] += hold_err[:, -1].sum().item()
            sums["cv_ade"] += cv_err.mean(1).sum().item()
            sums["cv_ade_exec"] += cv_err[:, :k].mean(1).sum().item()
            sums["cv_fde"] += cv_err[:, -1].sum().item()
            sums["prog_num"] += (pred[:, -1, :3] - cur).norm(dim=-1).sum().item()
            sums["prog_den"] += (gt[:, -1, :3] - cur).norm(dim=-1).sum().item()
            if pred.shape[-1] >= 7:
                sums["rot"] += _geodesic_deg(pred[..., 3:7], gt[..., 3:7]).mean(1).sum().item()
            if pred.shape[-1] >= 9:
                p_done = pred[:, 0, 8] >= done_threshold
                g_done = gt[:, 0, 8] >= 0.5
                tp += int((p_done & g_done).sum()); fp += int((p_done & ~g_done).sum())
                fn += int((~p_done & g_done).sum()); tn += int((~p_done & ~g_done).sum())
            n += pred.shape[0]
    if was_training:
        policy.train()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "n_windows": n,
        "ade_mm": sums["ade"] / n,
        "fde_mm": sums["fde"] / n,
        "ade_exec_mm": sums["ade_exec"] / n,
        "rot_deg": sums["rot"] / n,
        "hold_ade_mm": sums["hold_ade"] / n,
        "hold_ade_exec_mm": sums["hold_ade_exec"] / n,
        "hold_fde_mm": sums["hold_fde"] / n,
        "cv_ade_mm": sums["cv_ade"] / n if has_velocity else float("nan"),
        "cv_ade_exec_mm": sums["cv_ade_exec"] / n if has_velocity else float("nan"),
        "cv_fde_mm": sums["cv_fde"] / n if has_velocity else float("nan"),
        "progress_ratio": sums["prog_num"] / max(1e-9, sums["prog_den"]),
        "done_acc": (tp + tn) / max(1, n),
        "done_precision": prec,
        "done_recall": rec,
        "done_f1": 2 * prec * rec / max(1e-9, prec + rec),
    }


@torch.no_grad()
def evaluate_denoising_loss(policy, dataset, indices: Sequence[int], device: torch.device, batch_size: int = 128,
                            num_workers: int = 2, seed: int = 0) -> float:
    """Pérdida de difusión media (ruido y timesteps fijos por semilla) en la representación de entrenamiento."""
    if len(indices) == 0:
        return float("nan")
    was_training = policy.training
    policy.eval()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    total, n = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = policy.forward(batch)["loss"]
            total += float(loss.item()) * batch["action"].shape[0]
            n += batch["action"].shape[0]
    if was_training:
        policy.train()
    return total / max(1, n)


def fixed_subset(n_total: int, n: int, seed: int) -> np.ndarray:
    """Subconjunto reproducible de índices (todos si ``n <= 0`` o ``n >= n_total``)."""
    if n <= 0 or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, size=n, replace=False))
