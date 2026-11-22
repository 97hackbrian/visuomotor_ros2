"""Evaluación offline de una Diffusion Policy por muestreo completo (DDIM) sobre ventanas de episodios.

Mide lo que el robot ejecutaría, no la pérdida de difusión (la pérdida de validación NO predice el éxito de la
política: Mandlekar et al. 2021, §4.5; Chi et al. 2023). Métricas en la representación absoluta del estado
(``base - home``), comparando el horizonte ejecutable predicho (pasos 1..H-1) con la demostración:

* ``ade_mm`` / ``fde_mm``: error medio / final de posición del horizonte completo.
* ``ade_exec_mm``: error medio sobre los ``n_action_steps`` primeros pasos (lo que realmente se ejecuta).
* ``rot_deg``: error geodésico medio de orientación.
* ``done_*``: exactitud, precisión, exhaustividad y F1 del flag de término en el primer paso ejecutado.
* ``hold_ade_mm`` / ``hold_ade_exec_mm``: línea base trivial "quedarse quieto" (predicción = pose actual) en el
  horizonte completo / en los pasos ejecutados. Un modelo útil debe batirla con holgura.
* ``progress_ratio``: desplazamiento predicho / demostrado al final del horizonte (1 = avance correcto; << 1 indica
  política "perezosa" que se detendría en el robot).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _geodesic_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q2 = q2 / q2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def evaluate_sampling(policy, dataset, indices: Sequence[int], device: torch.device, n_action_steps: int,
                      batch_size: int = 128, num_workers: int = 2, seed: int = 0,
                      done_threshold: float = 0.5) -> Dict[str, float]:
    """``dataset`` debe entregar acciones ABSOLUTAS, sin aumentos ni dropout propioceptivo."""
    if len(indices) == 0:
        return {}
    was_training = policy.training
    policy.eval()
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    sums = {"ade": 0.0, "fde": 0.0, "ade_exec": 0.0, "rot": 0.0, "hold_ade": 0.0, "hold_ade_exec": 0.0,
            "prog_num": 0.0, "prog_den": 0.0}
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
            k = max(1, min(int(n_action_steps), h))
            sums["ade"] += pos_err.mean(1).sum().item()
            sums["fde"] += pos_err[:, -1].sum().item()
            sums["ade_exec"] += pos_err[:, :k].mean(1).sum().item()
            sums["hold_ade"] += hold_err.mean(1).sum().item()
            sums["hold_ade_exec"] += hold_err[:, :k].mean(1).sum().item()
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
