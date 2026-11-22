#!/usr/bin/env python
"""Entrenamiento supervisado de Diffusion Policy (off-ROS).

Novedades respecto a la versión histórica (todas opcionales; sin flags nuevos el entrenamiento es equivalente salvo
la EMA de pesos, activa por defecto como en Diffusion Policy/UMI):
* EMA de pesos (power 0.75, máx. 0.9999): se guardan y evalúan los pesos EMA.
* Warmup lineal del learning rate + coseno.
* Partición train/val estricta por episodio (``--val_ratio``) y evaluación offline por muestreo completo (ADE/FDE en
  mm, error de rotación, F1 del flag done, línea base "quedarse quieto") cada ``--eval_every`` épocas.
* Representación de acción ``absolute`` o ``relative_ee`` (UMI) y convención de hemisferio del cuaternión leída de los
  atributos del dataset (``--quaternion_hemisphere auto``); ambas quedan en config.json del checkpoint.
* ColorJitter (UMI) opcional, dropout propioceptivo configurable y figuras en inglés con el nombre del dataset.
"""

import csv
import json
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr

from visuomotor_core.dataset.zarr_dataset import ZarrDataset, list_episodes, split_episodes
from visuomotor_core.models.config import DiffusionConfig
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from visuomotor_core.training.augment import make_gpu_augment
from visuomotor_core.training.ema import EMAModel
from visuomotor_core.training.evaluation import (estimate_action_offset, evaluate_denoising_loss, evaluate_sampling,
                                                 fixed_subset)
from visuomotor_core.training.plots import plot_eval_curves, plot_training_curves
from visuomotor_core.utils.logger import benchmark, configure_logging

UMI_COLOR_JITTER = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}


def _rss_gb() -> float:
    """RSS del proceso principal (GB) desde /proc (sin dependencias)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def resolve_quaternion_convention(root: str, requested: str):
    """``auto``: lee la convención que dejó el conversor del dataset (atributos ``xarm5_conversion``)."""
    attrs = dict(zarr.open_group(root, mode="r").attrs).get("xarm5_conversion", {})
    declared = attrs.get("quaternion_hemisphere")
    if requested != "auto" and declared is not None and requested != declared:
        # el dataset NO se re-canoniza: etiquetar el checkpoint con otra convención engañaría a policy_node ('auto')
        raise ValueError(f"--quaternion_hemisphere {requested} contradice el dataset ({declared}); usar 'auto'")
    mode = (declared or "qw_positive") if requested == "auto" else requested
    if mode == "qw_positive":
        return "qw_positive", None
    ref = attrs.get("quaternion_reference_xyzw")
    if mode == "home" and ref is None:
        raise ValueError("quaternion_hemisphere='home' pero el dataset no declara quaternion_reference_xyzw")
    return mode, (list(map(float, ref)) if mode == "home" else None)


def main(
    root,
    epoch,
    batch_size,
    out_dir,
    save_every,
    no_tracker: bool = False,
    steps_per_epoch: int = 0,
    log_freq: int = 10,
    plot_freq: int = 500,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    eval_convergence: bool = True,
    max_episodes: int = 0,
    val_ratio: float = 0.0,
    val_seed: int = 42,
    eval_every: int = 0,
    eval_train_windows: int = 1024,
    eval_val_windows: int = 0,
    use_ema: bool = True,
    ema_power: float = 0.75,
    ema_max: float = 0.9999,
    warmup_steps: int = 500,
    base_lr: float = 1e-4,
    action_representation: str = "absolute",
    quaternion_hemisphere: str = "auto",
    color_jitter: bool = False,
    proprio_dropout: float = 0.15,
    seed: int = 0,
    resume: bool = False,
    max_rss_gb: float = 0.0,
    train_plot_dpi: int = 120,
):
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    configure_logging(level=logging.DEBUG, log_file=str(output_directory / "training.log"))
    log = logging.getLogger("train")

    log.info(f"Ruta de dataset: {root}")
    if root is None:
        log.error("¡Debe especificarse la ruta al dataset Zarr (--path / --root)!")
        return
    dataset_name = Path(str(root).rstrip("/")).name.replace(".zarr", "")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    # ---------------- Datos: partición estricta por episodio ----------------
    all_eps = list_episodes(root)
    if max_episodes > 0:
        all_eps = all_eps[:max_episodes]
    train_eps, val_eps = split_episodes(all_eps, val_ratio, val_seed)
    q_mode, q_ref = resolve_quaternion_convention(root, quaternion_hemisphere)
    log.info(f"Episodios: {len(train_eps)} train / {len(val_eps)} val (semilla {val_seed}). "
             f"Acción: {action_representation} | Cuaternión: {q_mode} ref={q_ref}")
    if val_eps:
        log.info(f"Episodios de validación: {val_eps}")
    # Alineación acción/estado para la línea base "velocidad constante" de la evaluación offline
    action_offset = estimate_action_offset(root, train_eps)
    log.info(f"Línea base cinemática: action[t] ~ state[t+{action_offset}] (estimado en los episodios de entrenamiento)")

    cfg = DiffusionConfig(action_representation=action_representation, quaternion_hemisphere=q_mode,
                          quaternion_reference=q_ref)
    jitter = UMI_COLOR_JITTER if color_jitter else None
    with benchmark("Inicialización ZarrDataset", log):
        # ColorJitter en GPU (augment.py): en los workers de CPU dividía por 10 el rendimiento del DataLoader
        base_dataset = ZarrDataset(root=root, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon, episodes=train_eps,
                                   action_representation=action_representation, color_jitter=None,
                                   proprio_dropout_prob=proprio_dropout)
    gpu_augment = make_gpu_augment(jitter)
    stats = base_dataset.stats
    # Conjuntos de evaluación: sin aumentos ni dropout. *_abs entrega acciones absolutas (verdad terreno).
    eval_kw = dict(root=root, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon, stats=stats,
                   proprio_dropout_prob=0.0, color_jitter=None)
    val_loss_ds = ZarrDataset(episodes=val_eps, action_representation=action_representation, **eval_kw) if val_eps else None
    val_abs_ds = ZarrDataset(episodes=val_eps, action_representation="absolute", **eval_kw) if val_eps else None
    train_abs_ds = ZarrDataset(episodes=train_eps, action_representation="absolute", **eval_kw)

    obs_dim = stats["observation.state"]["mean"].shape[0]
    act_dim = stats["action"]["mean"].shape[0]
    cfg.input_shapes["observation.state"] = [obs_dim]
    cfg.output_shapes["action"] = [act_dim]
    if getattr(base_dataset, "constant_state_dims", None):
        log.info(f"Dimensiones de estado constantes (std = 1 en la normalización): {base_dataset.constant_state_dims}")
    log.info(f"Dataset {dataset_name}: {len(base_dataset.episodes)} episodios train ({len(base_dataset.frames)} "
             f"ventanas). Dim Observación: {obs_dim} | Dim Acción: {act_dim}")

    dataloader = torch.utils.data.DataLoader(
        base_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=True,
        prefetch_factor=(max(1, prefetch_factor) if num_workers > 0 else None))
    # Presupuesto de RAM: cada lote en vuelo ocupa ~batch x n_obs x 3 x 256 x 256 x 4 B (≈100 MB a batch 64) en memoria
    # compartida + copia fijada; con varios entrenamientos simultáneos y prefetch alto se agotó la RAM del host (OOM).
    log.info(f"DataLoader: {num_workers} workers x prefetch {prefetch_factor} -> "
             f"~{num_workers * max(1, prefetch_factor) * batch_size * cfg.n_obs_steps * 3 * 256 * 256 * 4 / 1e9:.1f} GB "
             "de lotes en vuelo")

    policy = DiffusionPolicy(cfg, dataset_stats=stats)
    policy.train()
    policy.to(device)
    orig_policy = policy
    ema = EMAModel(orig_policy, power=ema_power, max_value=ema_max) if use_ema else None

    lr = base_lr * (batch_size / 64) ** 0.5  # escalado sqrt para Adam al variar el batch
    optimizer = torch.optim.AdamW(orig_policy.parameters(), lr=lr, betas=(0.95, 0.999), weight_decay=1e-6)

    n = len(base_dataset)
    dataset_steps_per_epoch = max(1, n // batch_size)
    epochs = epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = dataset_steps_per_epoch
    is_infinite_epochs = (epochs <= 0 or epochs >= 999999)
    total_steps = (epochs * steps_per_epoch) if not is_infinite_epochs else 999999999
    display_epochs = "∞" if is_infinite_epochs else str(epochs)
    display_total_steps = "∞" if is_infinite_epochs else f"{total_steps:,}"
    eval_every = eval_every if eval_every > 0 else max(1, save_every)
    log.info(f"batch_size={batch_size} | lr={lr:.2e} | warmup={warmup_steps} | EMA={use_ema} | épocas {display_epochs} "
             f"| pasos/época {steps_per_epoch:,} | pasos totales {display_total_steps} | eval cada {eval_every} ép.")

    scheduler_t_max = total_steps if not is_infinite_epochs else max(100 * steps_per_epoch, 200000)

    def lr_lambda(s):
        if s < warmup_steps:
            return (s + 1) / float(max(1, warmup_steps))
        prog = min(1.0, (s - warmup_steps) / float(max(1, scheduler_t_max - warmup_steps)))
        eta_min = 1e-6 / lr
        return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    training_meta = {
        "dataset_path": str(root), "dataset_name": dataset_name, "dataset_length": n, "batch_size": batch_size,
        "epochs": epochs, "is_infinite": is_infinite_epochs, "steps_per_epoch": steps_per_epoch,
        "dataset_steps_per_epoch": dataset_steps_per_epoch, "total_steps": total_steps if not is_infinite_epochs else -1,
        "lr": lr, "warmup_steps": warmup_steps, "use_ema": use_ema, "ema_power": ema_power, "ema_max": ema_max,
        "log_freq": log_freq, "plot_freq": plot_freq, "save_every": save_every, "eval_every": eval_every,
        "val_ratio": val_ratio, "val_seed": val_seed, "train_episodes": train_eps, "val_episodes": val_eps,
        "action_representation": action_representation, "quaternion_hemisphere": q_mode,
        "quaternion_reference": q_ref, "color_jitter": jitter, "proprio_dropout": proprio_dropout, "seed": seed,
        "eval_action_offset_steps": action_offset,
    }
    with open(output_directory / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    stop_requested = False

    def handle_stop_signal(sig, frame):
        nonlocal stop_requested
        log.warning(f"[Señal Recibida: {sig}] Solicitud de parada ordenada detectada. Guardando estado y métricas...")
        stop_requested = True

    try:
        signal.signal(signal.SIGINT, handle_stop_signal)
        signal.signal(signal.SIGTERM, handle_stop_signal)
    except Exception as e:
        log.warning(f"No se pudieron registrar manejadores de señales: {e}")

    step = 0
    step_history, epoch_history, eval_history, ema_history = [], [], [], []
    ema_loss = None
    current_lr = lr
    overall_start_time = time.time()
    start_epoch = 1
    state_path = output_directory / "training_state.pt"

    def save_training_state(last_epoch: int):
        """Estado completo para reanudar (--resume): modelo, optimizador, scheduler, EMA, RNG e historiales.

        ``last_epoch`` debe ser la última época COMPLETA: una época interrumpida se repite entera al reanudar (sus
        entradas parciales se descartan de los historiales).
        """
        state = {
            "step": step, "epoch": last_epoch, "ema_loss": ema_loss,
            "model": orig_policy.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "ema": ema.model.state_dict() if ema is not None else None,
            "ema_step": ema.optimization_step if ema is not None else 0,
            "ema_decay": ema.decay if ema is not None else 0.0,
            "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(), "rng_np": np.random.get_state(),
            "histories": {"step": step_history, "ema": ema_history,
                          "epoch": [r for r in epoch_history if r["epoch"] <= last_epoch],
                          "eval": [r for r in eval_history if r["epoch"] <= last_epoch]},
            "training_meta": training_meta,
        }
        tmp = state_path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, state_path)  # escritura atómica: un corte nunca deja un estado corrupto
        log.info(f"[Estado] training_state.pt guardado (época {last_epoch}, paso {step})")

    if resume:
        if not state_path.exists():
            raise FileNotFoundError(f"--resume sin {state_path}")
        st = torch.load(state_path, map_location=device, weights_only=False)
        for key in ("dataset_path", "action_representation", "batch_size", "epochs", "val_episodes", "steps_per_epoch",
                    "lr", "warmup_steps", "quaternion_hemisphere", "quaternion_reference"):
            if st["training_meta"].get(key) != training_meta.get(key):
                raise ValueError(f"--resume con configuración distinta en '{key}': "
                                 f"{st['training_meta'].get(key)!r} != {training_meta.get(key)!r}")
        orig_policy.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if ema is not None and st.get("ema") is not None:
            ema.model.load_state_dict(st["ema"])
            ema.optimization_step, ema.decay = st["ema_step"], st["ema_decay"]
        torch.set_rng_state(st["rng_cpu"].cpu())          # los estados RNG deben ser ByteTensor de CPU
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        np.random.set_state(st["rng_np"])
        step, ema_loss, start_epoch = st["step"], st["ema_loss"], st["epoch"] + 1
        h = st["histories"]
        step_history, epoch_history, eval_history, ema_history = h["step"], h["epoch"], h["eval"], h["ema"]
        current_lr = float(scheduler.get_last_lr()[0])
        log.warning(f"[Reanudación] Desde época {start_epoch} (paso {step}); historiales restaurados.")
        del st

    try:
        log.info("Compilando modelo con torch.compile para mayor velocidad...")
        policy = torch.compile(policy)
        log.info("¡Modelo compilado exitosamente!")
    except Exception as e:
        log.warning(f"Omitiendo compilación de modelo: {e}")

    def open_csv(path, header, rows=None):
        exists = path.exists() and rows is None
        fh = open(path, mode="a" if exists else "w", newline="")
        w = csv.writer(fh)
        if not exists:
            w.writerow(header)
            for r in rows or []:
                w.writerow(r)
            fh.flush()
        return fh, w

    eval_keys = ["n_windows", "ade_mm", "ade_exec_mm", "fde_mm", "rot_deg", "hold_ade_mm", "hold_ade_exec_mm", "progress_ratio",
                 "done_acc", "done_precision", "done_recall", "done_f1",
                 "hold_fde_mm", "cv_ade_mm", "cv_ade_exec_mm", "cv_fde_mm"]
    rows_step = rows_epoch = rows_eval = None
    if resume:
        rows_step = [[r["step"], r["epoch"], r["loss"], r["loss_ema"], f"{r['lr']:.6e}", r["vram_peak_mb"],
                      f"{e['decay']:.6f}", r.get("timestamp", "")] for r, e in zip(step_history, ema_history)]
        rows_epoch = [[r["epoch"], r["avg_loss"], f"{r['final_lr']:.6e}", r["duration_s"], f"{r['val_loss']:.6f}",
                       r.get("timestamp", "")] for r in epoch_history]
        rows_eval = [[r["epoch"], r["step"], r["split"], r["denoise_loss"]] + [r.get(k) for k in eval_keys]
                     + [r.get("timestamp", "")] for r in eval_history]

    step_csv_file, step_writer = open_csv(output_directory / "training_metrics.csv",
                                          ["step", "epoch", "loss", "loss_ema", "lr", "vram_peak_mb", "ema_decay", "timestamp"],
                                          rows_step)
    epoch_csv_file, epoch_writer = open_csv(output_directory / "epoch_metrics.csv",
                                            ["epoch", "avg_loss", "final_lr", "duration_s", "val_loss", "timestamp"], rows_epoch)
    eval_csv_file, eval_writer = open_csv(output_directory / "eval_metrics.csv",
                                          ["epoch", "step", "split", "denoise_loss"] + eval_keys + ["timestamp"], rows_eval)

    train_eval_idx = fixed_subset(len(train_abs_ds), eval_train_windows, seed=val_seed)
    val_eval_idx = fixed_subset(len(val_abs_ds), eval_val_windows, seed=val_seed) if val_abs_ds else []
    val_loss_idx = np.arange(len(val_loss_ds)) if val_loss_ds else []
    eval_model = (lambda: ema.model) if ema is not None else (lambda: orig_policy)

    def run_evaluation(epoch_idx):
        model = eval_model()
        res = {}
        with benchmark(f"Evaluación offline (época {epoch_idx})", log, track_gpu=False):
            val_loss = evaluate_denoising_loss(model, val_loss_ds, val_loss_idx, device) if val_loss_ds else float("nan")
            for split, ds, idx in (("val", val_abs_ds, val_eval_idx), ("train", train_abs_ds, train_eval_idx)):
                if ds is None or len(idx) == 0:
                    continue
                m = evaluate_sampling(model, ds, idx, device, n_action_steps=cfg.n_action_steps,
                                      action_offset=action_offset)
                row = {"epoch": epoch_idx, "step": step, "split": split,
                       "denoise_loss": val_loss if split == "val" else float("nan"), **m,
                       "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                eval_history.append(row)
                eval_writer.writerow([epoch_idx, step, split, row["denoise_loss"]] + [m.get(k) for k in eval_keys]
                                     + [datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                res[split] = m
                log.info(f"[Eval {split}] época {epoch_idx}: ADE {m['ade_mm']:.1f} mm | ADE exec {m['ade_exec_mm']:.1f} "
                         f"| FDE {m['fde_mm']:.1f} | rot {m['rot_deg']:.2f}° | hold-baseline {m['hold_ade_mm']:.1f}/{m['hold_ade_exec_mm']:.1f} mm "
                         f"| vel.-cte ADE/exec/FDE {m['cv_ade_mm']:.1f}/{m['cv_ade_exec_mm']:.1f}/{m['cv_fde_mm']:.1f} mm "
                         f"| progreso {m['progress_ratio']:.2f} | done F1 {m['done_f1']:.3f}")
            eval_csv_file.flush()
            try:
                plot_eval_curves(output_directory, eval_history, steps_per_epoch, dataset_name, dpi=train_plot_dpi)
            except Exception as e:
                log.debug(f"Figura de evaluación omitida: {e}")
        return val_loss

    dataloader_iter = iter(dataloader)
    epoch_idx = start_epoch
    last_completed_epoch = start_epoch - 1
    stop_file = output_directory / "STOP"
    global_stop_file = Path("/tmp/stop_visuomotor_training.flag")
    try:
        while (is_infinite_epochs or epoch_idx <= epochs) and not stop_requested:
            if stop_file.exists() or global_stop_file.exists():
                log.warning("[Parada Limpia] Archivo de parada detectado en disco. Finalizando entrenamiento...")
                break

            epoch_start_time = time.time()
            epoch_step_losses = []
            epoch_step_count = 0

            while epoch_step_count < steps_per_epoch:
                if stop_requested or stop_file.exists() or global_stop_file.exists():
                    log.warning(f"[Parada Limpia] Interrupción detectada en Época {epoch_idx}, Paso {step}.")
                    break
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if gpu_augment is not None:
                    batch["observation.image"] = gpu_augment(batch["observation.image"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = policy.forward(batch)["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(orig_policy.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.step(orig_policy)

                loss_val = float(loss.item())
                if not np.isfinite(loss_val):
                    log.error(f"Pérdida no finita en el paso {step}: {loss_val}. Deteniendo.")
                    stop_requested = True
                    break
                ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)
                current_lr = float(scheduler.get_last_lr()[0])

                is_terminal_step = (epoch_step_count == steps_per_epoch - 1) or (not is_infinite_epochs and step == total_steps - 1)
                if step % log_freq == 0 or is_terminal_step:
                    vram_mb = float(torch.cuda.max_memory_allocated() / 1e6)
                    ram_gb = _rss_gb()
                    torch.cuda.reset_peak_memory_stats()
                    decay = ema.decay if ema is not None else 0.0
                    if step % (log_freq * 10) == 0 or is_terminal_step:
                        log.info(f"Época {epoch_idx}/{display_epochs} | Paso {step}/{display_total_steps} | Loss: {loss_val:.4f} "
                                 f"(EMA: {ema_loss:.4f}) | LR: {current_lr:.2e} | EMA decay {decay:.5f} | VRAM(peak): {vram_mb:.0f}MB "
                                 f"| RAM proceso: {ram_gb:.1f}GB")
                    entry = {"step": step, "epoch": epoch_idx, "loss": round(loss_val, 6), "loss_ema": round(ema_loss, 6),
                             "lr": current_lr, "vram_peak_mb": round(vram_mb, 1),
                             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    step_history.append(entry)
                    ema_history.append({"step": step, "decay": decay})
                    step_writer.writerow([step, epoch_idx, entry["loss"], entry["loss_ema"], f"{current_lr:.6e}",
                                          entry["vram_peak_mb"], f"{decay:.6f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    step_csv_file.flush()
                    if max_rss_gb > 0 and ram_gb > max_rss_gb:
                        log.error(f"[Guardia RAM] RSS {ram_gb:.1f} GB > {max_rss_gb:.1f} GB: se guarda el estado y se detiene "
                                  "limpiamente para evitar un OOM del host. Reanudar con --resume.")
                        stop_requested = True
                        step += 1
                        epoch_step_count += 1
                        epoch_step_losses.append(loss_val)
                        break

                if plot_freq > 0 and step > 0 and step % plot_freq == 0:
                    try:
                        plot_training_curves(output_directory, step_history, epoch_history, steps_per_epoch,
                                             dataset_name, ema_history, dpi=train_plot_dpi)
                    except Exception as e:
                        log.debug(f"Renderizado periódico de curvas omitido: {e}")

                step += 1
                epoch_step_count += 1
                epoch_step_losses.append(loss_val)

            epoch_dur = round(time.time() - epoch_start_time, 2)
            epoch_avg_loss = float(np.mean(epoch_step_losses)) if epoch_step_losses else 0.0
            completed_epoch = epoch_step_count >= steps_per_epoch
            if completed_epoch:
                last_completed_epoch = epoch_idx

            val_loss = float("nan")
            if completed_epoch and (epoch_idx % eval_every == 0 or epoch_idx == 1):
                val_loss = run_evaluation(epoch_idx)

            epoch_entry = {"epoch": epoch_idx, "avg_loss": round(epoch_avg_loss, 6), "final_lr": current_lr,
                           "duration_s": epoch_dur, "val_loss": val_loss,
                           "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            epoch_history.append(epoch_entry)
            epoch_writer.writerow([epoch_idx, epoch_entry["avg_loss"], f"{current_lr:.6e}", epoch_dur,
                                   f"{val_loss:.6f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            epoch_csv_file.flush()
            log.info(f"[Resumen Época] Época {epoch_idx}/{display_epochs} en {epoch_dur}s | Loss promedio: {epoch_avg_loss:.4f}")

            # Curvas solo en épocas de evaluación (renderizar cada época a 300 DPI hacía crecer la RAM del proceso)
            if completed_epoch and (epoch_idx % eval_every == 0 or epoch_idx == 1):
                try:
                    plot_training_curves(output_directory, step_history, epoch_history, steps_per_epoch, dataset_name,
                                         ema_history, dpi=train_plot_dpi)
                except Exception as e:
                    log.debug(f"Curvas omitidas: {e}")

            save_model = eval_model()
            save_model.save_pretrained(output_directory)  # último checkpoint (pesos EMA si está activa)
            if completed_epoch and save_every > 0 and epoch_idx % save_every == 0:
                ckpt_dir = output_directory / "checkpoints" / f"epoch_{epoch_idx}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_model.save_pretrained(ckpt_dir)
                log.info(f"[Checkpoint] Guardado {ckpt_dir} (paso {step})")
                save_training_state(epoch_idx)

            if stop_requested:
                break
            epoch_idx += 1

    except KeyboardInterrupt:
        log.warning("Entrenamiento interrumpido manualmente (KeyboardInterrupt). Guardando estado actual...")
    finally:
        step_csv_file.close()
        epoch_csv_file.close()
        try:
            save_training_state(last_completed_epoch)
        except Exception as e:
            log.warning(f"[Estado] No se pudo guardar training_state.pt: {e}")

    # Evaluación final (si la última época no coincidió con eval_every)
    try:
        if step > 0 and (not eval_history or eval_history[-1]["step"] != step):
            run_evaluation(len(epoch_history))
    except Exception as e:
        log.warning(f"Evaluación final omitida: {e}")
    finally:
        eval_csv_file.close()

    plot_training_curves(output_directory, step_history, epoch_history, steps_per_epoch, dataset_name, ema_history, dpi=300)
    try:
        plot_eval_curves(output_directory, eval_history, steps_per_epoch, dataset_name, dpi=300)
    except Exception as e:
        log.debug(f"Figura final de evaluación omitida: {e}")

    convergence_metadata = {}
    if eval_convergence and step_history:
        try:
            from visuomotor_core.analysis.stopping_criteria import ConvergenceAnalyzer
            analyzer = ConvergenceAnalyzer(steps_per_epoch=steps_per_epoch, step_history=step_history,
                                           epoch_history=epoch_history, dataset_name=dataset_name)
            recs = analyzer.analyze()
            log.info("\n" + analyzer.generate_summary_table())
            conv_dir = output_directory / "convergence"
            analyzer.save_report(conv_dir)
            analyzer.plot_dual_axis(conv_dir, dpi=300, show_diagnostics=True, dataset_name=dataset_name, language="en")
            convergence_metadata = {k: {"epoch": v.recommended_epoch, "step": v.recommended_step,
                                        "loss": v.loss_at_stop, "confidence": v.confidence} for k, v in recs.items()}
        except Exception as e:
            log.warning(f"[Convergencia] Error no crítico al evaluar convergencia: {e}")

    eval_model().save_pretrained(output_directory)
    log.info(f"Modelo y métricas guardados exitosamente en: {output_directory}")

    if not no_tracker:
        try:
            from visuomotor_core.utils.tracker import record_train_run
            total_dur = round(time.time() - overall_start_time, 2)
            metadata_dict = {"horizon": cfg.horizon, "n_obs_steps": cfg.n_obs_steps, "obs_dim": obs_dim,
                             "act_dim": act_dim, "steps_per_epoch": steps_per_epoch,
                             "action_representation": action_representation, "quaternion_hemisphere": q_mode,
                             "use_ema": use_ema, "val_episodes": val_eps}
            if eval_history:
                metadata_dict["last_eval"] = eval_history[-2:]
            if convergence_metadata:
                metadata_dict["convergence"] = convergence_metadata
            record_train_run(
                run_id=f"dp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{output_directory.name}",
                dataset_path=str(root), epochs_requested=epochs, epochs_completed=len(epoch_history),
                batch_size=batch_size, lr=lr,
                final_loss=epoch_history[-1]["avg_loss"] if epoch_history else 0.0,
                final_loss_ema=step_history[-1]["loss_ema"] if step_history else 0.0,
                vram_peak_mb=max([r["vram_peak_mb"] for r in step_history]) if step_history else 0.0,
                duration_s=total_dur, checkpoint_dir=str(output_directory), metadata=metadata_dict)
        except Exception as e:
            log.warning(f"[DB Fail-Safe] Error no crítico al registrar en SQLite: {e}")


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Train diffusion policy')
    parser.add_argument('--path', type=str, help='Data path')
    parser.add_argument('--root', type=str, default=None, help='Data path (alias for --path)')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs (0 = infinito)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (n_obs_steps=2 duplica el batch de '
                        'imágenes del encoder; vigile la VRAM impresa)')
    parser.add_argument('--out_dir', type=str, default='outputs/train', help='Output directory')
    parser.add_argument('--save_every', type=int, default=25, help='Checkpoint separado cada N épocas')
    parser.add_argument('--steps_per_epoch', type=int, default=0, help='Pasos por época (0 = tamaño del dataset)')
    parser.add_argument('--log_freq', type=int, default=10, help='Frecuencia (pasos) de métricas a CSV')
    parser.add_argument('--plot_freq', type=int, default=500, help='Frecuencia (pasos) de figuras Matplotlib')
    parser.add_argument('--num_workers', type=int, default=8, help='Workers del DataLoader')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Lotes precargados por worker (RAM ~100 MB/lote a batch 64)')
    parser.add_argument('--max_episodes', type=int, default=0, help='Máximo de episodios a cargar (0 = todos)')
    parser.add_argument('--no_eval_convergence', action='store_true', help='Desactivar análisis de convergencia final')
    parser.add_argument('--no_tracker', action='store_true', help='Desactivar registro en SQLite')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='Fracción de EPISODIOS de validación (0 = sin val)')
    parser.add_argument('--val_seed', type=int, default=42)
    parser.add_argument('--eval_every', type=int, default=0, help='Evaluación offline cada N épocas (0 = save_every)')
    parser.add_argument('--eval_train_windows', type=int, default=1024, help='Ventanas de train evaluadas (0 = todas)')
    parser.add_argument('--eval_val_windows', type=int, default=0, help='Ventanas de val evaluadas (0 = todas)')
    parser.add_argument('--no_ema', action='store_true', help='Desactivar la EMA de pesos')
    parser.add_argument('--ema_power', type=float, default=0.75)
    parser.add_argument('--ema_max', type=float, default=0.9999)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4, help='LR base para batch 64 (escalado sqrt con el batch)')
    parser.add_argument('--action_representation', choices=['absolute', 'relative_ee'], default='absolute')
    parser.add_argument('--quaternion_hemisphere', choices=['auto', 'qw_positive', 'home'], default='auto')
    parser.add_argument('--color_jitter', action='store_true', help='ColorJitter UMI (brillo .3, contraste .4, sat .5, tono .08)')
    parser.add_argument('--proprio_dropout', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Reanudar desde <out_dir>/training_state.pt')
    parser.add_argument('--max_rss_gb', type=float, default=0.0,
                        help='Guardia de RAM: guardar estado y detener si el proceso supera estos GB (0 = desactivada)')
    parser.add_argument('--train_plot_dpi', type=int, default=120, help='DPI de las figuras intermedias (las finales: 300)')
    a = parser.parse_args()

    main(root=a.root if a.root is not None else a.path, epoch=a.epoch, batch_size=a.batch_size, out_dir=a.out_dir,
         save_every=a.save_every, no_tracker=a.no_tracker, steps_per_epoch=a.steps_per_epoch, log_freq=a.log_freq,
         plot_freq=a.plot_freq, num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, eval_convergence=not a.no_eval_convergence,
         max_episodes=a.max_episodes, val_ratio=a.val_ratio, val_seed=a.val_seed, eval_every=a.eval_every,
         eval_train_windows=a.eval_train_windows, eval_val_windows=a.eval_val_windows, use_ema=not a.no_ema,
         ema_power=a.ema_power, ema_max=a.ema_max, warmup_steps=a.warmup_steps, base_lr=a.lr,
         action_representation=a.action_representation, quaternion_hemisphere=a.quaternion_hemisphere,
         color_jitter=a.color_jitter, proprio_dropout=a.proprio_dropout, seed=a.seed, resume=a.resume,
         max_rss_gb=a.max_rss_gb, train_plot_dpi=a.train_plot_dpi)


if __name__ == '__main__':
    main_cli()
