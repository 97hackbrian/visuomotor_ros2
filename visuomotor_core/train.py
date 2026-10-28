#!/usr/bin/env python

import os
import csv
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import logging
from visuomotor_core.utils.logger import configure_logging, benchmark
from visuomotor_core.models.config import DiffusionConfig
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from visuomotor_core.dataset.zarr_dataset import ZarrDataset


def plot_training_curves(output_directory: Path, step_history: list, epoch_history: list):
    """
    Genera gráficos multi-panel de calidad de publicación para el entrenamiento
    supervisado de Diffusion Policy en formato PNG y PDF sin dependencias de X11.
    """
    if not step_history:
        return

    loss_dir = output_directory / "losses"
    loss_dir.mkdir(parents=True, exist_ok=True)

    steps = [r["step"] for r in step_history]
    losses = [r["loss"] for r in step_history]
    ema_losses = [r["loss_ema"] for r in step_history]
    lrs = [r["lr"] for r in step_history]
    vrams = [r["vram_peak_mb"] for r in step_history]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Dinámica de Entrenamiento Supervisado: Diffusion Policy (DDIM + U-Net)", fontsize=16, fontweight='bold')

    # Subplot 1: Step Loss crudo + Suavizado EMA
    axs[0, 0].plot(steps, losses, color='#1f77b4', alpha=0.25, linewidth=1.0, label='Loss Crudo (Paso)')
    axs[0, 0].plot(steps, ema_losses, color='#d62728', linewidth=2.0, label='Loss EMA (Suavizado)')
    axs[0, 0].set_title("Evolución de Pérdida por Paso de Optimización", fontweight='bold')
    axs[0, 0].set_xlabel("Paso de Entrenamiento (Step)")
    axs[0, 0].set_ylabel("Loss (MSE)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="upper right")

    # Subplot 2: Pérdida Media por Época (Macro Convergencia)
    if epoch_history:
        epochs = [r["epoch"] for r in epoch_history]
        epoch_losses = [r["avg_loss"] for r in epoch_history]
        axs[0, 1].plot(epochs, epoch_losses, color='#2ca02c', marker='o', linewidth=2.0, markersize=4, label='Pérdida Media / Época')
        axs[0, 1].set_title("Convergencia Macro por Época", fontweight='bold')
        axs[0, 1].set_xlabel("Época")
        axs[0, 1].set_ylabel("Pérdida Media (MSE)")
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)
        axs[0, 1].legend(loc="upper right")
    else:
        axs[0, 1].text(0.5, 0.5, "Esperando conclusión de la primera época...", ha='center', va='center')
        axs[0, 1].set_title("Convergencia Macro por Época", fontweight='bold')

    # Subplot 3: Tasa de Aprendizaje (Cosine Annealing LR)
    axs[1, 0].plot(steps, lrs, color='#9467bd', linewidth=2.0, label='Learning Rate')
    axs[1, 0].set_title("Cronograma de Tasa de Aprendizaje (Cosine Annealing)", fontweight='bold')
    axs[1, 0].set_xlabel("Paso de Entrenamiento (Step)")
    axs[1, 0].set_ylabel("Learning Rate")
    axs[1, 0].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend(loc="upper right")

    # Subplot 4: Uso de Memoria GPU VRAM Peak (MB)
    axs[1, 1].plot(steps, vrams, color='#ff7f0e', linewidth=1.8, label='VRAM Peak (MB)')
    axs[1, 1].set_title("Consumo de Memoria GPU (Peak VRAM)", fontweight='bold')
    axs[1, 1].set_xlabel("Paso de Entrenamiento (Step)")
    axs[1, 1].set_ylabel("Memoria VRAM (MB)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend(loc="upper right")

    plt.tight_layout()

    # Guardar versión canónica y versión timestamped
    plot_png = loss_dir / "training_curves.png"
    plot_pdf = loss_dir / "training_curves.pdf"
    plt.savefig(plot_png, dpi=300, bbox_inches="tight")
    plt.savefig(plot_pdf, bbox_inches="tight")

    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(loss_dir / f"loss_{time_str}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(root, epoch, batch_size, out_dir, save_every, no_tracker: bool = False):
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    configure_logging(level=logging.DEBUG, log_file=str(output_directory / "training.log"))
    log = logging.getLogger("train")

    log.info(f"Ruta de dataset: {root}")
    if root is None:
        log.error("¡Debe especificarse la ruta al dataset Zarr (--path / --root)!")
        return

    repo_id = 'test'

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    log_freq = 20
    checkpoint_freq = 2000

    cfg = DiffusionConfig()
    
    # Primero inicializamos el dataset para poder leer cuántas dimensiones tiene la data (8 o 9)
    with benchmark("Inicialización ZarrDataset", log):
        base_dataset = ZarrDataset(root=root, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon)
    
    # Ajustamos dinámicamente el tamaño de la acción y del estado
    obs_dim = base_dataset.stats["observation.state"]["mean"].shape[0]
    act_dim = base_dataset.stats["action"]["mean"].shape[0]
    
    cfg.input_shapes["observation.state"] = [obs_dim]
    cfg.output_shapes["action"] = [act_dim]
    
    log.info(f"Dataset cargado. Dim Observación: {obs_dim} | Dim Acción: {act_dim}")
    
    dataloader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    policy = DiffusionPolicy(cfg, dataset_stats=base_dataset.stats)
    policy.train()
    policy.to(device)

    # Optional torch.compile for extra step execution speed
    try:
        log.info("Compilando modelo con torch.compile para mayor velocidad...")
        policy = torch.compile(policy)
        log.info("¡Modelo compilado exitosamente!")
    except Exception as e:
        log.warning(f"Omitiendo compilación de modelo: {e}")

    # sqrt-scaled LR: appropriate for Adam when varying batch size
    base_lr = 1e-4
    lr = base_lr * (batch_size / 64) ** 0.5
    log.info(f"Parámetros de entrenamiento: batch_size={batch_size} | lr={lr:.2e}")

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6
    )

    n = len(base_dataset)
    steps_per_epoch = max(1, n // batch_size)
    epochs = epoch
    total_steps = epochs * steps_per_epoch
    log.info(f"Total épocas: {epochs} | Pasos por época: {steps_per_epoch} | Pasos totales: {total_steps}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    step = 0
    train_losses = []
    step_history = []
    epoch_history = []
    ema_loss = None
    overall_start_time = time.time()

    # Archivos CSV para almacenamiento de métricas offline
    step_csv_path = output_directory / "training_metrics.csv"
    epoch_csv_path = output_directory / "epoch_metrics.csv"

    step_csv_exists = step_csv_path.exists()
    step_csv_file = open(step_csv_path, mode="a" if step_csv_exists else "w", newline="")
    step_writer = csv.writer(step_csv_file)
    if not step_csv_exists:
        step_writer.writerow(["step", "epoch", "loss", "loss_ema", "lr", "vram_peak_mb", "timestamp"])
        step_csv_file.flush()

    epoch_csv_exists = epoch_csv_path.exists()
    epoch_csv_file = open(epoch_csv_path, mode="a" if epoch_csv_exists else "w", newline="")
    epoch_writer = csv.writer(epoch_csv_file)
    if not epoch_csv_exists:
        epoch_writer.writerow(["epoch", "avg_loss", "final_lr", "duration_s", "timestamp"])
        epoch_csv_file.flush()

    try:
        for epoch_idx in range(1, epochs + 1):
            epoch_start_time = time.time()
            epoch_step_losses = []

            for batch in dataloader:
                # Move batch to GPU
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_dict = policy.forward(batch)
                    loss = output_dict["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                loss_val = float(loss.item())
                ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)
                current_lr = float(scheduler.get_last_lr()[0])
                vram_mb = float(torch.cuda.max_memory_allocated() / 1e6)

                if step % log_freq == 0:
                    log.info(f"Época {epoch_idx}/{epochs} | Paso {step}/{total_steps} | Loss: {loss_val:.4f} (EMA: {ema_loss:.4f}) | LR: {current_lr:.2e} | VRAM(peak): {vram_mb:.0f}MB")
                    torch.cuda.reset_peak_memory_stats()

                    step_entry = {
                        "step": step,
                        "epoch": epoch_idx,
                        "loss": round(loss_val, 6),
                        "loss_ema": round(ema_loss, 6),
                        "lr": current_lr,
                        "vram_peak_mb": round(vram_mb, 1),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    step_history.append(step_entry)
                    step_writer.writerow([
                        step_entry["step"],
                        step_entry["epoch"],
                        step_entry["loss"],
                        step_entry["loss_ema"],
                        f"{step_entry['lr']:.6e}",
                        step_entry["vram_peak_mb"],
                        step_entry["timestamp"]
                    ])
                    step_csv_file.flush()

                step += 1
                train_losses.append(loss_val)
                epoch_step_losses.append(loss_val)

            epoch_dur = round(time.time() - epoch_start_time, 2)
            epoch_avg_loss = float(np.mean(epoch_step_losses)) if epoch_step_losses else 0.0
            epoch_entry = {
                "epoch": epoch_idx,
                "avg_loss": round(epoch_avg_loss, 6),
                "final_lr": current_lr,
                "duration_s": epoch_dur,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            epoch_history.append(epoch_entry)
            epoch_writer.writerow([
                epoch_entry["epoch"],
                epoch_entry["avg_loss"],
                f"{epoch_entry['final_lr']:.6e}",
                epoch_entry["duration_s"],
                epoch_entry["timestamp"]
            ])
            epoch_csv_file.flush()

            log.info(f"[Resumen Época] Época {epoch_idx}/{epochs} completada en {epoch_dur}s | Loss Promedio: {epoch_avg_loss:.4f}")

            # Generar/actualizar gráficos offline de matplotlib al finalizar la época
            with benchmark("Generación de curvas Matplotlib (Headless)", log):
                plot_training_curves(output_directory, step_history, epoch_history)

            # Guardar el checkpoint exactamente al finalizar la época (oficial)
            orig_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
            with benchmark("Guardado de Checkpoint Oficial", log):
                orig_policy.save_pretrained(output_directory)
            log.info(f"[Checkpoint] Pesos oficiales guardados al final de Época {epoch_idx} (Paso {step})")

            # Guardar en carpeta separada cada 'save_every' épocas
            if epoch_idx % save_every == 0:
                ckpt_dir = os.path.join(output_directory, "checkpoints", f"epoch_{epoch_idx}")
                os.makedirs(ckpt_dir, exist_ok=True)
                orig_policy.save_pretrained(ckpt_dir)
                log.info(f"[Checkpoint] Checkpoint periódico guardado en: {ckpt_dir}")

    except KeyboardInterrupt:
        log.warning("Entrenamiento interrumpido manualmente (KeyboardInterrupt). Guardando estado actual...")

    finally:
        step_csv_file.close()
        epoch_csv_file.close()

    # Generación final de gráficos consolidados
    plot_training_curves(output_directory, step_history, epoch_history)

    # Save the original uncompiled policy if compiled
    orig_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    orig_policy.save_pretrained(output_directory)
    log.info(f"Modelo y métricas guardados exitosamente en: {output_directory}")

    # Hook Relacional SQLite Fail-Safe y No Intrusivo
    if not no_tracker:
        try:
            from visuomotor_core.utils.tracker import record_train_run
            total_dur = round(time.time() - overall_start_time, 2)
            final_loss = epoch_history[-1]["avg_loss"] if epoch_history else 0.0
            final_ema = step_history[-1]["loss_ema"] if step_history else 0.0
            peak_vram = max([r["vram_peak_mb"] for r in step_history]) if step_history else 0.0
            epochs_done = len(epoch_history)
            run_name = output_directory.name

            record_train_run(
                run_id=f"dp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}",
                dataset_path=str(root),
                epochs_requested=epochs,
                epochs_completed=epochs_done,
                batch_size=batch_size,
                lr=lr,
                final_loss=final_loss,
                final_loss_ema=final_ema,
                vram_peak_mb=peak_vram,
                duration_s=total_dur,
                checkpoint_dir=str(output_directory),
                metadata={
                    "horizon": cfg.horizon,
                    "n_obs_steps": cfg.n_obs_steps,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
            )
        except Exception as e:
            log.warning(f"[DB Fail-Safe] Error no crítico al registrar en SQLite: {e}")


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Train diffusion policy')
    parser.add_argument('--path', type=str, help='Data path')
    parser.add_argument('--root', type=str, default=None, help='Data path (alias for --path)')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (n_obs_steps=2 doubles the effective '
                         'image batch through the ResNet encoder; 224x224 crops OOM well before batch_size=512 on an '
                         '8GB GPU — start low and raise it while watching the printed VRAM figure)')
    parser.add_argument('--out_dir', type=str, default='outputs/train', help='Output directory')
    parser.add_argument('--save_every', type=int, default=25, help='Save a separate checkpoint every N epochs')
    parser.add_argument('--no_tracker', action='store_true', help='Desactivar registro automático en base de datos SQLite')
    parsed_args = parser.parse_args()

    data_path = parsed_args.root if parsed_args.root is not None else parsed_args.path
    main(data_path, parsed_args.epoch, parsed_args.batch_size, parsed_args.out_dir, parsed_args.save_every, parsed_args.no_tracker)


if __name__ == '__main__':
    main_cli()
