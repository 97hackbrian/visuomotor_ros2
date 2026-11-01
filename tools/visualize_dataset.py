#!/usr/bin/env python3
"""
visualize_dataset.py — Visualizador e Inspector de Demostraciones Zarr
======================================================================
Permite inspeccionar datasets Zarr generados por pickup_approximate:
- Extrae videos MP4 de las observaciones de cámara (/rgb).
- Genera gráficas de trayectoria XYZ, orientación, estado del gripper y done_flag.
- Muestra una tabla resumen de métricas para certificar la calidad de los episodios.

Uso:
  python3 tools/visualize_dataset.py --dataset datasets/test_headless_5eps/demonstrations.zarr
  python3 tools/visualize_dataset.py --dataset datasets/test_headless_5eps/demonstrations.zarr --save-video --save-plots
"""

import os
import sys
import zarr
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Inspección y Visualización de Datasets Zarr")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ruta al dataset Zarr (ej. datasets/test_headless_5eps/demonstrations.zarr)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directorio de salida para videos y gráficas (defecto: mismo directorio del dataset)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS para los videos generados (defecto: 20.0)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Exportar cada episodio como video MP4",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Generar gráficas de telemetría y trayectoria por episodio",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset)

    if not os.path.exists(dataset_path):
        print(f"[ERROR] El dataset Zarr no existe: {dataset_path}")
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(dataset_path)
    os.makedirs(out_dir, exist_ok=True)

    root = zarr.open_group(dataset_path, mode="r")
    episode_keys = sorted(list(root.groups()), key=lambda x: int(x[0].split("_")[-1]) if "_" in x[0] else x[0])

    print("======================================================================")
    print("      Inspector de Datasets Zarr: Visuomotor Generative Framework     ")
    print("======================================================================")
    print(f"Dataset:            {dataset_path}")
    print(f"Total de Episodios: {len(episode_keys)}")
    print(f"Salida multimedia:  {out_dir}")
    print("======================================================================")

    if len(episode_keys) == 0:
        print("[ADVERTENCIA] El dataset está vacío (0 episodios).")
        return

    summary_rows = []

    for idx, (ep_name, ep_group) in enumerate(episode_keys):
        images = np.array(ep_group["image"])
        states = np.array(ep_group["state"])
        actions = np.array(ep_group["action"])

        n_steps = len(images)
        duration_s = n_steps / args.fps

        has_done = actions.shape[-1] >= 9
        done_steps = int(np.sum(actions[:, -1] >= 0.5)) if has_done else 0
        done_time_s = done_steps / args.fps

        final_pos = states[-1, :3] if len(states) > 0 else np.zeros(3)
        init_pos = states[0, :3] if len(states) > 0 else np.zeros(3)

        print(
            f"  ✔ [{idx+1}/{len(episode_keys)}] {ep_name}: {n_steps} pasos ({duration_s:.1f}s) | "
            f"Init Z: {init_pos[2]:.3f}m -> Final Z: {final_pos[2]:.3f}m | "
            f"Done=1.0: {done_steps} frames ({done_time_s:.1f}s)"
        )

        summary_rows.append({
            "episode": ep_name,
            "steps": n_steps,
            "duration_s": duration_s,
            "done_steps": done_steps,
            "done_time_s": done_time_s,
            "init_xyz": init_pos,
            "final_xyz": final_pos,
        })

        # 1. Exportar Video MP4
        if args.save_video and len(images) > 0:
            video_path = os.path.join(out_dir, f"{ep_name}.mp4")
            h, w, c = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

            for frame in images:
                # Convertir de RGB (Zarr) a BGR (OpenCV)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if c == 3 else frame
                out_video.write(frame_bgr)
            out_video.release()

        # 2. Generar Gráficas de Trayectoria
        if args.save_plots and len(states) > 0 and len(actions) > 0:
            plot_path = os.path.join(out_dir, f"{ep_name}_trajectory.png")
            fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

            t = np.arange(n_steps) / args.fps

            # Subplot 1: Trayectoria Cartesiana XYZ (Estado medido)
            axs[0].plot(t, states[:, 0], label="State X", color="red", linewidth=1.5)
            axs[0].plot(t, states[:, 1], label="State Y", color="green", linewidth=1.5)
            axs[0].plot(t, states[:, 2], label="State Z", color="blue", linewidth=1.5)
            axs[0].plot(t, actions[:, 0], "--", color="red", alpha=0.6, label="Action X")
            axs[0].plot(t, actions[:, 1], "--", color="green", alpha=0.6, label="Action Y")
            axs[0].plot(t, actions[:, 2], "--", color="blue", alpha=0.6, label="Action Z")
            axs[0].set_ylabel("Posición (m)")
            axs[0].set_title(f"{ep_name} — Trayectoria Cartesiana XYZ vs Comandos")
            axs[0].grid(True, linestyle="--", alpha=0.5)
            axs[0].legend(loc="upper right", ncol=2)

            # Subplot 2: Orientación (Cuaternión de Estado)
            axs[1].plot(t, states[:, 3], label="qx", color="purple")
            axs[1].plot(t, states[:, 4], label="qy", color="orange")
            axs[1].plot(t, states[:, 5], label="qz", color="cyan")
            axs[1].plot(t, states[:, 6], label="qw", color="black")
            axs[1].set_ylabel("Cuaternión")
            axs[1].set_title("Orientación del Efector Final (qw >= 0)")
            axs[1].grid(True, linestyle="--", alpha=0.5)
            axs[1].legend(loc="upper right", ncol=4)

            # Subplot 3: Estado y Comando del Gripper
            axs[2].plot(t, states[:, 7], label="Gripper State", color="teal", linewidth=1.8)
            axs[2].plot(t, actions[:, 7], "--", label="Gripper Cmd", color="darkred", linewidth=1.5)
            axs[2].axhline(y=-0.005, color="gray", linestyle=":", label="Umbral Agarre")
            axs[2].set_ylabel("Gripper (m)")
            axs[2].set_title("Comando de Pinza vs Apertura Real")
            axs[2].grid(True, linestyle="--", alpha=0.5)
            axs[2].legend(loc="upper right")

            # Subplot 4: Bandera Neuronal Done Flag (9na dimensión)
            if has_done:
                axs[3].plot(t, actions[:, 8], label="Done Flag (Acción)", color="crimson", linewidth=2.0)
                axs[3].axhline(y=0.50, color="black", linestyle="--", label="Umbral Auto-Stop (0.50)")
                axs[3].set_ylabel("Done (0 ó 1)")
                axs[3].set_ylim(-0.1, 1.2)
                axs[3].set_title("9na Dimensión (Intención de Término de Tarea)")
                axs[3].grid(True, linestyle="--", alpha=0.5)
                axs[3].legend(loc="upper left")

            axs[3].set_xlabel("Tiempo (segundos)")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)

    print("======================================================================")
    print(f"🎉 Visualización completada para {len(summary_rows)} episodios.")
    print(f"Archivos MP4 y gráficas guardados en: {out_dir}")
    print("======================================================================")


if __name__ == "__main__":
    main()
