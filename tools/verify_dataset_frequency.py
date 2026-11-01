#!/usr/bin/env python3
import os
import sys
import zarr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/ros/datasets/demonstrations_headless_10ep.zarr" if os.path.exists("/ros/datasets") else "datasets/demonstrations_headless_10ep.zarr"
    root = zarr.open_group(dataset_path, mode="r")
    ep_items = sorted(list(root.groups()), key=lambda x: int(x[0].split("_")[-1]) if "_" in x[0] else x[0])
    
    print(f"=== VERIFICACIÓN DE INTEGRIDAD Y FRECUENCIA TEMPORAL: {len(ep_items)} EPISODIOS ===")
    dt = 0.050 # 20 Hz (dt = 50ms)
    
    stats = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    
    for i, (ep_name, g) in enumerate(ep_items):
        imgs = np.array(g["image"])
        states = np.array(g["state"])
        actions = np.array(g["action"])
        n = len(states)
        
        # Velocidades cartesianas reales medidas del estado del EE (dx/dt)
        pos = states[:, :3]
        velocities = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
        max_v = np.max(velocities) if len(velocities) > 0 else 0
        mean_v = np.mean(velocities) if len(velocities) > 0 else 0
        
        # Done flag
        done_frames = np.sum(actions[:, -1] >= 0.5)
        
        stats.append({
            "ep": ep_name,
            "steps": n,
            "sim_duration": n * dt,
            "max_v": max_v,
            "mean_v": mean_v,
            "done_frames": int(done_frames),
            "done_duration": done_frames * dt
        })
        
        # Plot velocity profile for each episode
        time_axis = np.arange(len(velocities)) * dt
        axes[0].plot(time_axis, velocities, label=f"Ep {i}" if i < 5 else None, alpha=0.7)
        
        # Plot Z trajectory
        axes[1].plot(np.arange(n) * dt, pos[:, 2], label=f"Ep {i}" if i < 5 else None, alpha=0.7)
        
        # Plot Done flag
        axes[2].plot(np.arange(n) * dt, actions[:, -1], label=f"Ep {i}" if i < 5 else None, alpha=0.7)

    axes[0].axhline(0.30, color='r', linestyle='--', label="Límite Paramétrico max_speed (0.30 m/s)")
    axes[0].set_ylabel("Velocidad TCP (m/s)")
    axes[0].set_title("1. Velocidad Lineal Derivada (dt = 0.050s / 20 Hz) — Certificación Cinemática")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    
    axes[1].set_ylabel("Posición Z (m)")
    axes[1].set_title("2. Perfil de Altura TCP (m) a lo largo del Tiempo Simulado")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Tiempo Simulado (s)")
    axes[2].set_ylabel("Done Flag (0 o 1)")
    axes[2].set_title("3. Señal de Parada Neuronal (Done Flag) — Piso Garantizado de 2.5s (50 frames)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ds_base = os.path.basename(dataset_path.rstrip("/")).replace(".zarr", "")
    plot_out = f"/ros/datasets/{ds_base}_frequency_verification.png"
    plt.savefig(plot_out, dpi=200)
    print(f"Gráfica guardada en: {plot_out}")
    
    print("\n| Episodio | Pasos | Duración Sim | Vel Max (m/s) | Vel Media (m/s) | Done Frames | Piso Done (s) |")
    print("|---|---|---|---|---|---|---|")
    for s in stats:
        print(f"| {s['ep']} | {s['steps']} | {s['sim_duration']:.2f}s | {s['max_v']:.4f} | {s['mean_v']:.4f} | {s['done_frames']} | {s['done_duration']:.2f}s |")

if __name__ == "__main__":
    main()
