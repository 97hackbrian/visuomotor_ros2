#!/usr/bin/env python3
"""
benchmark_jetson.py — Validación de Latencia y Memoria VRAM en Borde
-------------------------------------------------------------------
Mide el tiempo de inferencia y consumo de VRAM de la Diffusion Policy
(ResNet-18 + Spatial Softmax + 1D U-Net + DDIM) directamente en Jetson Orin Nano.

Uso dentro del contenedor:
    python3 /ros_ws/src/visuomotor_ros2/deploy/jetson/scripts/benchmark_jetson.py --model_path /ros_ws/model
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch

from visuomotor_core.models.diffusion_policy import DiffusionPolicy


def run_benchmark(model_path: str, num_steps: int = 10, iterations: int = 30):
    print("=" * 70)
    print(" BENCHMARK DE INFERENCIA VISUOMOTORA — NVIDIA JETSON ORIN NANO")
    print("=" * 70)

    # 1. Diagnóstico de Dispositivo CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capacidad de Cómputo: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA Runtime en Contenedor: {torch.version.cuda}")
    else:
        print("⚠️ ADVERTENCIA: CUDA no está disponible. La inferencia correrá en CPU.")

    # 2. Carga del Modelo
    print(f"\nCargando pesos desde: {model_path}...")
    t0 = time.perf_counter()
    policy = DiffusionPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    t_load = time.perf_counter() - t0
    print(f"✓ Modelo cargado en {t_load:.2f} s")

    # Ajustar pasos de desruidificado DDIM
    policy.config.num_inference_steps = num_steps
    print(f"Pasos DDIM configurados: {num_steps}")

    # 3. Creación de Lote Dummy de Observación
    # Batch size = 1, Obs horizon = 2, RGB 3x256x256, State 8D
    dummy_batch = {
        "observation.image": torch.zeros((1, 2, 3, 256, 256), dtype=torch.float32, device=device),
        "observation.state": torch.zeros((1, 2, 8), dtype=torch.float32, device=device),
    }

    # 4. Calentamiento (Warm-up)
    print("\nEjecutando pases de calentamiento (Warm-up)...")
    with torch.no_grad():
        for _ in range(5):
            _ = policy.generate_actions(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 5. Evaluación Comparativa: FP32 vs FP16
    eval_configs = [
        ("FP32 (Precisión Simple Estándar)", False, torch.float32),
        ("FP16 (Precisión Media Acelerada Tensor Cores)", True, torch.float16),
    ]

    print("\n" + "-" * 70)
    print(f"{'Modo de Precisión':<42} | {'Latencia (ms)':<13} | {'Frecuencia':<10} | {'VRAM Pico':<10}")
    print("-" * 70)

    for mode_name, use_amp, amp_dtype in eval_configs:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                if use_amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred_action = policy.generate_actions(dummy_batch)
                else:
                    pred_action = policy.generate_actions(dummy_batch)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000.0)

        lat_median = np.median(latencies)
        lat_p95 = np.percentile(latencies, 95)
        hz = 1000.0 / lat_median
        vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

        status = "✓ APTO 20Hz" if hz >= 20.0 else "⚠️ INFERIOR A 20Hz"
        print(f"{mode_name:<42} | {lat_median:6.2f} ± {lat_p95-lat_median:4.1f} ms | {hz:6.1f} Hz   | {vram_mb:6.1f} MB  [{status}]")

    print("-" * 70)
    print("Acción predicha shape:", pred_action.shape)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Visuomotor Policy on Jetson Orin Nano")
    parser.add_argument("--model_path", type=str, default="/ros_ws/model", help="Ruta al checkpoint")
    parser.add_argument("--steps", type=int, default=10, help="Pasos de difusión DDIM")
    parser.add_argument("--iters", type=int, default=30, help="Iteraciones de prueba")
    args = parser.parse_args()

    run_benchmark(args.model_path, args.steps, args.iters)
