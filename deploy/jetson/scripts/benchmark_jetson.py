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

    # Ajustar pasos de desruidificado DDIM (igual que policy_node: atributo del modelo de difusión)
    policy.diffusion.num_inference_steps = num_steps
    print(f"Pasos DDIM configurados: {num_steps}")

    # 3. Lote de observación SIN normalizar, como el que arma policy_node: batch 1, n_obs = 2, RGB 3x256x256 en [0, 1]
    #    y estado 8D con un cuaternión unitario (con relative_ee la acción se recompone con el estado; un cuaternión
    #    nulo daría NaN). Se mide predict_chunk = normalización + DDIM + desnormalización + recomposición, es decir,
    #    el cómputo completo de cada inferencia de policy_node (select_action lo llama cada n_action_steps).
    state = torch.zeros((1, 2, 8), dtype=torch.float32, device=device)
    state[..., 3:7] = torch.tensor([0.0, 0.7071068, 0.0, 0.7071068], device=device)
    dummy_batch = {
        "observation.image": torch.rand((1, 2, 3, 256, 256), dtype=torch.float32, device=device),
        "observation.state": state,
    }

    # 4. Calentamiento (Warm-up)
    print("\nEjecutando pases de calentamiento (Warm-up)...")
    with torch.no_grad():
        for _ in range(5):
            _ = policy.predict_chunk(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 5. Comparación: ejecución normal frente a CUDA Graph (encoder + bucle DDIM capturados; mismo resultado).
    #    FP16 (autocast) se midió en la Orin y es MÁS lento que FP32 (los casts cuestan más de lo que ahorran): no se usa.
    eval_configs = [("FP32 normal", False), ("FP32 + CUDA Graph", True)]

    print("\n" + "-" * 70)
    print(f"{'Modo':<42} | {'Latencia (ms)':<13} | {'Frecuencia':<10} | {'VRAM Pico':<10}")
    print("-" * 70)

    for mode_name, use_graph in eval_configs:
        if use_graph:
            if device.type != "cuda" or not policy.enable_cuda_graph(batch_size=1):
                print(f"{mode_name:<42} | sin CUDA: omitido")
                continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                pred_action = policy.predict_chunk(dummy_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t_start) * 1000.0)

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
