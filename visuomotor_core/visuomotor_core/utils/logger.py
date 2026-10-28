#  Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.

import datetime
import logging
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Generator

try:
    import coloredlogs
except ImportError:
    coloredlogs = None

try:
    import humanize
except ImportError:
    humanize = None

try:
    import torch
except ImportError:
    torch = None


def configure_logging(level: int = logging.DEBUG, log_file: str | None = None) -> None:
    """
    Configura el sistema de logging científico con formato profesional enriquecido,
    estilos de color en consola y silenciado de librerías externas.
    Opcionalmente agrega un manejador de archivo para persistir logs del experimento.
    """
    if coloredlogs is not None:
        coloredlogs.install(
            level=level,
            fmt="%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d (%(process)d/%(threadName)s) - %(message)s",
            level_styles={
                "DEBUG": {"color": "black", "bright": True},
                "INFO": {"color": "green"},
                "WARNING": {"color": "magenta"},
                "ERROR": {"color": "red"},
                "CRITICAL": {"color": "red", "bold": True},
            },
            field_styles={
                "asctime": {"color": "yellow"},
                "levelname": {"bold": True},
                "name": {"color": "blue", "bold": True},
                "lineno": {"color": "magenta"},
                "process": {"color": "blue", "bold": True},
                "threadName": {"color": "cyan"},
                "message": {"color": "white"},
            },
            milliseconds=True,
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d (%(process)d/%(threadName)s) - %(message)s",
        )

    # Manejador dual a archivo si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d (%(process)d/%(threadName)s) - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    # Disminuir el nivel de loggers verbosos en librerías externas
    noisy_libraries = [
        "PIL",
        "asyncio",
        "cmdstanpy",
        "git.cmd",
        "git.util",
        "graphviz._tools",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
        "matplotlib.colorbar",
        "matplotlib.font_manager",
        "matplotlib.pyplot",
        "mlflow.store.db.utils",
        "mlflow.tracking.fluent",
        "numba.core",
        "scrapling",
        "websockets.client",
        "zarr",
        "numcodecs",
    ]
    for lib in noisy_libraries:
        logging.getLogger(lib).setLevel(logging.WARNING)


@contextmanager
def benchmark(
    operation_name: str | None = None,
    log: logging.Logger | None = None,
    track_gpu: bool = False,
) -> Generator[None, None, None]:
    """
    Gestor de contexto para medir el tiempo transcurrido en un bloque de código
    con resolución de nanosegundos y formateo legible (humanize).
    Opcionalmente monitorea el consumo de memoria pico VRAM si CUDA está disponible.
    """
    log = log or logging.getLogger(__name__)

    if track_gpu and torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_ns: int = time.perf_counter_ns()

    try:
        yield
    finally:
        elapsed_ns: int = time.perf_counter_ns() - start_ns
        delta: timedelta = datetime.timedelta(seconds=elapsed_ns / 1_000_000_000)

        if humanize is not None:
            human_time: str = humanize.precisedelta(
                delta, minimum_unit="microseconds", format="%.2f"
            )
        else:
            human_time = f"{delta.total_seconds():.4f}s"

        gpu_info = ""
        if track_gpu and torch is not None and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_info = f" | Peak VRAM: {peak_mb:.1f} MB"

        if operation_name:
            log.debug(f"⏱️ {operation_name} executed in {human_time}{gpu_info}")
        else:
            log.debug(f"⏱️ executed in {human_time}{gpu_info}")
