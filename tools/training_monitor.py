#!/usr/bin/env python3
"""Monitor de un entrenamiento de ``visuomotor_core/train.py``.

Muestra el progreso, la hora estimada de fin, la pérdida, las métricas de la evaluación offline de validación con sus
líneas base, el uso de GPU y RAM, y dónde están las gráficas y los checkpoints. Solo LEE la carpeta de salida: nunca
toca checkpoints ni pesos (AGENTS.md §16).

    # una vez (también para agentes; --json da un resumen legible por máquina)
    docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/training_monitor.py --run-dir visuomotor_core/outputs/<run> --once
    # panel que se refresca (p. ej. en tmux); --tz-offset para mostrar la hora local del host (el contenedor usa UTC)
    docker exec -it -u 1000 -w /ros visuomotor_ros2 python3 tools/training_monitor.py --run-dir visuomotor_core/outputs/<run> \
        --watch 30 --tz-offset "$(date +%z)"

Hora estimada: pasos restantes × tiempo por paso (mediana entre filas de ``training_metrics.csv``) + evaluaciones
restantes × su duración media (pausas largas entre filas: evaluación y checkpoint).
"""
import argparse
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

TS_FMT = "%Y-%m-%d %H:%M:%S"


def tail_lines(path: Path, n: int) -> list:
    """Últimas ``n`` líneas de un archivo de texto sin leerlo entero."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block, data = 65536, b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        return data.decode("utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def read_csv_rows(path: Path, tail: int = 0) -> list:
    """Filas de un CSV como dicts (``tail`` > 0: solo las últimas, con la cabecera de la primera línea)."""
    try:
        with open(path, newline="") as f:
            header = f.readline().strip()
            if not header:
                return []
            if tail <= 0:
                body = f.read()
            else:
                body = "\n".join(line for line in tail_lines(path, tail) if line.strip() and line.strip() != header)
        return list(csv.DictReader(io.StringIO(header + "\n" + body)))
    except OSError:
        return []


def fnum(x, default=float("nan")):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return default


def parse_ts(s):
    try:
        return datetime.strptime(s.strip(), TS_FMT)
    except (AttributeError, ValueError):
        return None


def parse_tz(s):
    if not s:
        return None
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", s.strip())
    if not m:
        raise SystemExit(f"--tz-offset inválido: {s!r} (ejemplo: -0300)")
    sign = -1 if m.group(1) == "-" else 1
    return sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))


def fmt_dur(seconds) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d} d {h} h {m:02d} min"
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {s:02d} s"


def train_processes(run_name: str) -> list:
    """PIDs de train.py cuyo comando nombra la carpeta del run (incluye los workers del DataLoader)."""
    try:
        out = subprocess.run(["pgrep", "-f", f"train.py.*{re.escape(run_name)}"], capture_output=True, text=True,
                             timeout=5).stdout
        return [int(p) for p in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def is_python(pid: int) -> bool:
    try:
        exe = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")[0]
        return b"python" in os.path.basename(exe)
    except OSError:
        return False


def rss_gb(pid: int) -> float:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return 0.0


def host_ram():
    info = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) / 1024 ** 2
        return info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0), info.get("MemTotal", 0.0)
    except (OSError, ValueError):
        return None, None


def gpu_status():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
                              "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10).stdout.strip().splitlines()[0]
        util, used, total, temp, power = [x.strip() for x in out.split(",")]
        return {"util_pct": fnum(util), "mem_used_gb": fnum(used) / 1024, "mem_total_gb": fnum(total) / 1024,
                "temp_c": fnum(temp), "power_w": fnum(power)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def timing(rows: list, epoch_rows: list = (), steps_per_epoch: int = 0):
    """(segundos por paso, sobrecosto medio de cada pausa larga = evaluación + checkpoint, n de pausas).

    Los sellos de los CSV tienen resolución de 1 s y las filas van cada ~2 s: la mediana de los intervalos queda
    cuantizada (con v8 daba 0,200 s/paso en vez de 0,178). Se usa la duración medida de las épocas completas (sin la
    primera: compilación y arranque), que excluye la evaluación, o la suma de los intervalos normales si aún no hay.
    """
    intervals = []
    prev = None
    for r in rows:
        t, s = parse_ts(r.get("timestamp", "")), fnum(r.get("step"))
        if t is None or s != s:
            continue
        if prev is not None and s > prev[1]:
            intervals.append(((t - prev[0]).total_seconds(), s - prev[1]))
        prev = (t, s)
    if len(intervals) < 3:
        return None, None, 0
    rough = statistics.median(dt / ds for dt, ds in intervals)      # solo para separar las pausas largas
    is_gap = [dt > max(10.0, 6 * rough * ds) for dt, ds in intervals]
    normal = [iv for iv, g in zip(intervals, is_gap) if not g]
    sec_per_step = sum(dt for dt, _ in normal) / max(1.0, sum(ds for _, ds in normal))
    durs = [fnum(r.get("duration_s")) for r in list(epoch_rows)[1:]]
    durs = [d for d in durs if d == d and d > 0]
    if steps_per_epoch and durs:
        sec_per_step = statistics.median(durs[-10:]) / steps_per_epoch
    gaps = [dt - sec_per_step * ds for (dt, ds), g in zip(intervals, is_gap) if g]
    return sec_per_step, (statistics.mean(gaps) if gaps else None), len(gaps)


def summarize(run_dir: Path, shown_dir: str = "") -> dict:
    run_name = run_dir.name
    shown = shown_dir or str(run_dir)
    meta = {}
    try:
        meta = json.loads((run_dir / "training_meta.json").read_text())
    except (OSError, ValueError):
        pass
    epochs = int(meta.get("epochs") or 0)
    steps_per_epoch = int(meta.get("steps_per_epoch") or 0)
    total_steps = int(meta.get("total_steps") or epochs * steps_per_epoch)
    eval_every = int(meta.get("eval_every") or meta.get("save_every") or 0)
    save_every = int(meta.get("save_every") or 0)

    try:
        with open(run_dir / "training_metrics.csv") as f:
            header = f.readline()
            first_line = f.readline()
        first = list(csv.DictReader(io.StringIO(header + first_line)))
    except OSError:
        first = []
    recent = read_csv_rows(run_dir / "training_metrics.csv", tail=4000)
    epoch_rows = read_csv_rows(run_dir / "epoch_metrics.csv")
    eval_rows = read_csv_rows(run_dir / "eval_metrics.csv")

    now = datetime.now()
    last = recent[-1] if recent else {}
    step = int(fnum(last.get("step"), 0)) + (1 if last else 0)
    epoch = int(fnum(last.get("epoch"), 0))
    completed_epochs = int(fnum(epoch_rows[-1]["epoch"], 0)) if epoch_rows else 0
    start = parse_ts(first[0]["timestamp"]) if first else None
    last_ts = parse_ts(last.get("timestamp", "")) if last else None

    sec_per_step, eval_overhead, n_gaps = timing(recent, epoch_rows, steps_per_epoch)
    remaining_steps = max(0, total_steps - step) if total_steps else None
    remaining_evals = 0
    if eval_every and epochs:
        remaining_evals = sum(1 for e in range(completed_epochs + 1, epochs + 1) if e % eval_every == 0)
    eta_s = None
    if sec_per_step and remaining_steps is not None:
        eta_s = remaining_steps * sec_per_step + remaining_evals * (eval_overhead or 0.0)
        if last_ts is not None:                      # descontar lo que ya pasó desde la última fila
            eta_s = max(0.0, eta_s - max(0.0, (now - last_ts).total_seconds()))

    pids = train_processes(run_name)
    py = [p for p in pids if is_python(p)]          # sin el "bash -c" que lanza train.py
    log_tail = "\n".join(tail_lines(run_dir / "training.log", 400))
    finished = "Modelo y métricas guardados exitosamente" in log_tail
    if pids:
        status = "STOP pedido (termina la época y guarda)" if (run_dir / "STOP").exists() else "CORRIENDO"
    elif finished:
        status = "TERMINADO"
    elif "Guardia RAM" in log_tail or (run_dir / "STOP").exists():
        status = "DETENIDO (reanudable con --resume)"
    elif not recent:
        status = "SIN DATOS (¿no arrancó?)"
    else:
        status = "CAÍDO: revisar training.log y el .stdout"

    val = [r for r in eval_rows if r.get("split") == "val"]
    trn = [r for r in eval_rows if r.get("split") == "train"]

    def best(rows, key, lower=True):
        cand = [(fnum(r.get(key)), int(fnum(r.get("epoch"), 0))) for r in rows if fnum(r.get(key)) == fnum(r.get(key))]
        if not cand:
            return None
        return (min if lower else max)(cand)

    ckpts = sorted((p for p in (run_dir / "checkpoints").glob("epoch_*") if p.is_dir()),
                   key=lambda p: int(re.sub(r"\D", "", p.name) or 0)) if (run_dir / "checkpoints").is_dir() else []
    ram_used, ram_total = host_ram()
    warn = [line for line in log_tail.splitlines() if re.search(r"\[ *(ERROR|CRITICAL)\]|Traceback|nan|NaN", line)]
    return {
        "run_dir": str(run_dir), "run_name": run_name, "status": status, "pids": pids,
        "dataset": meta.get("dataset_name"), "train_episodes": len(meta.get("train_episodes") or []),
        "val_episodes": len(meta.get("val_episodes") or []), "train_windows": meta.get("dataset_length"),
        "epochs": epochs, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "eval_every": eval_every,
        "save_every": save_every, "epoch": epoch, "completed_epochs": completed_epochs, "step": step,
        "progress": (step / total_steps) if total_steps else None,
        "start": start.strftime(TS_FMT) if start else None, "last_row": last_ts.strftime(TS_FMT) if last_ts else None,
        "now": now.strftime(TS_FMT),
        "elapsed_s": (((last_ts if (not pids and last_ts) else now) - start).total_seconds() if start else None),
        "sec_per_step": sec_per_step, "eval_overhead_s": eval_overhead, "remaining_evals": remaining_evals,
        "eta_s": eta_s, "eta": (now + timedelta(seconds=eta_s)).strftime(TS_FMT) if eta_s is not None else None,
        "loss": fnum(last.get("loss")), "loss_ema": fnum(last.get("loss_ema")), "lr": fnum(last.get("lr")),
        "vram_peak_mb": fnum(last.get("vram_peak_mb")),
        "last_epoch": epoch_rows[-1] if epoch_rows else None,
        "last_val_loss": next(((int(fnum(r["epoch"], 0)), fnum(r["val_loss"])) for r in reversed(epoch_rows)
                               if fnum(r.get("val_loss")) == fnum(r.get("val_loss"))), None),
        "val_evals": val, "train_evals": trn,
        "best_val_fde": best(val, "fde_mm"), "best_val_ade": best(val, "ade_mm"),
        "best_val_done_f1": best(val, "done_f1", lower=False),
        "checkpoints": [p.name for p in ckpts],
        "rss_gb": sum(rss_gb(p) for p in py), "rss_main_gb": rss_gb(min(py)) if py else 0.0,
        "host_ram_used_gb": ram_used, "host_ram_total_gb": ram_total, "gpu": gpu_status(),
        "warnings": warn[-5:],
        "shown_dir": shown,
        "plots": {name: f"{shown}/{rel}" for name, rel in (
            ("curvas de pérdida", "losses/training_curves.png"),
            ("evaluación offline", "evaluation/offline_eval_curves.png"),
            ("convergencia (al terminar)", "convergence/"),
            ("informe final (al terminar)", "final_report/"))},
    }


def local(ts: str, tz):
    if not ts:
        return "?"
    t = parse_ts(ts)
    if t is None:
        return ts
    if tz is not None:                                # el contenedor usa UTC: pasar a la hora del host
        t = t + tz
    return t.strftime("%Y-%m-%d %H:%M")


def render(s: dict, tz) -> str:
    L = []
    bar_w = 40
    p = s["progress"] or 0.0
    bar = "#" * int(round(p * bar_w)) + "-" * (bar_w - int(round(p * bar_w)))
    tzs = ""
    if tz is not None:
        mins = int(tz.total_seconds() // 60)
        tzs = f" (hora local {'+' if mins >= 0 else '-'}{abs(mins) // 60:02d}{abs(mins) % 60:02d})"
    L.append(f"=== {s['run_name']} === {s['status']} === {local(s['now'], tz)}{tzs}")
    L.append(f"Dataset: {s['dataset']}  ({s['train_episodes']} episodios train / {s['val_episodes']} val, "
             f"{s['train_windows']} ventanas de train)")
    L.append(f"Época {s['epoch']}/{s['epochs']} (completas {s['completed_epochs']})  ·  paso {s['step']:,}/"
             f"{s['total_steps']:,}  ·  {s['steps_per_epoch']:,} pasos por época".replace(",", " "))
    L.append(f"[{bar}] {100 * p:5.1f} %")
    if s["sec_per_step"]:
        ev = f"{fmt_dur(s['eval_overhead_s'])} por evaluación" if s["eval_overhead_s"] else "evaluación: sin medir aún"
        L.append(f"Velocidad: {1 / s['sec_per_step']:.2f} pasos/s ({s['sec_per_step']:.3f} s/paso) · época ≈ "
                 f"{fmt_dur(s['sec_per_step'] * s['steps_per_epoch'])} · {ev} (cada {s['eval_every']} épocas; "
                 f"faltan {s['remaining_evals']})")
    if s["status"] == "TERMINADO":
        L.append(f"Inicio: {local(s['start'], tz)} · terminó: {local(s['last_row'], tz)} · duró {fmt_dur(s['elapsed_s'])}")
    else:
        L.append(f"Inicio: {local(s['start'], tz)} · transcurrido {fmt_dur(s['elapsed_s'])} · "
                 f"FIN ESTIMADO: {local(s['eta'], tz)} (faltan {fmt_dur(s['eta_s'])})")
    le = s["last_epoch"]
    lv = s["last_val_loss"]
    L.append(f"Pérdida: paso {s['loss']:.4f} · EMA {s['loss_ema']:.4f}"
             + (f" · época {le['epoch']}: {fnum(le['avg_loss']):.4f}" if le else "")
             + (f" · val (denoise) {lv[1]:.4f} en la época {lv[0]}" if lv else "")
             + f" · LR {s['lr']:.2e}")
    g = s["gpu"]
    ram = (f"RAM host {s['host_ram_used_gb']:.1f}/{s['host_ram_total_gb']:.0f} GB" if s["host_ram_used_gb"] is not None
           else "RAM host ?")
    L.append(f"Recursos: {ram} · train.py principal {s['rss_main_gb']:.1f} GB (guardia --max_rss_gb) · VRAM pico "
             f"{s['vram_peak_mb'] / 1024:.1f} GB" + (f" · GPU {g['util_pct']:.0f} % {g['mem_used_gb']:.1f}/"
                                                       f"{g['mem_total_gb']:.0f} GB {g['temp_c']:.0f} °C "
                                                       f"{g['power_w']:.0f} W" if g else " · GPU ?"))
    if s["val_evals"]:
        L.append("")
        L.append("Validación (muestreo completo, episodios que no se entrenan). Más bajo es mejor, salvo F1 y progreso:")
        L.append(f"{'época':>5} {'ADE':>6} {'ADEex':>6} {'FDE':>6} {'rot°':>5} {'F1done':>6} {'progr':>5} │ "
                 f"{'quieto ADE/ex':>13} {'vel.cte ADE/ex/FDE':>19}")
        for r in s["val_evals"][-8:]:
            cv = (f"{fnum(r.get('cv_ade_mm')):.1f}/{fnum(r.get('cv_ade_exec_mm')):.1f}/{fnum(r.get('cv_fde_mm')):.1f}"
                  if r.get("cv_fde_mm") not in (None, "") else "-")
            L.append(f"{int(fnum(r['epoch'], 0)):>5} {fnum(r['ade_mm']):6.1f} {fnum(r['ade_exec_mm']):6.1f} "
                     f"{fnum(r['fde_mm']):6.1f} {fnum(r['rot_deg']):5.2f} {fnum(r['done_f1']):6.3f} "
                     f"{fnum(r['progress_ratio']):5.2f} │ {fnum(r['hold_ade_mm']):6.1f}/{fnum(r['hold_ade_exec_mm']):<6.1f} "
                     f"{cv:>19}")
        b = s["best_val_fde"]
        d = s["best_val_done_f1"]
        L.append(f"Mejor FDE de val: {b[0]:.1f} mm (época {b[1]})" if b else "Mejor FDE: -")
        if d:
            L[-1] += f" · mejor F1 del done: {d[0]:.3f} (época {d[1]})"
    L.append("")
    L.append("Gráficas (se refrescan en cada evaluación):")
    for k, v in s["plots"].items():
        L.append(f"  {k}: {v}")
    ck = s["checkpoints"]
    L.append(f"Checkpoints ({len(ck)}): {s['shown_dir']}/checkpoints/ · último: {ck[-1] if ck else '-'} "
             f"(cada {s['save_every']} épocas)")
    L.append(f"Parar limpio: touch {s['shown_dir']}/STOP · reanudar: el comando del README.md con --resume")
    if s["warnings"]:
        L.append("")
        L.append("Avisos en training.log:")
        L.extend("  " + w[-160:] for w in s["warnings"])
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="Carpeta de salida del entrenamiento (--out_dir de train.py)")
    ap.add_argument("--once", action="store_true", help="Mostrar una vez y salir")
    ap.add_argument("--watch", type=float, default=30.0, help="Segundos entre refrescos (sin --once)")
    ap.add_argument("--json", action="store_true", help="Resumen JSON (implica --once)")
    ap.add_argument("--host-root", default=os.environ.get("HOST_REPO_DIR", ""),
                    help='Raíz del repo en el host para mostrar rutas abribles, p. ej. "$PWD" (el contenedor usa /ros)')
    ap.add_argument("--tz-offset", default=os.environ.get("HOST_TZ_OFFSET", ""),
                    help='Desfase de la hora del host respecto del reloj de los CSV, p. ej. "$(date +%%z)" = -0300')
    a = ap.parse_args()
    run_dir = Path(a.run_dir).resolve()
    tz = parse_tz(a.tz_offset)
    shown = str(run_dir)
    if a.host_root:
        try:
            shown = str(Path(a.host_root) / run_dir.relative_to(Path.cwd().resolve()))
        except ValueError:
            pass
    if a.json:
        s = summarize(run_dir, shown)
        s["val_evals"] = s["val_evals"][-3:]
        s["train_evals"] = s["train_evals"][-1:]
        print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
        return
    while True:
        text = render(summarize(run_dir, shown), tz)
        if a.once:
            print(text)
            return
        sys.stdout.write("\x1b[2J\x1b[H" + text + f"\n\n(refresco cada {a.watch:.0f} s · Ctrl-b d sale de tmux sin "
                         "detener nada)\n")
        sys.stdout.flush()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
