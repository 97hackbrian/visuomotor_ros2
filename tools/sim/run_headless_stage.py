#!/usr/bin/env python3
"""
run_headless_stage.py — Headless Stage Execution Engine for NVIDIA Isaac Sim
=============================================================================
Orquestador headless optimizado para Isaac Sim (4.2 / 6.0) y ROS 2 Jazzy.

Especificaciones técnicas:
1. SimulationApp con headless=True y renderizador offscreen ("RayTracedLighting").
2. Reajuste en memoria de RenderProduct a 256x256 para /rgb (sin modificar .usda en disco).
3. rateLimitEnabled=False para máxima tasa de pasos de simulación en GPU.
4. Activación automática de extensión ROS 2 Bridge bajo ROS_DOMAIN_ID=42.
5. Captura limpia de señales SIGINT / SIGTERM para desmonte seguro de recursos.
6. Telemetría de rendimiento periódico (steps/s y FPS) cada 1000 pasos.
"""

import os
import sys
import time
import signal
import argparse

# ---------------------------------------------------------------------------
# 1. Configuración de Middleware DDS y Entorno ROS 2 (PREVIA a Omniverse)
# ---------------------------------------------------------------------------
os.environ["RMW_IMPLEMENTATION"] = os.getenv("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ["ROS_LOCALHOST_ONLY"] = os.getenv("ROS_LOCALHOST_ONLY", "1")
os.environ["ROS_DOMAIN_ID"] = os.getenv("ROS_DOMAIN_ID", "42")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

# ---------------------------------------------------------------------------
# 2. Argumentos de Línea de Comandos
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Lanzador Headless de Escenas Isaac Sim con ROS 2 Bridge"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="new_world_3.usda",
        help="Ruta al archivo .usda/.usd del escenario (defecto: new_world_3.usda)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Ancho en píxeles del RenderProduct para /rgb (defecto: 256)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Alto en píxeles del RenderProduct para /rgb (defecto: 256)",
    )
    parser.add_argument(
        "--resize-in-memory",
        action="store_true",
        default=False,
        help="Reajustar RenderProducts en memoria (defecto: False, ROS 2 ya redimensiona en origen)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Frecuencia de pasos para reportar telemetría steps/s (defecto: 1000)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. Resolución de Rutas de Stage
# ---------------------------------------------------------------------------
def resolve_stage_path(stage_name):
    """
    Resuelve la ubicación física del stage buscando en los directorios estándar
    dentro del contenedor Isaac Sim (/workspace/robo_imitate/...) o en Host.
    Mapea de forma transparente rutas absolutas entre el entorno Host y Docker.
    """
    candidates = [
        stage_name,
        os.path.abspath(stage_name),
    ]

    # Mapeo bidireccional de rutas Host <-> Contenedor
    host_prefix = "/home/hackbrian/Documents/gits/robo_imitate"
    container_prefix = "/workspace/robo_imitate"

    if stage_name.startswith(host_prefix):
        candidates.append(stage_name.replace(host_prefix, container_prefix, 1))
    elif stage_name.startswith(container_prefix):
        candidates.append(stage_name.replace(container_prefix, host_prefix, 1))

    base_name = os.path.basename(stage_name)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    candidates.extend([
        os.path.join(current_dir, stage_name),
        os.path.join(current_dir, base_name),
        os.path.join(container_prefix, "xarm_bringup/isaac", base_name),
        os.path.join(host_prefix, "xarm_bringup/isaac", base_name),
    ])

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return stage_name


# ---------------------------------------------------------------------------
# 4. Reajuste de RenderProducts en Memoria (Zero Disk Modification)
# ---------------------------------------------------------------------------
def resize_render_products_in_memory(stage, target_w=256, target_h=256):
    """
    Recorre el grafo USD en memoria para reajustar los RenderProducts y nodos
    de cámara OmniGraph a (target_w, target_h).
    CRÍTICO: No invoca stage.Save(), preservando 100% la integridad del .usda en disco.
    """
    try:
        from pxr import UsdRender, Gf
        has_pxr = True
        target_vec = Gf.Vec2i(target_w, target_h)
    except ImportError:
        has_pxr = False
        target_vec = (target_w, target_h)

    resized_count = 0

    for prim in stage.Traverse():
        # A. Prim RenderProduct nativo USD
        is_render_prod = False
        if has_pxr:
            try:
                if prim.IsA(UsdRender.Product):
                    is_render_prod = True
            except Exception:
                pass
        if not is_render_prod and prim.GetTypeName() == "RenderProduct":
            is_render_prod = True

        if is_render_prod:
            res_attr = prim.GetAttribute("resolution")
            if res_attr and res_attr.IsValid():
                old_res = res_attr.Get()
                res_attr.Set(target_vec)
                print(
                    f"[Isaac Sim Headless] RenderProduct reajustado en memoria: "
                    f"{prim.GetPath()} de {old_res} -> ({target_w}, {target_h})"
                )
                resized_count += 1

        # B. OmniGraph Node de creación de RenderProduct (IsaacCreateRenderProduct / CameraHelper)
        if prim.HasAttribute("inputs:width") and prim.HasAttribute("inputs:height"):
            is_match = False
            prim_name = prim.GetName().lower()
            if "render_product" in prim_name or "camera" in prim_name:
                is_match = True
            elif prim.HasAttribute("node:type"):
                type_val = str(prim.GetAttribute("node:type").Get() or "").lower()
                if "renderproduct" in type_val or "camera" in type_val:
                    is_match = True

            if is_match:
                w_attr = prim.GetAttribute("inputs:width")
                h_attr = prim.GetAttribute("inputs:height")
                old_w = w_attr.Get() if w_attr.IsValid() else None
                old_h = h_attr.Get() if h_attr.IsValid() else None
                if w_attr.IsValid():
                    w_attr.Set(target_w)
                if h_attr.IsValid():
                    h_attr.Set(target_h)
                print(
                    f"[Isaac Sim Headless] OmniGraph Prim {prim.GetPath()} reajustado en memoria: "
                    f"({old_w}x{old_h}) -> ({target_w}x{target_h})"
                )
                resized_count += 1

    return resized_count


# ---------------------------------------------------------------------------
# 5. Protección de Nodos Script OmniGraph en Memoria (Zero Disk Modification)
# ---------------------------------------------------------------------------
def patch_script_nodes_in_memory(stage):
    """
    Inspecciona y protege los nodos Script de OmniGraph en memoria RAM.
    Evita que llamen a og.Controller.attribute(..., None) previniendo segmentation faults,
    sin modificar el archivo .usda en disco.
    """
    patched = 0
    for prim in stage.Traverse():
        if prim.HasAttribute("inputs:script"):
            attr = prim.GetAttribute("inputs:script")
            if attr and attr.IsValid():
                code = attr.Get()
                if isinstance(code, str) and "outputs:linearVelocity" in code and "twist_node" in code:
                    if "if twist_node is None:" not in code:
                        safe_code = code.replace(
                            'lin = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", twist_node))',
                            'if twist_node is None:\n'
                            '        try:\n'
                            '            twist_node = og.Controller.node("/World/husky_A200_robot/ActionGraph/ros2_subscribe_twist")\n'
                            '        except Exception:\n'
                            '            twist_node = None\n'
                            '    if twist_node is None:\n'
                            '        return\n'
                            '    lin = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", twist_node))'
                        )
                        attr.Set(safe_code)
                        print(f"[Isaac Sim Headless] Prim {prim.GetPath()} script protegido en memoria contra nullptr.")
                        patched += 1
    return patched


# ---------------------------------------------------------------------------
# 6. Función Principal de Ejecución
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()
    resolved_stage = resolve_stage_path(args.stage)

    if not os.path.isfile(resolved_stage):
        print(f"[Isaac Sim Headless] [ERROR FATAL] Stage no encontrado: {resolved_stage}")
        print("Búsqueda realizada en rutas locales y montajes /workspace/robo_imitate.")
        sys.exit(1)

    print("======================================================================")
    print("      NVIDIA Isaac Sim Headless ROS 2 Runner (3x-7x Acelerado)       ")
    print("======================================================================")
    print(f"Stage objetivo:       {resolved_stage}")
    print(f"Resolución /rgb:      {args.width}x{args.height} píxeles (en memoria)")
    print(f"ROS_DOMAIN_ID:        {os.environ['ROS_DOMAIN_ID']}")
    print(f"RMW_IMPLEMENTATION:   {os.environ['RMW_IMPLEMENTATION']}")
    print(f"ROS_LOCALHOST_ONLY:   {os.environ['ROS_LOCALHOST_ONLY']}")
    print(f"Telemetría cada:      {args.log_interval} pasos")
    print("======================================================================")

    # A. Inicialización de SimulationApp
    launch_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": args.width,
        "height": args.height,
        "anti_aliasing": 0,
        "fast_shutdown": True,
    }

    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError:
            print("[ERROR FATAL] SimulationApp no pudo ser importado.")
            print("Este script debe ejecutarse con Isaac Sim python.sh:")
            print("  ./python.sh run_headless_stage.py --stage <archivo.usda>")
            sys.exit(1)

    simulation_app = SimulationApp(launch_config)

    # B. Módulos Omniverse (importados post-SimulationApp)
    import carb.settings
    import omni.kit.app
    import omni.usd
    import omni.timeline

    # C. Desactivar limitadores de tasa de fotogramas (rateLimitEnabled = False)
    settings = carb.settings.get_settings()
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    settings.set_int("/app/runLoops/main/rateLimitFrequency", 0)
    settings.set_bool("/app/runLoops/rendering/rateLimitEnabled", False)
    settings.set_int("/app/runLoops/rendering/rateLimitFrequency", 0)
    try:
        settings.set_int("/persistent/simulation/minFrameRate", 0)
    except Exception:
        pass
    print("[Isaac Sim Headless] Limitador de tasa desactivado (rateLimitEnabled=False).")

    # D. Carga de Extensión ROS 2 Bridge
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ros2_bridge_candidates = [
        "isaacsim.ros2.bridge",
        "omni.isaac.ros2_bridge",
    ]
    bridge_loaded = False
    for ext in ros2_bridge_candidates:
        try:
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)
            if ext_manager.is_extension_enabled(ext):
                print(f"[Isaac Sim Headless] Extensión ROS 2 Bridge activa: {ext}")
                bridge_loaded = True
                break
        except Exception as e:
            print(f"[Isaac Sim Headless] Intentando cargar {ext}: {e}")

    if not bridge_loaded:
        print("[Isaac Sim Headless] [AVISO] Ninguna extensión ROS 2 Bridge pudo confirmarse.")

    # E. Carga del Stage en Contexto USD
    usd_context = omni.usd.get_context()
    print(f"[Isaac Sim Headless] Abriendo stage USD: {resolved_stage} ...")
    usd_context.open_stage(resolved_stage)

    # Esperar a que el escenario termine de cargarse (con límite de seguridad de 300 actualizaciones)
    load_iter = 0
    max_load_iters = 300
    while usd_context.get_stage_loading_status()[2] > 0 and load_iter < max_load_iters:
        simulation_app.update()
        load_iter += 1

    stage = usd_context.get_stage()
    if not stage:
        print(f"[Isaac Sim Headless] [ERROR FATAL] Falló la apertura del stage: {resolved_stage}")
        simulation_app.close()
        sys.exit(1)

    print("[Isaac Sim Headless] Stage cargado exitosamente.")

    # F. Protección de Nodos Script en Memoria (Zero Disk Modification)
    count_patched = patch_script_nodes_in_memory(stage)
    if count_patched > 0:
        print(f"[Isaac Sim Headless] {count_patched} nodos Script de OmniGraph protegidos en memoria RAM.")

    # G. Reajuste Opcional de RenderProducts en memoria
    if args.resize_in_memory:
        count_resized = resize_render_products_in_memory(stage, args.width, args.height)
        print(f"[Isaac Sim Headless] Total de RenderProducts/Nodos reajustados en memoria: {count_resized}")
    else:
        print("[Isaac Sim Headless] Resizing en memoria omitido (ROS 2 ya redimensiona /rgb en origen).")

    # H. Manejo de Señales SIGINT / SIGTERM para Cierre Seguro
    running = True

    def sig_handler(signum, frame):
        nonlocal running
        print(f"\n[Isaac Sim Headless] Señal interceptada ({signum}). Iniciando parada limpia...")
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # I. Arranque de Simulación con Timeline nativo de Omniverse
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[Isaac Sim Headless] Timeline.play() activado.", flush=True)

    # Dejar asentar las físicas durante 10 pasos iniciales
    for _ in range(10):
        simulation_app.update()

    print("[Isaac Sim Headless] Simulación en ejecución activa (Timeline PLAY).")
    print("[Isaac Sim Headless] Publicando sensores y tópicos ROS 2.")

    # I. Bucle Principal con Telemetría de Steps/s cada 1000 Pasos
    step_count = 0
    last_log_time = time.perf_counter()

    try:
        while running and simulation_app.is_running():
            simulation_app.update()
            step_count += 1

            if step_count % args.log_interval == 0:
                current_time = time.perf_counter()
                elapsed = current_time - last_log_time
                steps_per_sec = args.log_interval / elapsed if elapsed > 0 else 0.0
                print(
                    f"[Isaac Sim Headless] Paso {step_count:7d} | "
                    f"Rendimiento: {steps_per_sec:6.2f} steps/s (FPS) | "
                    f"Tiempo tramo: {elapsed:5.2f}s"
                )
                last_log_time = current_time

    except Exception as exc:
        print(f"[Isaac Sim Headless] Excepción capturada en bucle: {exc}")
    finally:
        print("[Isaac Sim Headless] Deteniendo simulación...")
        try:
            timeline.stop()
        except Exception:
            pass
        simulation_app.close()
        print("[Isaac Sim Headless] SimulationApp cerrado correctamente. Recursos GPU liberados.")


if __name__ == "__main__":
    main()
