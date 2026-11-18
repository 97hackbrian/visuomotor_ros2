"""Bringup del xArm5 REAL: driver UFACTORY + pose_smoother + puente seguro xarm5_bridge.

No lanza la política: se ejecuta aparte (visuomotor_ros2/inference.launch.py) cuando el brazo está listo.

Ejemplos::

    # 1) Sin hardware: robot simulado, nada se conecta (prueba de la cadena completa)
    ros2 launch visuomotor_bringup xarm5_real.launch.py simulate_robot_state:=true

    # 2) Primera conexión real: driver conectado, puente en dry-run (lee estado, NO mueve)
    ros2 launch visuomotor_bringup xarm5_real.launch.py robot_ip:=192.168.1.228

    # 3) Movimiento real (solo tras validar 2): dry_run:=false y luego `ros2 service call /xarm5_bridge/arm ...`
    ros2 launch visuomotor_bringup xarm5_real.launch.py dry_run:=false

Todo argumento vacío ('') toma el valor de config/xarm5/xarm5_real_params.yaml.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

GRIPPER_TYPES = ("xarm_gripper", "vacuum", "none")


def _as_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"Valor booleano inválido: {text!r}")


def _resolve(arg_value: str, yaml_value):
    return yaml_value if str(arg_value).strip() == "" else arg_value


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory("visuomotor_bringup")
    params_file = LaunchConfiguration("params_file").perform(context)
    with open(params_file, "r") as f:
        cfg = yaml.safe_load(f)
    drv = cfg["ufactory_driver"]["ros__parameters"]
    brg = cfg["xarm5_bridge"]["ros__parameters"]

    robot_ip = str(_resolve(LaunchConfiguration("robot_ip").perform(context), drv["robot_ip"]))
    gripper_type = str(_resolve(LaunchConfiguration("gripper_type").perform(context), brg["gripper_type"]))
    dry_run = _as_bool(_resolve(LaunchConfiguration("dry_run").perform(context), brg["dry_run"]))
    simulate = _as_bool(_resolve(LaunchConfiguration("simulate_robot_state").perform(context),
                                 brg["simulate_robot_state"]))
    start_driver = _as_bool(LaunchConfiguration("start_driver").perform(context)) and not simulate
    start_smoother = _as_bool(LaunchConfiguration("start_smoother").perform(context))

    if gripper_type not in GRIPPER_TYPES:
        raise ValueError(f"gripper_type={gripper_type!r} inválido. Opciones: {GRIPPER_TYPES}")
    if simulate and not dry_run:
        raise ValueError("simulate_robot_state:=true exige dry_run:=true")

    actions = []
    if start_driver:
        xarm_default_params = os.path.join(get_package_share_directory("xarm_api"), "config", "xarm_params.yaml")
        actions.append(Node(
            package="xarm_api",
            executable="xarm_driver_node",
            name="ufactory_driver",
            output="screen",
            emulate_tty=True,
            parameters=[
                xarm_default_params,   # valores por defecto del paquete oficial
                params_file,           # superposición del proyecto (servicios habilitados, report_type, ...)
                {
                    "robot_ip": robot_ip,
                    "add_gripper": gripper_type == "xarm_gripper",
                    "add_vacuum_gripper": gripper_type == "vacuum",
                },
            ],
        ))

    if start_smoother:
        actions.append(Node(
            package="visuomotor_bringup",
            executable="pose_smoother.py",
            name="pose_smoother",
            output="screen",
            parameters=[os.path.join(share, "config", "pose_smoother_params.yaml")],
        ))

    actions.append(Node(
        package="visuomotor_bringup",
        executable="xarm5_bridge.py",
        name="xarm5_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[
            params_file,
            {"dry_run": dry_run, "simulate_robot_state": simulate, "gripper_type": gripper_type},
        ],
    ))

    mode = "DRY-RUN" if dry_run else "REAL (el brazo SE MOVERÁ tras ~/arm)"
    source = "SIMULADO (sin driver)" if simulate else (f"driver -> {robot_ip}" if start_driver else "driver externo")
    summary = LogInfo(msg=f"[xarm5_real] modo={mode} | estado={source} | gripper={gripper_type}")
    return [summary] + actions


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory("visuomotor_bringup"), "config", "xarm5", "xarm5_real_params.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="YAML con secciones ufactory_driver y xarm5_bridge"),
        DeclareLaunchArgument("robot_ip", default_value="", description="IP del controlador ('' = YAML)"),
        DeclareLaunchArgument("gripper_type", default_value="",
                              description="xarm_gripper | vacuum | none ('' = YAML)"),
        DeclareLaunchArgument("dry_run", default_value="", description="true = no se envían comandos ('' = YAML)"),
        DeclareLaunchArgument("simulate_robot_state", default_value="",
                              description="true = sin driver, robot simulado ('' = YAML)"),
        DeclareLaunchArgument("start_driver", default_value="true",
                              description="Lanzar el driver xarm_api (false si ya corre aparte)"),
        DeclareLaunchArgument("start_smoother", default_value="true",
                              description="Lanzar pose_smoother (target_frame_raw -> target_frame)"),
        OpaqueFunction(function=launch_setup),
    ])
