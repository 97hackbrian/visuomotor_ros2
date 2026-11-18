"""RViz2 para teleoperar el xArm5 REAL desde OTRO equipo (no lanza driver, puente ni robot_state_publisher).

En la Jetson debe estar corriendo ``xarm5_real.launch.py`` (driver + xarm5_bridge + robot_state_publisher +
xarm5_teleop_marker). Este equipo solo visualiza y opera el marcador; requisitos:
  * Mismo ROS_DOMAIN_ID y RMW que la Jetson (deploy/jetson/docker-compose.yml) y red con descubrimiento DDS.
  * Paquete ``xarm_description`` instalado aquí (RViz2 carga las mallas package://xarm_description/...).

Uso:  ros2 launch visuomotor_bringup xarm5_rviz.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory("visuomotor_bringup"), "rviz", "xarm5_teleop.rviz")
    return LaunchDescription([
        DeclareLaunchArgument("rviz_config", default_value=default_cfg, description="Configuración de RViz2"),
        Node(
            package="rviz2",
            executable="rviz2",
            name="xarm5_rviz",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
        ),
    ])
