"""Solo la cámara D435i (sin odometría) para la inferencia en el robot.

Usa el MISMO archivo de parámetros que la recolección de demostraciones (config/realsense/d435i_camera_params.yaml):
color 640x480 a 30 Hz, profundidad alineada y emisor IR encendido. Así el color llega a la política procesado igual
que al grabar. Publica /camera/camera/color/image_raw, el camera_topic de deploy/jetson/config/jetson_policy_params.yaml.

El IMU no lo usa la política y se apaga por defecto (enable_imu:=true lo enciende). No cambia la imagen de color.
Si la Orin va justa de CPU, align_depth:=false quita la alineación de profundidad (tampoco cambia el color).

    ros2 launch visuomotor_bringup d435i_camera.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    share = get_package_share_directory('visuomotor_bringup')
    config = os.path.join(share, 'config', 'realsense', 'd435i_camera_params.yaml')
    as_bool = lambda name: context.perform_substitution(LaunchConfiguration(name)).strip().lower() in ('true', '1')  # noqa: E731
    imu = as_bool('enable_imu')
    overrides = {
        'enable_accel': imu,
        'enable_gyro': imu,
        'align_depth.enable': as_bool('align_depth'),
    }
    return [Node(package='realsense2_camera', executable='realsense2_camera_node', name='camera', namespace='camera',
                 parameters=[config, overrides], output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_imu', default_value='false', description='Encender acelerómetro y giroscopio'),
        DeclareLaunchArgument('align_depth', default_value='true',
                              description='Alinear profundidad al color (como en la recolección)'),
        OpaqueFunction(function=_setup),
    ])
