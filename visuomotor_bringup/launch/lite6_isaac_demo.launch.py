import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Ejemplo: cargar la inferencia de visuomotor_ros2
    inference_launch_path = os.path.join(
        get_package_share_directory('visuomotor_ros2'),
        'launch',
        'inference.launch.py'
    )
    
    inference_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(inference_launch_path)
    )

    # TODO: Include xArm specific nodes or Isaac Sim bridge here
    
    return LaunchDescription([
        inference_launch
    ])
