import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('visuomotor_ros2')
    default_config = os.path.join(pkg_share, 'config', 'policy_params.yaml')

    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/ros/visuomotor_core/outputs/train_500_dataV7_fixed/',
        description='Ruta absoluta al checkpoint del modelo a inferir'
    )

    policy_node = Node(
        package='visuomotor_ros2',
        executable='policy_node',
        name='policy_node',
        output='screen',
        parameters=[
            default_config,
            {'checkpoint_path': LaunchConfiguration('checkpoint_path')}
        ]
    )

    return LaunchDescription([
        checkpoint_arg,
        policy_node
    ])
