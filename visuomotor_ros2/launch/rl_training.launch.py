import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('visuomotor_ros2'),
        'config',
        'rl_training_params.yaml'
    )

    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='/ros/visuomotor_core/outputs/train_500_dataV7_fixed/',
        description='Ruta absoluta al checkpoint supervisado base para fine-tuning DPPO'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/ros/visuomotor_core/outputs/dppo_v7_rl_finetuned/',
        description='Directorio de salida para los checkpoints RL fine-tuneados'
    )

    rl_train_node = Node(
        package='visuomotor_ros2',
        executable='rl_train_node',
        name='rl_train_node',
        output='screen',
        parameters=[
            config,
            {
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'output_dir': LaunchConfiguration('output_dir'),
            }
        ]
    )

    return LaunchDescription([
        checkpoint_arg,
        output_dir_arg,
        rl_train_node
    ])
