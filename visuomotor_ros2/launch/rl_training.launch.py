import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('visuomotor_ros2'),
        'config',
        'rl_training_params.yaml'
    )

    rl_train_node = Node(
        package='visuomotor_ros2',
        executable='rl_train_node',
        name='rl_train_node',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        rl_train_node
    ])
