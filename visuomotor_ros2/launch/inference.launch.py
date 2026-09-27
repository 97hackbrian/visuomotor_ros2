import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('visuomotor_ros2'),
        'config',
        'policy_params.yaml'
    )

    policy_node = Node(
        package='visuomotor_ros2',
        executable='policy_node',
        name='policy_node',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        policy_node
    ])
