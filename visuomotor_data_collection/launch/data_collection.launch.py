import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('visuomotor_data_collection'),
        'config',
        'data_collection_params.yaml'
    )

    sim_collector_node = Node(
        package='visuomotor_data_collection',
        executable='sim_collector',
        name='sim_collector_node',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        sim_collector_node
    ])
