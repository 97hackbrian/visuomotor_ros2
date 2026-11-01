import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value='AUTO',
        description='Path to the output Zarr dataset. Use "AUTO" to auto-increment (v2, v3, etc).'
    )
    max_episodes_arg = DeclareLaunchArgument(
        'max_episodes',
        default_value='0',
        description='Max number of episodes to record before stopping (0 = unlimited).'
    )
    auto_start_arg = DeclareLaunchArgument(
        'auto_start',
        default_value='false',
        description='Auto-start collection on joint state reception (default false; true if max_episodes > 0).'
    )

    config = os.path.join(
        get_package_share_directory('visuomotor_data_collection'),
        'config',
        'data_collection_params.yaml'
    )

    pickup_node = Node(
        package='visuomotor_data_collection',
        executable='pickup_approximate',
        name='pickup_approximate_node',
        output='screen',
        parameters=[config, {
            'dataset_path': LaunchConfiguration('dataset_path'),
            'max_episodes': LaunchConfiguration('max_episodes'),
            'auto_start': LaunchConfiguration('auto_start'),
            'use_sim_time': True
        }]
    )

    return LaunchDescription([
        dataset_path_arg,
        max_episodes_arg,
        auto_start_arg,
        pickup_node
    ])
