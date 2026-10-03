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
        parameters=[config, {'dataset_path': LaunchConfiguration('dataset_path')}]
    )

    return LaunchDescription([
        dataset_path_arg,
        pickup_node
    ])
