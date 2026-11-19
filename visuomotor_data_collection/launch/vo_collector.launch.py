import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('visuomotor_data_collection')
    config = os.path.join(pkg_share, 'config', 'vo_data_collection_params.yaml')

    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value='AUTO',
        description='Ruta o nombre de carpeta del dataset Zarr a continuar/aumentar, o AUTO para crear demonstrations_real_vX.zarr'
    )

    collector_node = Node(
        package='visuomotor_data_collection',
        executable='vo_collector',
        name='vo_collector_node',
        output='screen',
        parameters=[config, {'dataset_path': LaunchConfiguration('dataset_path')}]
    )

    return LaunchDescription([
        dataset_path_arg,
        collector_node
    ])
