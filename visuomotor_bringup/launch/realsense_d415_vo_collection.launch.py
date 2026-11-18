import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    bringup_share = get_package_share_directory('visuomotor_bringup')
    data_coll_share = get_package_share_directory('visuomotor_data_collection')
    
    gui_arg = DeclareLaunchArgument('gui', default_value='false', description='Lanzar visor gráfico')
    ultra_precision_arg = DeclareLaunchArgument('ultra_precision', default_value='false', description='Usar modo ultra-precisión (SIFT)')
    dataset_path_arg = DeclareLaunchArgument('dataset_path', default_value='AUTO', description='Ruta de guardado Zarr')

    vo_launch_path = os.path.join(bringup_share, 'launch', 'realsense_d415_vo.launch.py')
    collector_config = os.path.join(data_coll_share, 'config', 'vo_data_collection_params.yaml')

    # 1. Bringup de cámara D415 + RTAB-Map VO
    vo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vo_launch_path),
        launch_arguments={
            'gui': LaunchConfiguration('gui'),
            'ultra_precision': LaunchConfiguration('ultra_precision')
        }.items()
    )

    # 2. Nodo recolector interactivo Zarr
    collector_node = Node(
        package='visuomotor_data_collection',
        executable='vo_collector',
        name='vo_collector_node',
        output='screen',
        parameters=[collector_config, {'dataset_path': LaunchConfiguration('dataset_path')}]
    )

    return LaunchDescription([
        gui_arg,
        ultra_precision_arg,
        dataset_path_arg,
        vo_launch,
        collector_node
    ])
