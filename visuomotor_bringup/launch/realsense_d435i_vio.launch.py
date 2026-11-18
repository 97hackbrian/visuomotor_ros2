import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('visuomotor_bringup')
    
    gui = context.perform_substitution(LaunchConfiguration('gui')).strip().lower() in ('true', '1')
    use_ultra = context.perform_substitution(LaunchConfiguration('ultra_precision')).strip().lower() in ('true', '1')
    
    camera_config = os.path.join(pkg_share, 'config', 'realsense', 'd435i_camera_params.yaml')
    vio_config = os.path.join(pkg_share, 'config', 'realsense', 'rtabmap_vio_params.yaml')

    # 1. Driver RealSense D435i (RGB + Depth + IMU unificado)
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[camera_config],
        output='screen'
    )

    # 2. Filtro Madgwick para fusión inercial y cálculo de cuaternión AHRS
    madgwick_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False
        }],
        remappings=[
            ('imu/data_raw', '/camera/camera/imu'),
            ('imu/data', '/imu/data')
        ]
    )

    # 3. RTAB-Map Visual-Inertial Odometry (VIO)
    odom_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[vio_config],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('imu', '/imu/data'),
            ('odom', '/rtabmap/odom')
        ]
    )

    nodes = [realsense_node, madgwick_node, odom_node]

    # 4. Visor gráfico RViz2 con vo_vis.rviz
    if gui:
        rviz_config = os.path.join(pkg_share, 'rviz', 'vo_vis.rviz')
        ws_rviz = '/ros/.rviz2/vo_vis.rviz'
        if os.path.exists(ws_rviz):
            rviz_config = ws_rviz

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        )
        nodes.append(rviz_node)

    return nodes

def generate_launch_description():
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Lanzar visor gráfico RViz2 con vo_vis.rviz'
    )
    
    ultra_precision_arg = DeclareLaunchArgument(
        'ultra_precision',
        default_value='false',
        description='Modo ultra-precisión'
    )

    return LaunchDescription([
        gui_arg,
        ultra_precision_arg,
        OpaqueFunction(function=launch_setup)
    ])
